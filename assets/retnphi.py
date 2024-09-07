"""
# RetNPhi: Byte-Level Hybrid of Phi-3.5 and RetNet

RetNPhi is an experimental architecture that transforms Phi-3.5 into a byte-level language model, incorporating RetNet-inspired mechanisms. This innovative approach enables the model to process raw byte sequences, allowing for universal file type handling.

## Key Features:

1. **Byte-Level Processing**: Operates directly on raw byte sequences, enabling universal application to any file type.
2. **RetNet Integration**: Incorporates RetNet's multi-scale exponential decay and group normalization for efficient long-range dependency modeling.
3. **Dual-mode Processing**: Supports parallel mode for efficient training and recurrent mode for inference.
4. **Selective Fine-tuning**: Trains only specific layers (e.g., token embedding, post-attention layer normalizations) while keeping most of the original Phi-3.5 weights frozen.
5. **Weight-Decomposed Low-Rank Adaptation (DoRA)**: Applies DoRA to self-attention output projections for efficient adaptation while preserving pretrained knowledge.

## Implementation Strategy:

- **Weight Reuse**: Utilizes frozen weights from the original Phi-3.5 model for most layers.
- **Flexible DoRA Application**: Allows configuration of which layers and targets to apply DoRA.
- **Configurable Architecture**: Supports both retention-based and original attention mechanisms.
- **Untied Embeddings Option**: Provides the ability to use separate input and output embeddings.

## Training and Inference:

- Implements efficient training loops with customizable learning rate schedules.
- Supports both training from scratch and fine-tuning from a checkpoint.
- Provides a generation function for text completion tasks.

## Goals:

- Explore the potential of retention-like mechanisms in a byte-level Phi architecture.
- Leverage dual-mode processing for efficient training and inference.
- Develop a universal model capable of processing any file type.

Note: This is a highly experimental implementation, designed for research and exploration rather than production use. It demonstrates the potential of combining pretrained models with novel architectures and efficient fine-tuning techniques.

Author: Josef Albers
Date: Aug 28, 2024
"""

import glob
import json
import math
import time
from datetime import datetime
from types import SimpleNamespace

import fire
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from huggingface_hub import snapshot_download
from mlx.utils import tree_flatten, tree_unflatten

from datasets import load_dataset

class Tokenizer:
    def __init__(self, file_path=None):
        if file_path is None:
            self.vocab = list(range(256))
        else:
            with open(file_path, 'r') as f:
                content = f.read().lower().encode('utf-8')
            self.vocab = sorted(set(content))
        self.vocab_size = len(self.vocab)
        self.byte_to_index = {byte: index for index, byte in enumerate(self.vocab)}
        self.index_to_byte = {index: byte for index, byte in enumerate(self.vocab)}

    def encode(self, text):
        byte_seq = text.encode('utf-8')
        return [self.byte_to_index[byte] for byte in byte_seq]

    def decode(self, indices):
        byte_seq = bytes(self.index_to_byte[index] for index in indices)
        return byte_seq.decode('utf-8', errors='ignore')

class SuRoPE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim = config.hidden_size // config.num_attention_heads
        self.original_max_position_embeddings = config.original_max_position_embeddings
        self.rope_theta = config.rope_theta
        self.scaling_factor = math.sqrt(1 + math.log(config.max_position_embeddings / config.original_max_position_embeddings) / math.log(config.original_max_position_embeddings))
        self._long_factor = mx.array(config.rope_scaling["long_factor"], dtype=mx.float32)
        self._short_factor = mx.array(config.rope_scaling["short_factor"], dtype=mx.float32)

    def __call__(self, q, k, position_ids):
        cos, sin = self._get_cos_sin(position_ids)
        q = (q * cos) + (self._rotate_half(q) * sin)
        k = (k * cos) + (self._rotate_half(k) * sin)
        return q, k

    def _get_cos_sin(self, position_ids):
        su_factor = self._short_factor
        position_ids_expanded = position_ids[:, None, :]
        inv_freq = 1.0 / (su_factor * self.rope_theta**(mx.arange(0, self.dim, 2, dtype=mx.float32) / self.dim))
        inv_freq_expanded = mx.repeat(inv_freq[None, :, None], position_ids.shape[0], axis=0)
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(0, 2, 1)
        emb = mx.concatenate([freqs, freqs], axis=-1)
        cos = mx.expand_dims(mx.cos(emb) * self.scaling_factor, axis=1)
        sin = mx.expand_dims(mx.sin(emb) * self.scaling_factor, axis=1)
        return cos, sin

    def _rotate_half(self, x):
        midpoint = x.shape[-1] // 2
        x1, x2 = x[..., :midpoint], x[..., midpoint:]
        return mx.concatenate([-x2, x1], axis=-1)

class Phi3Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim = config.hidden_size
        self.n_heads = n_heads = config.num_attention_heads
        self.n_kv_heads = n_kv_heads = config.num_key_value_heads
        self.num_hidden_layers = config.num_hidden_layers
        self.head_dim = head_dim = config.hidden_size // n_heads
        self.scale = head_dim**-0.5
        chop_1 = self.n_heads * self.head_dim
        chop_2 = chop_1 + self.n_kv_heads * self.head_dim
        self.chop = [chop_1, chop_2]
        op_size = n_heads * head_dim + 2 * (n_kv_heads * head_dim)
        self.qkv_proj = nn.Linear(dim, op_size, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)
        self.rope = SuRoPE(config)

    def __call__(self, x, position_ids, attention_mask, cache, use_recurrent_mode):
        B, L, _ = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = mx.split(qkv, self.chop, axis=-1)
        q = q.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        if cache is None:
            position_ids = mx.arange(q.shape[2], dtype=mx.float32)[None] if position_ids is None else position_ids
            q, k = self.rope(q,k,position_ids)
            mask = mx.triu(mx.full((v.shape[2], v.shape[2]), -mx.inf), k=1)
            if attention_mask is not None:
                mask += mx.where(attention_mask[:, :, None]*attention_mask[:, None, :]==1, 0, -mx.inf)
                mask = mx.expand_dims(mask, 1)
            else:
                mask = mask[None, None]
        else:
            past_k, past_v, past_p, past_m = cache
            position_ids = past_p[:,-1:]+1
            mask = mx.pad(past_m[:,:,-1:,:], ((0,0),(0,0),(0,0),(0,1)))
            q, k = self.rope(q, k, position_ids)
            k = mx.concatenate([past_k, k], axis=2)
            v = mx.concatenate([past_v, v], axis=2)
        cache = (k, v, position_ids, mask)
        w = (q * self.scale) @ k.transpose(0, 1, 3, 2)
        w += mask
        w = mx.softmax(w, axis=-1)
        o = w @ v
        o = o.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(o).astype(x.dtype), cache

class Phi3Retention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim = dim = config.hidden_size
        self.n_heads = n_heads = config.num_attention_heads
        self.n_kv_heads = n_kv_heads = config.num_key_value_heads
        self.num_hidden_layers = config.num_hidden_layers
        self.head_dim = head_dim = config.hidden_size // n_heads
        self.scale = head_dim**-0.5
        chop_1 = self.n_heads * self.head_dim
        chop_2 = chop_1 + self.n_kv_heads * self.head_dim
        self.chop = [chop_1, chop_2]
        op_size = n_heads * head_dim + 2 * (n_kv_heads * head_dim)
        self.qkv_proj = nn.Linear(dim, op_size, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)
        self.rope = SuRoPE(config)
        xmin, xmax = math.log(1 / 32), math.log(1 / 512)
        x = mx.linspace(xmin, xmax, num=n_heads)
        self._gamma =  1 - x.exp()
        self.gn = nn.GroupNorm(num_groups=head_dim, dims=-1, affine=False)

    def __call__(self, x, position_ids, attention_mask, cache, use_recurrent_mode):
        if use_recurrent_mode:
            return self.recurrent_mode(x, cache)
        B, L, _ = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = mx.split(qkv, self.chop, axis=-1)
        q = q.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        position_ids = mx.arange(q.shape[2], dtype=mx.float32)[None] if position_ids is None else position_ids
        q, k = self.rope(q,k,position_ids)
        cache = None
        w = (q * self.scale) @ k.transpose(0, 1, 3, 2)
        w = w * self._decay(L)
        o = w @ v
        o = o.transpose(0, 2, 1, 3).reshape(B*L, -1)
        o = self.gn(o).reshape(B, L, -1)
        return self.o_proj(o).astype(x.dtype), cache

    def recurrent_mode(self, x, cache):
        if cache is None:
            s = mx.zeros((1, 32, 96, 96))
            n = 0
        else:
            s, n = cache
        qkv = self.qkv_proj(x)
        q, k, v = mx.split(qkv, self.chop, axis=-1)
        q = q.reshape(1, 1, self.n_heads, -1).transpose(0, 2, 1, 3)
        k = k.reshape(1, 1, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        v = v.reshape(1, 1, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        position_ids = mx.array([[n]])
        q, k = self.rope(q,k,position_ids)
        k = k * self.scale
        s = self._gamma[None, :, None, None] * s + (k.transpose(0, 1, 3, 2) @ v)
        o = q @ s
        o = o.transpose(0, 2, 1, 3).reshape(1, -1)
        o = self.gn(o).reshape(1, 1, -1)
        o = self.o_proj(o).astype(x.dtype)
        return o, (s, n+1)

    def _decay(self, sequence_length):
        n = mx.arange(sequence_length)[:,None]
        m = mx.arange(sequence_length)[None]
        D = (self._gamma[:, None, None] ** (n-m)) * (n >= m)
        return D

class Phi3MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_up_proj = nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def __call__(self, x):
        x = self.gate_up_proj(x)
        gate, x = mx.split(x, 2, axis=-1)
        return self.down_proj(nn.silu(gate) * x)

class Phi3DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.use_retention:
            self.self_attn = Phi3Retention(config)
        else:
            self.self_attn = Phi3Attention(config)
        self.mlp = Phi3MLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, x, position_ids, attention_mask, cache, use_recurrent_mode):
        r, cache = self.self_attn(self.input_layernorm(x), position_ids, attention_mask, cache, use_recurrent_mode)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r, cache

class Phi3Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_new = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [Phi3DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, input_ids, pixel_values, image_sizes, position_ids, attention_mask, cache, use_recurrent_mode):
        x = self.embed_new(input_ids)
        cache = [None]*len(self.layers) if cache is None else cache
        for i, l in enumerate(self.layers):
            x, cache[i] = l(x, position_ids, attention_mask, cache[i], use_recurrent_mode)
        return self.norm(x), cache

class Phi3ForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = Phi3Model(config)
        if config.untie_embedding:
            self.lm_new = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            self.untie = True
        else:
            self.untie = False

    def __call__(self, input_ids, pixel_values=None, image_sizes=None, position_ids=None, attention_mask=None, cache=None, use_recurrent_mode=False):
        x, cache = self.model(input_ids, pixel_values, image_sizes, position_ids, attention_mask, cache, use_recurrent_mode)
        if self.untie:
            return self.lm_new(x), cache
        return self.model.embed_new.as_linear(x), cache

    @property
    def layers(self):
        return self.model.layers

class DoRALinear(nn.Module):
    @staticmethod
    def from_linear(linear, r, alpha, scale, dropout):
        output_dims, input_dims = linear.weight.shape
        if isinstance(linear, nn.QuantizedLinear):
            input_dims *= 32 // linear.bits
        lora_lin = DoRALinear(input_dims=input_dims, output_dims=output_dims, r=r, alpha=alpha, scale=scale, dropout=dropout)
        lora_lin.linear = linear
        return lora_lin

    def __init__(self, input_dims, output_dims, r, alpha, scale, dropout, bias=False):
        super().__init__()
        self.linear = nn.Linear(input_dims, output_dims, bias=bias)
        self.dropout = nn.Dropout(p=dropout)
        self.scale = scale * (alpha / r)
        scale = 1 / math.sqrt(input_dims)
        self.lora_a = mx.random.uniform(low=-scale, high=scale, shape=(input_dims, r))
        self.lora_b = mx.zeros(shape=(r, output_dims))
        self.m = mx.linalg.norm(self._dequantized_weight(), axis=1).astype(mx.float32)

    def _dequantized_weight(self):
        weight = self.linear.weight
        if isinstance(self.linear, nn.QuantizedLinear):
            weight = mx.dequantize(weight, self.linear.scales, self.linear.biases, self.linear.group_size, self.linear.bits)
        return weight

    def __call__(self, x):
        y = self.linear(x)
        z = (self.dropout(x) @ self.lora_a) @ self.lora_b
        z = y + (self.scale * z)
        adapted = self._dequantized_weight() + (self.scale * self.lora_b.T) @ self.lora_a.T
        denom = mx.stop_gradient(mx.linalg.norm(adapted, axis=1))
        z = (self.m / denom) * z
        return z.astype(x.dtype)

def linear_to_lora_layers(model, lora_layers, lora_targets, lora_rank, lora_scale, lora_dropout):
    if lora_layers == 'all':
        lora_layers = model.layers
    elif isinstance(lora_layers, int):
        lora_layers = model.layers[-lora_layers:]
    elif isinstance(lora_layers, list):
        lora_layers = [model.layers[i] for i in lora_layers]
    else:
        raise ValueError("Invalid type for lora_layers. Expected int (number of layers) or list (layer indices or names).")
    def to_lora(layer):
        return DoRALinear.from_linear(layer, r=lora_rank, alpha=lora_rank, scale=lora_scale, dropout=lora_dropout)
    for l in lora_layers:
        lora_layers = [(k, to_lora(m)) for k, m in l.named_modules() if k in lora_targets]
        l.update_modules(tree_unflatten(lora_layers))

def load_base_model(model_cfg, init=False):
    model_id='microsoft/Phi-3.5-mini-instruct'
    model_path = snapshot_download(model_id, allow_patterns=["*.safetensors", "config.json"])
    with open(f"{model_path}/config.json", "r") as f:
        config = json.load(f)
    config = config|model_cfg
    model_config = SimpleNamespace(**config)
    model = Phi3ForCausalLM(model_config)
    model_weight = [(k, v) for wf in glob.glob(f"{model_path}/*.safetensors") for k, v in mx.load(wf).items()]
    model.load_weights(model_weight, strict=False)
    model.set_dtype(mx.float32)
    if init:
        init_fn_embed = nn.init.normal(mean=-0.000453949, std=0.0344238)
        model.apply_to_modules(lambda k, v: v.apply(init_fn_embed) if k.endswith('embed_new') else None)
        if model_config.untie_embedding:
            init_fn_lm = nn.init.normal(mean=-0.000231743, std=0.043457)
            model.apply_to_modules(lambda k, v: v.apply(init_fn_lm) if k.endswith('lm_new') else None)
    class_predicate = lambda k, m: hasattr(m, "to_quantized") and not k.endswith('new')
    nn.quantize(model, 64, 4, class_predicate)
    mx.eval(model.parameters())
    return model

def load_model_for_training(lora_cfg, model_cfg, thaws, from_path=None):
    model = load_base_model(model_cfg, init=False)
    if from_path:
        model.load_weights(from_path, strict=False)
    model.freeze()
    if len(lora_cfg['targets']) > 1:
        linear_to_lora_layers(model, lora_layers=lora_cfg['layers'], lora_targets=lora_cfg['targets'], lora_rank=lora_cfg['rank'], lora_scale=lora_cfg['scale'], lora_dropout=lora_cfg['dropout'])
    model.apply_to_modules(lambda k, v: v.unfreeze() if any(k.endswith(t) for t in thaws) else None)
    mx.eval(model.parameters())
    # print("Trainable parameters:", [i[0] for i in tree_flatten(model.trainable_parameters())])
    model.train()
    return model

def load_model_for_inference(lora_cfg, model_cfg):
    model = load_base_model(model_cfg, init=False)
    if len(lora_cfg['targets']) > 1:
        linear_to_lora_layers(model, lora_layers=lora_cfg['layers'], lora_targets=lora_cfg['targets'], lora_rank=lora_cfg['rank'], lora_scale=lora_cfg['scale'], lora_dropout=lora_cfg['dropout'])
    _path = 'trained_retnphi.safetensors' if model_cfg['use_retention'] else 'trained_orgnphi.safetensors'
    model.load_weights(_path, strict=False)
    mx.eval(model.parameters())
    model.eval()
    return model

def generate(prompt, lora_cfg, model_cfg, max_tokens=50, verbose = True):
    model = load_model_for_inference(lora_cfg=lora_cfg, model_cfg=model_cfg)
    input_ids = mx.array(tokenizer.encode(prompt))
    if model_cfg['use_retention']:
        cache = None
        for i in input_ids:
            logits, cache = model(i[None, None], cache=cache, use_recurrent_mode=True)
    else:
        logits, cache = model(input_ids[None])
    token = mx.argmax(logits[:,-1,:], axis=-1)
    mx.eval(token, cache)
    list_tokens = token.tolist()
    for i in range(max_tokens):
        logits, cache = model(token[None], cache=cache, use_recurrent_mode=True)
        token = mx.argmax(logits[:,-1,:], axis=-1)
        mx.eval(token, cache)
        list_tokens += token.tolist()
        if tokenizer.decode(list_tokens[-2:]) == '\n\n':
            break
    output = tokenizer.decode(list_tokens)
    if verbose:
        print(f'{prompt=} + {output=}\n-> {prompt+output}')
    del model
    return output

def train_gsm(learning_rates, num_epochs, batch_size, seq_length, lora_cfg, model_cfg, thaws, take, from_path=None):
    def load_gsm_data(tokenizer, is_tiny=True):
        if is_tiny:
            data = load_dataset("TinyGSM/TinyGSM")["train"]
            data = data.take(take)
            data = data.filter(lambda x: len(x['question']) < 100 and ':' not in x['question'] and '-' not in x['question'] and "'" not in x['code'] and '\n    result =' in x['code'])
            split_point = int(len(data) * 0.8)
            train_data = data.select(range(split_point))
            eval_data = data.select(range(split_point, len(data)))
            def format_example(example):
                code_raw = example['code']
                start = code_raw.rfind('\n    """')
                if start == -1:
                    print('Wrong format to start')
                    return code_raw.strip()
                start = start + 8
                end = code_raw.rfind('\n    result =')
                if end == -1:
                    print('Wrong format to end')
                    end = len(code_raw)
                code_block = code_raw[start:end]
                code_lines = code_block.split('\n    ')
                formatted_code = '\n'.join(line for line in code_lines if line.strip())
                formatted_code = '\n' + formatted_code.strip() + '\n\n'
                result = (example['question'].strip(), formatted_code)
                return result
        else:
            dataset = load_dataset("openai/gsm8k", "main")
            train_data = dataset["train"]
            eval_data = dataset["test"]
            def format_example(example):
                return (example['question'].strip(), '\n'+example['answer'].strip()+'\n\n')
        train_formatted = [format_example(ex) for ex in train_data]
        eval_formatted = [format_example(ex) for ex in eval_data]
        return train_formatted, eval_formatted

    def create_batches(data, tokenizer, batch_size, seq_length):
        def _encode(x):
            return [tokenizer.encode(i) for i in x]
        encoded_data = [_encode(x) for x in data]
        encoded_data = [x for x in encoded_data if len(x[0]+x[1]) <= seq_length+1]
        if batch_size is None:
            batch_size = min(len(encoded_data), 64)
        else:
            encoded_data = encoded_data[:(len(encoded_data) // batch_size) * batch_size]
            np.random.shuffle(encoded_data)
        for i in range(0, len(encoded_data), batch_size):
            batch = encoded_data[i:i+batch_size]
            max_len = min(max(len(q+a)-1 for q, a in batch), seq_length)
            x_batch = []
            y_batch = []
            mask_batch = []
            for q, a in batch:
                combined = (q+a)[:max_len+1]
                x = combined[:-1]
                y = combined[1:]
                pad_length = max_len - len(x)
                x = x + [0] * pad_length
                y = y + [0] * pad_length
                mask = [False] * (len(q)-1) + [True] * (len(a)) + [False] * (pad_length)
                x_batch.append(x)
                y_batch.append(y)
                mask_batch.append(mask)
            yield mx.array(x_batch), mx.array(y_batch), mx.array(mask_batch)

    def loss_fn(model, X, y, mask):
        logits, _ = model(X)
        logits = logits.astype(mx.float32)
        ce = nn.losses.cross_entropy(logits, y, reduction='none')
        masked_loss = ce * mask
        return masked_loss.sum(), mask.sum()

    def evaluate(model, data, tokenizer, seq_length):
        model.eval()
        total_loss = 0
        total_samples = 0
        for X, y, mask in create_batches(data, tokenizer, None, seq_length):
            loss, ntoks = loss_fn(model, X, y, mask)
            total_loss += loss.item()
            total_samples += ntoks.item()
        return total_loss / total_samples if total_samples > 0 else -1

    def get_optimizer(train_data):
        num_batches_per_epoch = len(list(create_batches(train_data, tokenizer, batch_size, seq_length)))
        num_steps = num_epochs * num_batches_per_epoch
        num_warmup = num_steps // 10
        max_lr, min_lr = learning_rates
        if num_warmup > 2:
            warmup = optim.linear_schedule(min_lr*0.1, max_lr, steps=num_warmup)
            cosine = optim.cosine_decay(max_lr, num_steps - num_warmup, min_lr)
            lr_schedule = optim.join_schedules([warmup, cosine], [num_warmup])
        else:
            lr_schedule = optim.cosine_decay(max_lr, num_steps, min_lr)
        return optim.Lion(learning_rate=lr_schedule), num_steps

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f'--- {timestamp} ---')
    train_data, eval_data = load_gsm_data(tokenizer=tokenizer)
    model = load_model_for_training(lora_cfg=lora_cfg, model_cfg=model_cfg, thaws=thaws)
    optimizer, num_steps = get_optimizer(train_data)
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    mx.eval(model, optimizer)
    metrics = {
        'steps': [],
        'learning_rates': [],
        'all_train_losses': [],
        'avg_train_losses': [],
        'val_losses': [],
        'trained_toks': [],
    }
    step = 0
    trained_toks = 0
    losses = []
    tic = time.perf_counter()
    for epoch in range(num_epochs):
        for X, y, loss_mask in create_batches(data=train_data, tokenizer=tokenizer, batch_size=batch_size, seq_length=seq_length):
            model.train()
            (loss, ntoks), grads = loss_and_grad_fn(model, X, y, loss_mask)
            optimizer.update(model, grads)
            mx.eval(loss, ntoks, model, optimizer)
            losses.append(loss.item())
            trained_toks += ntoks.item()
            step += 1
            if (step % (num_steps // 30) == 0):
                avg_train_loss = np.mean(losses)
                lr = optimizer.learning_rate.item()
                print(f"{avg_train_loss:8.4f} @ {step//(num_steps//30):2}/30 w/ {lr:.2e} ({time.perf_counter() - tic:.2f} sec)")
                tic = time.perf_counter()
                metrics['steps'].append(step)
                metrics['learning_rates'].append(lr)
                metrics['all_train_losses'].extend(losses)
                metrics['avg_train_losses'].append(avg_train_loss)
                metrics['trained_toks'].append(trained_toks)
                losses = []
                trained_toks = 0
    _path = f'trained_retnphi.safetensors' if model_cfg['use_retention'] else f'trained_orgnphi.safetensors'
    mx.save_safetensors(_path, dict(tree_flatten(model.trainable_parameters())))
    log = {
        'args': {
            'learning_rates': learning_rates,
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'seq_length': seq_length,
            'lora_cfg': lora_cfg,
            'model_cfg': model_cfg,
            'thaws': thaws,
            'from_path': from_path
        },
        'metrics': metrics
    }
    with open(f'train_log_{timestamp}.json', 'w') as f:
        json.dump(log, f, indent=2)
    del model

tokenizer = Tokenizer()

def main(take=1000, layers='all', targets=["self_attn.o_proj"], thaws=['new', 'post_attention_layernorm'], rank=32, scale=0.1, dropout=0.0, lr_max=1e-4, lr_min=1e-5, num_epochs=90, batch_size=1, seq_length=256, vocab_size=256, use_retention=True, untie_embedding=True, prompt='There are 8 candies in a carton. How many candies will be in 5 cartons?'):
    lora_cfg = dict(layers=layers, targets=targets, rank=rank, scale=scale, dropout=dropout)
    model_cfg = dict(vocab_size=vocab_size, use_retention=use_retention, untie_embedding=untie_embedding)
    train_gsm(learning_rates=(lr_max, lr_min), num_epochs=num_epochs, batch_size=batch_size, seq_length=seq_length, lora_cfg=lora_cfg, model_cfg=model_cfg, thaws=thaws, take=take)
    generate(prompt=prompt, lora_cfg=lora_cfg, model_cfg=model_cfg, max_tokens=(seq_length-len(prompt)))

if __name__ == "__main__":
    main()

# Output:
# 388.7268 @  1/30 w/ 3.36e-05 (64.73 sec)
# ...
#   4.3768 @ 30/30 w/ 1.00e-05 (64.36 sec)
# prompt='There are 8 candies in a carton. How many candies will be in 5 cartons?' + output='\ncandies_in_carton = 8 \nnumber_of_cartons = 5\ntotal_no_of_candies = candies_in_carton * number_of_cartons\n\n'
# -> There are 8 candies in a carton. How many candies will be in 5 cartons?
# candies_in_carton = 8
# number_of_cartons = 5
# total_no_of_candies = candies_in_carton * number_of_cartons
