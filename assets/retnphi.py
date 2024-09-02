"""
RetNPhi: Byte-Level Hybrid of Phi-3.5 and RetNet

RetNPhi transforms Phi-3.5 into a byte-level language model, incorporating RetNet-inspired mechanisms.
This experimental architecture operates directly on raw byte sequences, enabling universal file type processing.
It preserves most original Phi-3.5 weights, selectively fine-tuning post-normalization layers and embeddings,
while applying Low-Rank Adaptation (LoRA) to self-attention output projections. Remarkably, despite training on
merely 64 lines from Tiny Shakespeare, the model generates coherent text, demonstrating the potential of
byte-level processing, hybrid architectures, and efficient fine-tuning in advancing language model capabilities.

Key Features:
1. Byte-Level Processing: Operates on raw byte sequences, enabling universal application to any file type
2. RetNet Integration: Incorporates RetNet's multi-scale exponential decay and group normalization for efficient
   long-range dependency modeling
3. Dual-mode Processing: Supports parallel mode for efficient training and recurrent mode for inference

Implementation Strategy:
- Weight Reuse: Frozen weights from original Phi-3.5 model for most layers
- Selective Fine-tuning: Trains only the token embedding layer, post-attention layer normalizations,
  and self-attention output projections (LoRA)
- LoRA Application: Applies LoRA to self-attention output projections for efficient adaptation while preserving
  pretrained knowledge

Results:
- Successfully produces coherent text after training on a small dataset (60 epochs on 64 lines of text)
- Demonstrates the potential of combining pretrained models with novel architectures for efficient fine-tuning

Goals:
- Explore potential of retention-like mechanisms in byte-level Phi architecture
- Leverage dual-mode processing for efficient training and inference
- Develop a universal model capable of processing any file type

Note: This is a highly experimental implementation, not a functional alternative to either model.

Author: Josef Albers
Date: Aug 28, 2024
"""

import glob
import json
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from huggingface_hub import snapshot_download
from mlx.utils import tree_flatten, tree_unflatten
import time
import math
import fire

VOCAB_SIZE = 39

class Tokenizer:
    def __init__(self, file_path='input.txt'):
        with open(file_path, 'r') as f:
            content = f.read().lower().encode('utf-8')
        self.vocab = sorted(set(content))
        self.vocab_size = len(self.vocab)
        self.byte_to_index = {byte: index for index, byte in enumerate(self.vocab)}
        self.index_to_byte = {index: byte for index, byte in enumerate(self.vocab)}

    def encode(self, text):
        byte_seq = text.lower().encode('utf-8')
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
        # su_factor = self._long_factor if mx.max(position_ids) > self.original_max_position_embeddings else self._short_factor
        su_factor = self._short_factor
        position_ids_expanded = position_ids[:, None, :]
        inv_freq = 1.0 / (su_factor * self.rope_theta**(mx.arange(0, self.dim, 2, dtype=mx.float32) / self.dim))
        inv_freq_expanded = mx.repeat(inv_freq[None, :, None], position_ids.shape[0], axis=0)
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(0, 2, 1)
        emb = mx.concatenate([freqs, freqs], axis=-1)
        cos = mx.expand_dims(mx.cos(emb) * self.scaling_factor, axis=1)
        sin = mx.expand_dims(mx.sin(emb) * self.scaling_factor, axis=1)
        return cos, sin

    @staticmethod
    def _rotate_half(x):
        midpoint = x.shape[-1] // 2
        x1, x2 = x[..., :midpoint], x[..., midpoint:]
        return mx.concatenate([-x2, x1], axis=-1)

class Phi3Attention(nn.Module):
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
        self._gn = nn.GroupNorm(num_groups=n_heads, dims=-1, affine=False)

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
        o = o.transpose(0, 2, 1, 3).reshape(B, L, -1)
        o = self._gn(o.reshape(B*L, -1)).reshape(B, L, -1)
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
        o = o.transpose(0, 2, 1, 3).reshape(1, 1, -1)
        o = self._gn(o.reshape(1*1, -1)).reshape(1, 1, -1)
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
        # self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_new = nn.Embedding(VOCAB_SIZE, config.hidden_size)
        self.layers = [Phi3DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, input_ids, pixel_values, image_sizes, position_ids, attention_mask, cache, use_recurrent_mode):
        # x = self.embed_tokens(input_ids)
        x = self.embed_new(input_ids)
        cache = [None]*len(self.layers) if cache is None else cache
        for i, l in enumerate(self.layers):
            x, cache[i] = l(x, position_ids, attention_mask, cache[i], use_recurrent_mode)
        return self.norm(x), cache

class Phi3ForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = Phi3Model(config)
        # self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(self, input_ids, pixel_values=None, image_sizes=None, position_ids=None, attention_mask=None, cache=None, use_recurrent_mode=False):
        x, cache = self.model(input_ids, pixel_values, image_sizes, position_ids, attention_mask, cache, use_recurrent_mode)
        # return self.lm_head(x), cache
        return self.model.embed_new.as_linear(x), cache

    @property
    def layers(self):
        return self.model.layers

class DoRALinear(nn.Module):
    """ For linears without biases """
    @staticmethod
    def from_linear(linear, r, alpha, dropout, scale):
        output_dims, input_dims = linear.weight.shape
        if isinstance(linear, nn.QuantizedLinear):
            input_dims *= 32 // linear.bits
        lora_lin = DoRALinear(input_dims=input_dims, output_dims=output_dims, r=r, alpha=alpha, dropout=dropout, scale=scale)
        lora_lin.linear = linear
        return lora_lin

    def __init__(self, input_dims, output_dims, r, alpha, dropout, scale, bias=False):
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

def linear_to_lora_layers(model, lora_targets, lora_layers, lora_rank):
    if lora_layers == 'all':
        lora_layers = model.layers
    elif isinstance(lora_layers, int):
        lora_layers = model.layers[-lora_layers:]
    elif isinstance(lora_layers, list):
        lora_layers = [model.layers[i] for i in lora_layers]
    else:
        raise ValueError("Invalid type for lora_layers. Expected int (number of layers) or list (layer indices or names).")
    def to_lora(layer):
        return DoRALinear.from_linear(layer, r=lora_rank, alpha=lora_rank, scale=0.1, dropout=0.0)
    for l in lora_layers:
        lora_layers = [(k, to_lora(m)) for k, m in l.named_modules() if k in lora_targets]
        l.update_modules(tree_unflatten(lora_layers))

def load_base_model(init=False):
    model_id='microsoft/Phi-3.5-mini-instruct'
    model_path = snapshot_download(model_id, allow_patterns=["*.safetensors", "config.json"])
    with open(f"{model_path}/config.json", "r") as f:
        config = json.load(f)
    model_config = SimpleNamespace(**config)
    model = Phi3ForCausalLM(model_config)
    if init:
        init_fn = nn.init.glorot_uniform()
        model.apply_to_modules(lambda k, v: v.apply(init_fn) if k.endswith('new') else None)
    model_weight = [(k, v) for wf in glob.glob(f"{model_path}/*.safetensors") for k, v in mx.load(wf).items()]
    model.load_weights(model_weight, strict=False)
    model.set_dtype(mx.float32)
    class_predicate = lambda _, m: hasattr(m, "to_quantized") and not isinstance(m, nn.Embedding)
    nn.quantize(model, 64, 4, class_predicate)
    mx.eval(model.parameters())
    return model

def load_model_for_training(lora_cfg, from_path=None):
    model = load_base_model(init=False)
    if from_path:
        model.load_weights(from_path, strict=False)
    model.freeze()
    if len(lora_cfg['targets']) > 1:
        linear_to_lora_layers(model, lora_targets=lora_cfg['targets'], lora_layers=lora_cfg['layers'], lora_rank=lora_cfg['rank'] )
    model.apply_to_modules(lambda k, v: v.unfreeze() if (k.endswith("new") or k.endswith("post_attention_layernorm")) else None)
    mx.eval(model.parameters())
    # print("Trainable parameters:", [i[0] for i in tree_flatten(model.trainable_parameters())])
    model.train()
    return model

def load_model_for_inference(lora_cfg):
    model = load_base_model(init=False)
    if len(lora_cfg['targets']) > 1:
        linear_to_lora_layers(model, lora_targets=lora_cfg['targets'], lora_layers=lora_cfg['layers'], lora_rank=lora_cfg['rank'] )
    model.load_weights('trained_retnet_long.safetensors', strict=False)
    mx.eval(model.parameters())
    model.eval()
    return model

def generate(model, prompt='First Citi', max_tokens=50, verbose = True):
    model.eval()
    input_ids = mx.array(tokenizer.encode(prompt))
    cache = None
    for i in input_ids:
        logits, cache = model(i[None, None], cache=cache, use_recurrent_mode=True)
        token = mx.argmax(logits[:,-1,:], axis=-1)
    list_tokens = token.tolist()
    for i in range(max_tokens):
        logits, cache = model(token[None], cache=cache, use_recurrent_mode=True)
        token = mx.argmax(logits[:,-1,:], axis=-1)
        list_tokens += token.tolist()
    output = tokenizer.decode(list_tokens)
    if verbose:
        print(f'{prompt=} + {output=}\n-> {prompt+output}')
    return output

def train(lora_cfg, learning_rate, num_epochs, batch_size=1, seq_length=64, from_path=None):
    def load_data(file_path):
        sep = -(seq_length)
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        train_sep = seq_length if num_epochs > 3 else -seq_length
        train_data = '\n\n' + ''.join(lines[:train_sep])
        eval_data = ''.join(lines[-seq_length:])
        return train_data, eval_data

    def create_batches(data, batch_size, seq_length):
        starts = [i for i in range(len(data)) if data[i-2:i] == '\n\n']
        sequences = [data[i:i+seq_length + 1] for i in starts if i + seq_length + 1 <= len(data)]
        np.random.shuffle(sequences)
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i+batch_size]
            batch = [tokenizer.encode(seq)[:seq_length+1] for seq in batch]
            x = [seq[:-1] for seq in batch]
            y = [seq[1:] for seq in batch]
            yield mx.array(x), mx.array(y)

    def get_optimizer(train_data):
        num_batches_per_epoch = len(list(create_batches(train_data, batch_size, seq_length)))
        num_steps = num_epochs * num_batches_per_epoch
        num_warmup = num_steps // 10
        min_lr = 5e-5
        if num_warmup > 2:
            warmup = optim.linear_schedule(min_lr, learning_rate, steps=num_warmup)
            cosine = optim.cosine_decay(learning_rate, num_steps - num_warmup, min_lr)
            lr_schedule = optim.join_schedules([warmup, cosine], [num_warmup])
        else:
            lr_schedule = optim.cosine_decay(learning_rate, num_steps, min_lr)
        return optim.Lion(learning_rate=lr_schedule)

    def loss_fn(model, X, y):
        logits, _ = model(X)
        weights = mx.ones_like(y)
        weights[:,:10] = mx.linspace(0,1.0,num=10)[None]
        return nn.losses.cross_entropy(logits, y, reduction='mean', weights=weights)

    def evaluate(model, data, batch_size, seq_length):
        model.eval()
        total_loss = 0
        num_batches = 0
        for X, y in create_batches(data, 16, seq_length):
            total_loss += loss_fn(model, X, y).item()
            num_batches += 1
        return total_loss / num_batches

    train_data, val_data = load_data('input.txt')
    model = load_model_for_training(lora_cfg)
    optimizer = get_optimizer(train_data)
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    mx.eval(model, optimizer)
    step = 0
    for epoch in range(num_epochs):
        model.train()
        tic = time.perf_counter()
        total_loss = 0
        num_batches = 0
        for X, y in create_batches(train_data, batch_size, seq_length):
            loss, grads = loss_and_grad_fn(model, X, y)
            optimizer.update(model, grads)
            mx.eval(loss, model, optimizer)
            total_loss += loss.item()
            num_batches += 1
            step += 1
            if (step % 30 == 0) and (num_epochs < 4):
                avg_train_loss = total_loss / num_batches
                val_loss = evaluate(model, val_data, batch_size, seq_length)
                print(f"-  Step {step} LR: {optimizer.learning_rate.item():.5f} Loss: {avg_train_loss:.4f} ({val_loss:.4f})")
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f} ({time.perf_counter() - tic:.2f} sec)")
        tic = time.perf_counter()
        if val_data is not None:
            val_loss = evaluate(model, val_data, batch_size, seq_length)
            print(f"Validation Loss: {val_loss:.4f} ({time.perf_counter() - tic:.2f} sec)")
    mx.save_safetensors(f'trained_retnet_long.safetensors', dict(tree_flatten(model.trainable_parameters())))
    del model

tokenizer = Tokenizer('input.txt')

def main(layers='all', targets=["self_attn.o_proj"], rank=32, lr=5e-4, epochs=60):
    lora_cfg = dict(layers=layers, targets=targets, rank=rank)
    train(lora_cfg=lora_cfg, learning_rate=lr, num_epochs=epochs)
    model = load_model_for_inference(lora_cfg=lora_cfg)
    generate(model, 'First Citi')

if __name__ == "__main__":
    fire.Fire(main)

# Output
# Epoch 1/60, Average Loss: 3.8745 (4.21 sec)
# Epoch 2/60, Average Loss: 3.4078 (4.28 sec)
# Epoch 3/60, Average Loss: 3.1443 (4.25 sec)
# ...
# Epoch 58/60, Average Loss: 0.0717 (4.27 sec)
# Epoch 59/60, Average Loss: 0.0713 (4.27 sec)
# Epoch 60/60, Average Loss: 0.0659 (4.26 sec)

# prompt='First Citi' + output='zen:\nyou are all resolved rather to die than to fam'
# -> First Citizen:
# you are all resolved rather to die than to fam
