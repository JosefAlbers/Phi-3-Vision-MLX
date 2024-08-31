"""
RetNPhi: Experimental Byte-Level Phi Model with RetNet-Inspired Mechanisms

Key Features:
1. Byte-Level Processing: Operates on raw byte sequences, enabling universal application to any file type.
2. Retention-like Mechanism: Simplified version of RetNet's retention concept using single-scale exponential decay.
3. Dual-mode Processing: Supports parallel mode for efficient training and recurrent mode for inference.
4. SuRoPE: Uses Scaled and Updated Rotary Positional Embedding.
5. Phi Model Integration: Adapts RetNet concepts to fit within the Phi model structure.
6. Strategic Weight Reuse: Utilizes pretrained Phi-3.5 weights for Q, K, V projections.

Implementation Strategy:
- Weight Reuse: Frozen Q, K, V projection weights from original Phi-3.5 model.
- Selective Fine-tuning: Trains only layer normalization, embedding layers, and LM heads.
- Dual-mode Architecture: Enables efficient parallel processing during training and memory-efficient recurrent processing during inference.

Goals:
- Explore potential of retention-like mechanisms in byte-level Phi architecture
- Leverage dual-mode processing for efficient training and inference
- Investigate weight freezing and fine-tuning strategies in hybrid model designs
- Develop a universal model capable of processing any file type

Advantages:
- Tokenizer-free approach allows for more flexible processing of diverse data types
- Parallel mode enables faster training on modern hardware
- Recurrent mode allows for efficient inference, especially for long sequences

Limitations:
- Deviates significantly from both original Phi model and RetNet architecture
- Effectiveness may vary depending on specific tasks and domains

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

VOCAB_SIZE = 39
LORA_LAYERS = 'all'
LORA_TARGETS = ["self_attn.o_proj", "mlp.down_proj"]
LORA_RANK = 16
NUM_EPOCHS = 30

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
        su_factor = self._long_factor if mx.max(position_ids) > self.original_max_position_embeddings else self._short_factor
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
        # self.rope = SuRoPE(config)
        self.gamma = 0.5

    def __call__(self, x, position_ids, attention_mask, cache, use_recurrent_mode):
        if use_recurrent_mode:
            return self.recurrent_mode(x, cache)
        B, L, _ = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = mx.split(qkv, self.chop, axis=-1)
        q = q.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        # if cache is None:
        #     position_ids = mx.arange(q.shape[2], dtype=mx.float32)[None] if position_ids is None else position_ids
        #     q, k = self.rope(q,k,position_ids)
        #     mask = mx.triu(mx.full((v.shape[2], v.shape[2]), -mx.inf), k=1)
        #     if attention_mask is not None:
        #         mask += mx.where(attention_mask[:, :, None]*attention_mask[:, None, :]==1, 0, -mx.inf)
        #         mask = mx.expand_dims(mask, 1)
        # else:
        #     past_k, past_v, past_p, past_m = cache
        #     position_ids = past_p[:,-1:]+1
        #     q, k = self.rope(q, k, position_ids)
        #     k = mx.concatenate([past_k, k], axis=2)
        #     v = mx.concatenate([past_v, v], axis=2)
        #     mask = mx.pad(past_m[:,:,-1:,:], ((0,0),(0,0),(0,0),(0,1)))
        # cache = (k, v, position_ids, mask)
        cache = None
        w = (q * self.scale) @ k.transpose(0, 1, 3, 2)
        # w += mask
        w = w * self._decay(L)
        # w = mx.softmax(w, axis=-1)
        o = w @ v
        o = o.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(o).astype(qkv.dtype), cache

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
        # q, k = self.rope(q,k,position_ids)
        k = k * self.scale
        s = self.gamma * s + (k.transpose(0, 1, 3, 2) @ v)
        o = q @ s
        o = o.transpose(0, 2, 1, 3).reshape(1, 1, -1)
        o = self.o_proj(o).astype(qkv.dtype)
        return o, (s, n+1)

    def _decay(self, sequence_length):
        n = mx.arange(sequence_length)[:,None]
        m = mx.arange(sequence_length)[None]
        D = (self.gamma ** (n-m)) * (n >= m)
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
        # self.head_new = nn.Linear(config.hidden_size, VOCAB_SIZE, bias=False)

    def __call__(self, input_ids, pixel_values=None, image_sizes=None, position_ids=None, attention_mask=None, cache=None, use_recurrent_mode=False):
        x, cache = self.model(input_ids, pixel_values, image_sizes, position_ids, attention_mask, cache, use_recurrent_mode)
        # return self.lm_head(x), cache
        # return self.head_new(x), cache
        return self.model.embed_new.as_linear(x), cache

    @property
    def layers(self):
        return self.model.layers

class LoRALinear(nn.Module):
    @staticmethod
    def from_linear(linear, r, alpha, dropout, scale):
        output_dims, input_dims = linear.weight.shape
        if isinstance(linear, nn.QuantizedLinear):
            input_dims *= 32 // linear.bits
        lora_lin = LoRALinear(input_dims=input_dims, output_dims=output_dims, r=r, alpha=alpha, dropout=dropout, scale=scale)
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

    def __call__(self, x):
        y = self.linear(x)
        z = (self.dropout(x) @ self.lora_a) @ self.lora_b
        z = y + (self.scale * z)
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
        return LoRALinear.from_linear(layer, r=lora_rank, alpha=lora_rank, scale=0.1, dropout=0.0)
    for l in lora_layers:
        lora_layers = [(k, to_lora(m)) for k, m in l.named_modules() if k in lora_targets]
        l.update_modules(tree_unflatten(lora_layers))

def load_model(model_id='microsoft/Phi-3.5-mini-instruct', init=False, from_path=None):
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
    if from_path:
        model.load_weights(from_path, strict=False)
    mx.eval(model.parameters())
    model.eval()
    return model

def load_model_for_training(use_lora=True, from_path=None):
    model = load_model(init=True, from_path=from_path)
    model.set_dtype(mx.float32)
    model.freeze()
    model.apply_to_modules(lambda k, v: v.unfreeze() if (k.endswith("new") or k.endswith("norm")) else None)
    if use_lora:
        linear_to_lora_layers(model, lora_targets=LORA_TARGETS, lora_layers=LORA_LAYERS, lora_rank=LORA_RANK)
    mx.eval(model.parameters())
    # print("Trainable parameters:", [i[0] for i in tree_flatten(model.trainable_parameters())])
    model.train()
    return model

def load_model_for_inference(use_lora=True):
    model = load_model(from_path='trained_retnet.safetensors')
    if use_lora:
        linear_to_lora_layers(model, lora_targets=LORA_TARGETS, lora_layers=LORA_LAYERS, lora_rank=LORA_RANK)
        model.load_weights('trained_retnet.safetensors', strict=False)
    mx.eval(model.parameters())
    model.eval()
    return model

def generate(model, prompt='First Citi', max_tokens=10):
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
    return tokenizer.decode(list_tokens)

def train(learning_rate, num_epochs, batch_size=1, seq_length=64, from_path=None, use_lora=True):
    def load_data(file_path, train_sep=256, eval_sep=-127):
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        train_data = ''.join(lines[:train_sep])
        eval_data = ''.join(lines[eval_sep:])
        return train_data, eval_data

    def create_batches(data, batch_size, seq_length):
        starts = [i for i in range(len(data)) if i == 0 or data[i-2:i] == '\n\n']
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
        num_warmup = num_steps // 6
        min_lr = 1e-5
        if num_warmup > 1:
            warmup = optim.linear_schedule(min_lr, learning_rate, steps=num_warmup)
            cosine = optim.cosine_decay(learning_rate, num_steps, min_lr)
            lr_schedule = optim.join_schedules([warmup, cosine], [num_warmup])
        else:
            lr_schedule = optim.cosine_decay(learning_rate, num_epochs, min_lr)
        return optim.Adam(learning_rate=lr_schedule)

    def loss_fn(model, X, y):
        logits, _ = model(X)
        return nn.losses.cross_entropy(logits, y, reduction='mean')

    def evaluate(model, data, batch_size, seq_length):
        model.eval()
        total_loss = 0
        num_batches = 0
        for X, y in create_batches(data, batch_size, seq_length):
            total_loss += loss_fn(model, X, y).item()
            num_batches += 1
        return total_loss / num_batches

    train_data, val_data = load_data('input.txt')
    model = load_model_for_training()
    optimizer = get_optimizer(train_data)
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    mx.eval(model, optimizer)
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
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f} ({time.perf_counter() - tic:.2f} sec)")
        tic = time.perf_counter()
        if val_data is not None:
            val_loss = evaluate(model, val_data, batch_size, seq_length)
            print(f"Validation Loss: {val_loss:.4f} ({time.perf_counter() - tic:.2f} sec)")
    mx.save_safetensors(f'trained_retnet.safetensors', dict(tree_flatten(model.trainable_parameters())))
    # print(generate(model))
    del model

# Train byte-level tokenizer on input file
tokenizer = Tokenizer('input.txt')

# Train RetNet model
train(learning_rate=5e-4, num_epochs=NUM_EPOCHS)

# Load trained model for inference
model = load_model_for_inference()

# Test model with a prompt
prompt = 'First Citi'
output = generate(model, prompt)
print(f'{prompt=} + {output=}')
print('->', prompt+output)

# Output
# Epoch 1/30, Average Loss: 4.1366 (11.59 sec)
# Validation Loss: 3.2002 (3.23 sec)
# Epoch 2/30, Average Loss: 3.1139 (11.55 sec)
# Validation Loss: 3.1396 (3.23 sec)
# Epoch 3/30, Average Loss: 3.0642 (11.55 sec)
# Validation Loss: 3.1779 (3.23 sec)
# Epoch 4/30, Average Loss: 3.0521 (11.55 sec)
# Validation Loss: 3.1383 (3.23 sec)
# Epoch 5/30, Average Loss: 2.9916 (11.55 sec)
# Validation Loss: 3.1366 (3.23 sec)
# Epoch 6/30, Average Loss: 2.9490 (11.55 sec)
# Validation Loss: 3.3014 (3.26 sec)
# Epoch 7/30, Average Loss: 3.0182 (11.56 sec)
# Validation Loss: 3.2257 (3.23 sec)
# Epoch 8/30, Average Loss: 2.8310 (11.56 sec)
# Validation Loss: 3.2753 (3.23 sec)
# Epoch 9/30, Average Loss: 2.7679 (11.56 sec)
# Validation Loss: 3.2056 (3.24 sec)
# Epoch 10/30, Average Loss: 2.7292 (11.56 sec)
# Validation Loss: 3.2539 (3.24 sec)
# Epoch 11/30, Average Loss: 2.6357 (11.56 sec)
# Validation Loss: 3.2288 (3.23 sec)
# Epoch 12/30, Average Loss: 2.7162 (11.56 sec)
# Validation Loss: 3.1763 (3.23 sec)
# Epoch 13/30, Average Loss: 2.6720 (11.56 sec)
# Validation Loss: 3.2270 (3.23 sec)
# Epoch 14/30, Average Loss: 2.6110 (11.56 sec)
# Validation Loss: 3.3275 (3.23 sec)
# Epoch 15/30, Average Loss: 2.6408 (11.56 sec)
# Validation Loss: 3.3357 (3.24 sec)
# Epoch 16/30, Average Loss: 2.5870 (11.58 sec)
# Validation Loss: 3.2620 (3.24 sec)
# Epoch 17/30, Average Loss: 2.6187 (11.56 sec)
# Validation Loss: 3.3364 (3.23 sec)
# Epoch 18/30, Average Loss: 2.7265 (11.56 sec)
# Validation Loss: 3.2666 (3.23 sec)
# Epoch 19/30, Average Loss: 2.6852 (11.56 sec)
# Validation Loss: 3.2319 (3.24 sec)
# Epoch 20/30, Average Loss: 2.6865 (11.56 sec)
# Validation Loss: 3.2493 (3.24 sec)
# Epoch 21/30, Average Loss: 2.6392 (11.56 sec)
# Validation Loss: 3.3557 (3.24 sec)
# Epoch 22/30, Average Loss: 2.5768 (11.56 sec)
# Validation Loss: 3.3023 (3.23 sec)
# Epoch 23/30, Average Loss: 2.5089 (11.56 sec)
# Validation Loss: 3.3639 (3.23 sec)
# Epoch 24/30, Average Loss: 2.4731 (11.56 sec)
# Validation Loss: 3.4077 (3.23 sec)
# Epoch 25/30, Average Loss: 2.4682 (11.57 sec)
# Validation Loss: 3.3771 (3.24 sec)
# Epoch 26/30, Average Loss: 2.4333 (11.56 sec)
# Validation Loss: 3.4988 (3.24 sec)
# Epoch 27/30, Average Loss: 2.3989 (11.56 sec)
# Validation Loss: 3.5066 (3.23 sec)
# Epoch 28/30, Average Loss: 2.4059 (11.56 sec)
# Validation Loss: 3.4583 (3.23 sec)
# Epoch 29/30, Average Loss: 2.4148 (11.57 sec)
# Validation Loss: 3.7499 (3.23 sec)
# Epoch 30/30, Average Loss: 2.3782 (11.56 sec)
# Validation Loss: 3.7135 (3.23 sec)
# prompt='First Citi' + output='zen:\n\ne    '
# -> First Citizen:

# e
