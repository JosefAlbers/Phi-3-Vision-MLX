"""
RetNPhi: Experimental Phi Model with RetNet-Inspired Mechanisms

This script presents an initial proof-of-concept that explores the integration
of select RetNet (Retention Networks) principles into the Phi-3 model framework.
It is important to note that this is NOT a comprehensive or faithful
reproduction of RetNet. Rather, it serves as an experimental adaptation,
selectively incorporating RetNet-inspired mechanisms to investigate their
potential within the Phi-3 architecture.

Key Features and Modifications:

1. Retention-like Mechanism: Attempts to implement a simplified version of
   RetNet's retention concept, using a single-scale exponential decay.
2. Dual-mode Processing: Supports both parallel and recurrent processing modes,
   inspired by RetNet's approach, but not implemented equivalently.
3. SuRoPE: Uses Scaled and Updated Rotary Positional Embedding, which differs
   from RetNet's xPos encoding.
4. Phi Model Integration: Adapts these concepts to fit within the Phi model
   structure, resulting in a hybrid approach.

Important Discrepancies from true RetNet:

- Lacks multi-scale retention mechanism
- Missing gated multi-scale retention
- Uses different positional encoding (SuRoPE instead of xPos)
- Simplified retention state handling
- Absence of some RetNet-specific optimizations and scaling techniques

This implementation is highly experimental and deviates significantly from
both the original Phi model and the RetNet architecture. It should be
viewed as a starting point for exploration rather than a functional
alternative to either model.

Goals:

- Explore the potential of retention-like mechanisms in the Phi architecture
- Experiment with dual-mode (parallel/recurrent) processing
- Serve as a basis for further research and refinement

Note: This code is not suitable for production use and may contain
inconsistencies or suboptimal implementations. It is intended purely
for experimental purposes and as a springboard for further research.

References:
- RetNet paper: "Retentive Network: A Successor to Transformer for
  Large Language Models" (https://arxiv.org/abs/2307.08621)

Author: Josef Albers
Date: Aug 28, 2024
"""

import glob
import json
from types import SimpleNamespace
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from typing import List, Union
import math

class SuRoPE:
    def __init__(self, config):
        self.dim = config.hidden_size // config.num_attention_heads
        self.original_max_position_embeddings = config.original_max_position_embeddings
        self.rope_theta = config.rope_theta
        self.scaling_factor = math.sqrt(1 + math.log(config.max_position_embeddings / config.original_max_position_embeddings) / math.log(config.original_max_position_embeddings))
        self.long_factor = config.rope_scaling["long_factor"]
        self.short_factor = config.rope_scaling["short_factor"]

    def __call__(self, q, k, position_ids):
        cos, sin = self._get_cos_sin(position_ids)
        q = (q * cos) + (self._rotate_half(q) * sin)
        k = (k * cos) + (self._rotate_half(k) * sin)
        return q, k

    def _get_cos_sin(self, position_ids):
        su_factor = self.long_factor if mx.max(position_ids) > self.original_max_position_embeddings else self.short_factor
        position_ids_expanded = position_ids[:, None, :]
        inv_freq = 1.0 / (mx.array(su_factor, dtype=mx.float32) * self.rope_theta**(mx.arange(0, self.dim, 2, dtype=mx.float32) / self.dim))
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
        self.gamma = 0.9

    def __call__(self, x, position_ids, attention_mask, cache, use_recurrent_mode):
        if use_recurrent_mode:
            return self.recurrent_mode(x, cache)
        B, L, _ = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = mx.split(qkv, self.chop, axis=-1)
        q = q.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        if cache is None:
            position_ids = mx.arange(q.shape[2], dtype=mx.float32)[None] if position_ids is None else position_ids
            q, k = self.rope(q,k,position_ids)
        #    mask = mx.triu(mx.full((v.shape[2], v.shape[2]), -mx.inf), k=1)
        #    if attention_mask is not None:
        #        mask += mx.where(attention_mask[:, :, None]*attention_mask[:, None, :]==1, 0, -mx.inf)
        #        mask = mx.expand_dims(mask, 1)
        else:
        #    past_k, past_v, past_p, past_m = cache
            past_k, past_v, past_p = cache
            position_ids = past_p[:,-1:]+1
        #    mask = mx.pad(past_m[:,:,-1:,:], ((0,0),(0,0),(0,0),(0,1)))
            q, k = self.rope(q, k, position_ids)
            k = mx.concatenate([past_k, k], axis=2)
            v = mx.concatenate([past_v, v], axis=2)
        # cache = (k, v, position_ids, mask)
        cache = (k, v, position_ids)
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
        q, k = self.rope(q,k,position_ids)

        s = self.gamma * s + (k.transpose(0, 1, 3, 2) @ v)

        o = (q * self.scale) @ s
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
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [Phi3DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, input_ids, pixel_values, image_sizes, position_ids, attention_mask, cache, use_recurrent_mode):
        x = self.embed_tokens(input_ids)
        cache = [None]*len(self.layers) if cache is None else cache
        for i, l in enumerate(self.layers):
            x, cache[i] = l(x, position_ids, attention_mask, cache[i], use_recurrent_mode)
        return self.norm(x), cache

class Phi3ForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = Phi3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(self, input_ids, pixel_values=None, image_sizes=None, position_ids=None, attention_mask=None, cache=None, use_recurrent_mode=False):
        x, cache = self.model(input_ids, pixel_values, image_sizes, position_ids, attention_mask, cache, use_recurrent_mode)
        return self.lm_head(x), cache

def load(model_id = 'microsoft/Phi-3.5-mini-instruct'):
    model_path = snapshot_download(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    with open(f"{model_path}/config.json", "r") as f:
        config = json.load(f)
    model_config = SimpleNamespace(**config)
    model_weight = [(k, v) for wf in glob.glob(f"{model_path}/*.safetensors") for k, v in mx.load(wf).items()]
    model = Phi3ForCausalLM(model_config)
    model.load_weights(model_weight)
    mx.eval(model.parameters())
    model.eval()
    return model, tokenizer

model, tokenizer = load()

# Parallel mode
inputs = tokenizer('Hello world!', return_tensors='np')
input_ids = mx.array(inputs['input_ids'])
logits, cache = model(input_ids)
token = mx.argmax(logits[:, -1, :], axis=-1)
list_tokens = token.tolist()
for i in range(5):
    logits, cache = model(token[:,None], cache=cache)
    token = mx.argmax(logits[:, -1, :], axis=-1)
    list_tokens += token.tolist()
print(tokenizer.decode(list_tokens)) # -> ................................................ Eisen Eisenüb

# Recurrent mode
cache = None
for i in input_ids[0]:
    logits, cache = model(i[None, None], cache=cache, use_recurrent_mode=True)
    token = mx.argmax(logits[:,-1,:], axis=-1)
    # list_tokens += token.tolist()
list_tokens = token.tolist()
for i in range(5):
    logits, cache = model(token[None], cache=cache, use_recurrent_mode=True)
    token = mx.argmax(logits[:,-1,:], axis=-1)
    list_tokens += token.tolist()
print(tokenizer.decode(list_tokens)) # -> ................................ Eisen Eisen Ehr Justice
