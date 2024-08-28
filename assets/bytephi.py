"""
BytePhi: Experimental Universal Byte-Level Phi-3 Variant with RNN Integration

This script presents an exploratory architectural prototype that reimagines
the Phi-3 model as a universal byte-level processor, inspired by recent
advancements in byte-level models like bGPT. By operating directly on raw
bytes, this implementation eliminates the need for a tokenizer and extends
its potential application beyond text to any file type, aligning with the
concept of "Digital World Simulators" introduced in the bGPT paper.

Key Features and Modifications:

1. Universal Byte-Level Processing: Works directly with raw byte sequences,
   enabling potential application to any file type, not just text.
2. Tokenizer-Free Approach: Bypasses traditional tokenization, allowing for
   more flexible and potentially more efficient processing.
3. RNN Integration: Substitutes the self-attention mechanism with LSTM layers,
   exploring synergies between recurrent architectures and the Phi model.
4. Simplified Architecture: Streamlines certain components of the Phi-3 model
   to focus on byte-level and RNN experiments.

It's important to note that this implementation represents a significant
departure from the original Phi-3 architecture. It serves as a proof-of-concept
and a starting point for further research rather than a refined or optimized model.

Goals:

- The feasibility of a universal model capable of processing any file type,
  similar to bGPT's approach to simulating the digital world.
- Potential advantages of bypassing traditional tokenization in language models.
- Synergies between RNN characteristics and the Phi model's architecture.

Note: This code is not suitable for production use and may contain
inconsistencies or suboptimal implementations. It is intended purely
for experimental purposes and as a springboard for further research.

References:
- bGPT paper: "Beyond Language Models: Byte Models are Digital World Simulators"
  (https://arxiv.org/abs/2402.19155)

Author: Josef Albers
Date: Aug 28, 2024
"""

import glob
import json
import math
from types import SimpleNamespace
from typing import List, Union

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from huggingface_hub import snapshot_download
from mlx.utils import tree_flatten
from transformers import AutoTokenizer
import time

RNN_SIZE = 2
VCB_SIZE = 128

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
        # self.self_attn = Phi3Attention(config)
        self.rnn = nn.LSTM(input_size=config.hidden_size, hidden_size=RNN_SIZE, bias=False)
        self.proj_rnn = nn.Linear(RNN_SIZE, config.hidden_size, bias=False)
        self.mlp = Phi3MLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, x, position_ids, attention_mask, cache):
        r, cache = self.rnn(self.input_layernorm(x), cache) #, position_ids, attention_mask, cache)
        r = self.proj_rnn(r)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r, cache

class Phi3Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        # self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_rnn = nn.Embedding(VCB_SIZE, config.hidden_size)
        self.layers = [Phi3DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, input_ids, pixel_values, image_sizes, position_ids, attention_mask, cache):
        # x = self.embed_tokens(input_ids)
        x = self.embed_rnn(input_ids)
        cache = [None]*len(self.layers) if cache is None else cache
        for i, l in enumerate(self.layers):
            x, cache[i] = l(x, position_ids, attention_mask, cache[i])
        return self.norm(x), cache

class Phi3ForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = Phi3Model(config)
        # self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_rnn = nn.Linear(config.hidden_size, VCB_SIZE, bias=False)

    def __call__(self, input_ids, pixel_values=None, image_sizes=None, position_ids=None, attention_mask=None, cache=None):
        x, cache = self.model(input_ids, pixel_values, image_sizes, position_ids, attention_mask, cache)
        # return self.lm_head(x), cache
        return self.lm_rnn(x), cache

def load_model(model_id='microsoft/Phi-3.5-mini-instruct', adapter_path=None, init=False):
    model_path = snapshot_download(model_id)
    with open(f"{model_path}/config.json", "r") as f:
        config = json.load(f)
    model_config = SimpleNamespace(**config)
    model = Phi3ForCausalLM(model_config)

    model_weight = [(k, v) for wf in glob.glob(f"{model_path}/*.safetensors") for k, v in mx.load(wf).items()]
    model.load_weights(model_weight, strict=False)
    if adapter_path:
        model.load_weights(adapter_path, strict=False)
    elif init:
        init_fn = nn.init.glorot_uniform()
        model.apply_to_modules(lambda k, v: v.apply(init_fn) if k.endswith('rnn') else None)
    mx.eval(model.parameters())
    model.eval()
    return model

def generate(model, prompt='Hello world!'):
    model.eval()
    prompt = prompt.lower()
    input_ids = mx.array(list(prompt.encode('utf-8')))[None]
    logits, cache = model(input_ids)
    # print('hey', logits.shape)
    token = mx.argmax(logits[:, -1, :], axis=-1)
    list_tokens = token.tolist()
    for i in range(10):
        logits, cache = model(token[:,None], cache)
        # print('hoy', logits.shape)
        token = mx.argmax(logits[:, -1, :], axis=-1)
        list_tokens += token.tolist()
    output = bytes(list_tokens).decode('utf-8', errors='ignore')
    return output

def train(learning_rate=1e-3, num_epochs=10, batch_size=32, seq_length=64):
    def load_data(file_path, train_sep=512, eval_sep=-127, encoding='utf-8'):
        with open(file_path, 'r', encoding=encoding) as f:
            lines = f.readlines()
        train_data = ''.join(lines[:train_sep]).lower()
        eval_data = ''.join(lines[eval_sep:]).lower()
        return train_data, eval_data

    def create_batches(data, batch_size, seq_length, encoding='utf-8'):
        starts = [i for i in range(len(data)) if i == 0 or data[i-2:i] == '\n\n']
        sequences = [data[i:i+seq_length + 1] for i in starts if i + seq_length + 1 <= len(data)]
        np.random.shuffle(sequences)
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i+batch_size]
            # print('batch:', batch)
            batch = [list(seq.encode(encoding))[:seq_length+1] for seq in batch]
            # print('batch encoded:', batch)
            x = [seq[:-1] for seq in batch]
            # print('x:', x)
            y = [seq[1:] for seq in batch]
            # print('y:', y)
            yield mx.array(x), mx.array(y)

    def loss_fn(model, X, y):
        logits, _ = model(X)
        return mx.mean(nn.losses.cross_entropy(logits, y))

    def evaluate(model, data, batch_size, seq_length):
        model.eval()
        total_loss = 0
        num_batches = 0
        for X, y in create_batches(data, batch_size, seq_length):
            logits, _ = model(X)
            loss = mx.mean(nn.losses.cross_entropy(logits, y))
            total_loss += loss.item()
            num_batches += 1
        return total_loss / num_batches

    model = load_model(init=True)
    train_data, val_data = load_data('input.txt')
    model.set_dtype(mx.float32)
    model.freeze()
    model.apply_to_modules(lambda k, v: v.unfreeze() if k.endswith("rnn") else None)
    mx.eval(model.parameters())
    # print("Trainable parameters:", [i[0] for i in tree_flatten(model.trainable_parameters())])
    model.train()
    optimizer = optim.Adam(learning_rate=learning_rate)
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    for epoch in range(num_epochs):
        model.train()
        tic = time.perf_counter()
        model.train()
        total_loss = 0
        num_batches = 0
        for X, y in create_batches(train_data, batch_size, seq_length):
            loss, grads = loss_and_grad_fn(model, X, y)
            optimizer.update(model, grads)
            mx.eval(loss, model.state, optimizer.state)
            total_loss += loss.item()
            num_batches += 1
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f} ({time.perf_counter() - tic:.2f} sec)")
        tic = time.perf_counter()
        if val_data is not None:
            val_loss = evaluate(model, val_data, batch_size, seq_length)
            print(f"Validation Loss: {val_loss:.4f} ({time.perf_counter() - tic:.2f} sec)")
    mx.save_safetensors(f'trained.safetensors', dict(tree_flatten(model.trainable_parameters())))
    seed_text = "To be or not to be, "
    generated_text = generate(model, seed_text)
    print(f"Generated text:\n{seed_text}{generated_text}")

train(learning_rate=1e-3, num_epochs=6, batch_size = 1, seq_length = 64)
model = load_model(adapter_path='trained.safetensors')
print(generate(model, 'First Citi'))

# Output:
# Epoch 1/6, Average Loss: 4.0448 (53.28 sec)
# Validation Loss: 3.1952 (5.28 sec)
# Epoch 2/6, Average Loss: 2.6916 (51.76 sec)
# Validation Loss: 3.0308 (5.26 sec)
# Epoch 3/6, Average Loss: 2.5070 (52.08 sec)
# Validation Loss: 2.9546 (5.28 sec)
# Epoch 4/6, Average Loss: 2.4094 (52.64 sec)
# Validation Loss: 2.9938 (5.29 sec)
# Epoch 5/6, Average Loss: 2.3065 (51.41 sec)
# Validation Loss: 3.0093 (5.24 sec)
# Epoch 6/6, Average Loss: 2.2415 (51.29 sec)
# Validation Loss: 2.9584 (5.26 sec)
# Generated text:
# To be or not to be, tin in in i
# zen in in i
