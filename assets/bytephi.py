"""
BytePhi: Experimental Byte-Level Phi-3 Variant with RNN Integration

Key Features:
1. Universal Byte-Level Processing: Operates on raw byte sequences, enabling application to any file type.
2. Tokenizer-Free Approach: Bypasses traditional tokenization for more flexible processing.
3. RNN Integration: Uses LSTM layers instead of self-attention, exploring synergies between recurrent architectures and the Phi model.
4. Simplified Architecture: Streamlines Phi-3 components to focus on byte-level and RNN experiments.

Advantages of RNN over Attention in this context:
- Constant memory usage regardless of sequence length
- Efficient processing of long byte sequences
- Natural handling of sequential data
- Potential for streaming applications
- Simpler implementation for byte-level inputs

Goals:
- Explore feasibility of a universal model for processing any file type
- Investigate advantages of bypassing traditional tokenization
- Examine synergies between RNN characteristics and Phi model architecture

Note: This is an experimental prototype not suitable for production use.

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
from mlx.utils import tree_flatten
import time

RNN_SIZE = 2
VCB_SIZE = 39

class BasicRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell_rnn = nn.Linear(input_size + hidden_size, hidden_size)
        self.output_rnn = nn.Linear(hidden_size, output_size)

    def __call__(self, x, hidden=None):
        if hidden is None:
            hidden = mx.zeros((x.shape[0], self.hidden_size))

        outputs = []
        for i in range(x.shape[1]):
            combined = mx.concatenate([x[:, i, :], hidden], axis=1)
            hidden = mx.tanh(self.cell_rnn(combined))
            outputs.append(hidden)

        outputs = mx.stack(outputs, axis=1)
        return self.output_rnn(outputs), hidden

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
        # self.rnn = nn.LSTM(input_size=config.hidden_size, hidden_size=RNN_SIZE, bias=False)
        # self.proj_rnn = nn.Linear(RNN_SIZE, config.hidden_size, bias=False)
        self.rnn = BasicRNN(config.hidden_size, RNN_SIZE, config.hidden_size)
        self.mlp = Phi3MLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, x, position_ids, attention_mask, cache):
        r, cache = self.rnn(self.input_layernorm(x), cache) #, position_ids, attention_mask, cache)
        # r = self.proj_rnn(r)
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
        # self.lm_rnn = nn.Linear(config.hidden_size, VCB_SIZE, bias=False)

    def __call__(self, input_ids, pixel_values=None, image_sizes=None, position_ids=None, attention_mask=None, cache=None):
        x, cache = self.model(input_ids, pixel_values, image_sizes, position_ids, attention_mask, cache)
        # return self.lm_head(x), cache
        # return self.lm_rnn(x), cache
        return self.model.embed_rnn.as_linear(x), cache

def load_model(model_id='microsoft/Phi-3.5-mini-instruct', adapter_path=None):
    model_path = snapshot_download(model_id, allow_patterns=["*.safetensors", "config.json"])
    with open(f"{model_path}/config.json", "r") as f:
        config = json.load(f)
    model_config = SimpleNamespace(**config)
    model = Phi3ForCausalLM(model_config)
    model_weight = [(k, v) for wf in glob.glob(f"{model_path}/*.safetensors") for k, v in mx.load(wf).items()]
    model.load_weights(model_weight, strict=False)
    if adapter_path:
        model.load_weights(adapter_path, strict=False)
    mx.eval(model.parameters())
    model.eval()
    return model

def generate(model, prompt='Hello world!', max_tokens=10):
    model.eval()
    input_ids = mx.array(tokenizer.encode(prompt))[None]
    logits, cache = model(input_ids)
    token = mx.argmax(logits[:, -1, :], axis=-1)
    list_tokens = token.tolist()
    for i in range(max_tokens):
        logits, cache = model(token[:,None], cache=cache)
        token = mx.argmax(logits[:, -1, :], axis=-1)
        list_tokens += token.tolist()
    output = tokenizer.decode(list_tokens)
    return output

def train(learning_rate, num_epochs, batch_size=1, seq_length=64):
    def load_data(file_path, train_sep=512, eval_sep=-127, encoding='utf-8'):
        with open(file_path, 'r', encoding=encoding) as f:
            lines = f.readlines()
        train_data = ''.join(lines[:train_sep])
        eval_data = ''.join(lines[eval_sep:])
        return train_data, eval_data

    def create_batches(data, batch_size, seq_length, encoding='utf-8'):
        starts = [i for i in range(len(data)) if i == 0 or data[i-2:i] == '\n\n']
        sequences = [data[i:i+seq_length + 1] for i in starts if i + seq_length + 1 <= len(data)]
        np.random.shuffle(sequences)
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i+batch_size]
            batch = [tokenizer.encode(seq)[:seq_length+1] for seq in batch]
            x = [seq[:-1] for seq in batch]
            y = [seq[1:] for seq in batch]
            yield mx.array(x), mx.array(y)

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
    model = load_model()
    model.set_dtype(mx.float32)
    model.freeze()
    model.apply_to_modules(lambda k, v: v.unfreeze() if (k.endswith("rnn") or k.endswith("norm")) else None)
    mx.eval(model.parameters())
    model.train()
    # print("Trainable parameters:", [i[0] for i in tree_flatten(model.trainable_parameters())])
    num_batches_per_epoch = len(list(create_batches(train_data, batch_size, seq_length)))
    num_steps = num_epochs * num_batches_per_epoch
    num_warmup = num_steps // 6
    if num_warmup > 1:
        warmup = optim.linear_schedule(1e-5, learning_rate, steps=num_warmup)
        cosine = optim.cosine_decay(learning_rate, num_steps, 1e-5)
        lr_schedule = optim.join_schedules([warmup, cosine], [num_warmup])
    else:
        lr_schedule = optim.cosine_decay(learning_rate, num_epochs, 1e-5)
    optimizer = optim.Adam(learning_rate=lr_schedule)
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
    mx.save_safetensors(f'trained_rnn.safetensors', dict(tree_flatten(model.trainable_parameters())))
    # print(generate(model, 'First Citi'))
    del model

# Train byte-level tokenizer on input file
tokenizer = Tokenizer('input.txt')

# Train BasicRNN model
train(learning_rate=5e-4, num_epochs=30)

# Load trained model for inference
model = load_model(adapter_path='trained_rnn.safetensors')

# Test model with a prompt
prompt = 'First Citi'
output = generate(model, prompt)
print(f'{prompt=} + {output=}')
print('->', prompt+output)

# Output
# Epoch 1/30, Average Loss: 3.1337 (35.23 sec)
# Validation Loss: 3.0752 (3.86 sec)
# Epoch 2/30, Average Loss: 2.6952 (35.28 sec)
# Validation Loss: 2.9284 (3.87 sec)
# Epoch 3/30, Average Loss: 2.5035 (35.15 sec)
# Validation Loss: 2.8503 (3.87 sec)
# Epoch 4/30, Average Loss: 2.4400 (35.17 sec)
# Validation Loss: 2.9207 (3.87 sec)
# Epoch 5/30, Average Loss: 2.4050 (35.15 sec)
# Validation Loss: 2.8648 (3.87 sec)
# Epoch 6/30, Average Loss: 2.3595 (35.13 sec)
# Validation Loss: 2.8455 (3.87 sec)
# Epoch 7/30, Average Loss: 2.3162 (35.33 sec)
# Validation Loss: 2.8317 (3.87 sec)
# Epoch 8/30, Average Loss: 2.2820 (35.35 sec)
# Validation Loss: 2.8188 (3.87 sec)
# Epoch 9/30, Average Loss: 2.2303 (35.19 sec)
# Validation Loss: 2.8053 (3.87 sec)
# Epoch 10/30, Average Loss: 2.1794 (35.16 sec)
# Validation Loss: 2.7430 (3.87 sec)
# Epoch 11/30, Average Loss: 2.1205 (35.14 sec)
# Validation Loss: 2.8014 (3.87 sec)
# Epoch 12/30, Average Loss: 2.0724 (35.21 sec)
# Validation Loss: 2.7510 (3.87 sec)
# Epoch 13/30, Average Loss: 2.0021 (35.19 sec)
# Validation Loss: 2.8714 (3.87 sec)
# Epoch 14/30, Average Loss: 1.9445 (35.15 sec)
# Validation Loss: 2.7669 (4.06 sec)
# Epoch 15/30, Average Loss: 1.8880 (35.23 sec)
# Validation Loss: 2.8897 (3.88 sec)
# Epoch 16/30, Average Loss: 1.8303 (35.35 sec)
# Validation Loss: 2.8665 (3.87 sec)
# Epoch 17/30, Average Loss: 1.7786 (35.24 sec)
# Validation Loss: 2.9124 (3.87 sec)
# Epoch 18/30, Average Loss: 1.7248 (35.17 sec)
# Validation Loss: 2.9546 (3.88 sec)
# Epoch 19/30, Average Loss: 1.6639 (35.15 sec)
# Validation Loss: 2.8796 (3.88 sec)
# Epoch 20/30, Average Loss: 1.6003 (35.14 sec)
# Validation Loss: 2.9574 (3.87 sec)
# Epoch 21/30, Average Loss: 1.5466 (35.18 sec)
# Validation Loss: 2.9303 (3.88 sec)
# Epoch 22/30, Average Loss: 1.4900 (35.18 sec)
# Validation Loss: 3.0458 (3.87 sec)
# Epoch 23/30, Average Loss: 1.4178 (35.34 sec)
# Validation Loss: 3.0667 (3.87 sec)
# Epoch 24/30, Average Loss: 1.3671 (35.37 sec)
# Validation Loss: 3.0175 (3.87 sec)
# Epoch 25/30, Average Loss: 1.3240 (35.18 sec)
# Validation Loss: 3.1004 (3.87 sec)
# Epoch 26/30, Average Loss: 1.2761 (35.23 sec)
# Validation Loss: 3.2157 (3.87 sec)
# Epoch 27/30, Average Loss: 1.2167 (35.43 sec)
# Validation Loss: 3.3013 (3.89 sec)
# Epoch 28/30, Average Loss: 1.1635 (35.16 sec)
# Validation Loss: 3.3696 (3.88 sec)
# Epoch 29/30, Average Loss: 1.1254 (35.16 sec)
# Validation Loss: 3.4558 (3.87 sec)
# Epoch 30/30, Average Loss: 1.0822 (35.17 sec)
# Validation Loss: 3.4743 (4.03 sec)
# prompt='First Citi' + output='zen:\nwhat t'
# -> First Citizen:
# what t
