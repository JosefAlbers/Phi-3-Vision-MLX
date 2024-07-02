# This file contains codes from vegaluisjose's model.py (https://github.com/vegaluisjose/mlx-rag),
# with additional VDB class implementation.
# Copyright (c) vegaluisjose
# Licensed under The Apache License 2.0 (https://github.com/vegaluisjose/mlx-rag/blob/main/LICENSE)

import json
import mlx.core as mx
import mlx.nn as nn

from pydantic import BaseModel
from huggingface_hub import snapshot_download
from typing import List, Optional
from transformers import BertTokenizer

import datasets
import numpy as np
import os

PATH_GTE = 'models/gte'

def average_pool(last_hidden_state: mx.array, attention_mask: mx.array) -> mx.array:
    last_hidden = mx.multiply(last_hidden_state, attention_mask[..., None])
    return last_hidden.sum(axis=1) / attention_mask.sum(axis=1)[..., None]


class ModelConfig(BaseModel):
    dim: int = 1024
    num_attention_heads: int = 16
    num_hidden_layers: int = 24
    vocab_size: int = 30522
    attention_probs_dropout_prob: float = 0.1
    hidden_dropout_prob: float = 0.1
    layer_norm_eps: float = 1e-12
    max_position_embeddings: int = 512


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        dims: int,
        num_heads: int,
        mlp_dims: Optional[int] = None,
        layer_norm_eps: float = 1e-12,
    ):
        super().__init__()
        mlp_dims = mlp_dims or dims * 4
        self.attention = nn.MultiHeadAttention(dims, num_heads, bias=True)
        self.ln1 = nn.LayerNorm(dims, eps=layer_norm_eps)
        self.ln2 = nn.LayerNorm(dims, eps=layer_norm_eps)
        self.linear1 = nn.Linear(dims, mlp_dims)
        self.linear2 = nn.Linear(mlp_dims, dims)
        self.gelu = nn.GELU()

    def __call__(self, x, mask):
        attention_out = self.attention(x, x, x, mask)
        add_and_norm = self.ln1(x + attention_out)

        ff = self.linear1(add_and_norm)
        ff_gelu = self.gelu(ff)
        ff_out = self.linear2(ff_gelu)
        x = self.ln2(ff_out + add_and_norm)

        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self, num_layers: int, dims: int, num_heads: int, mlp_dims: Optional[int] = None
    ):
        super().__init__()
        self.layers = [
            TransformerEncoderLayer(dims, num_heads, mlp_dims)
            for i in range(num_layers)
        ]

    def __call__(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)

        return x


class BertEmbeddings(nn.Module):
    def __init__(self, config: ModelConfig):
        self.word_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.token_type_embeddings = nn.Embedding(2, config.dim)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.dim
        )
        self.norm = nn.LayerNorm(config.dim, eps=config.layer_norm_eps)

    def __call__(self, input_ids: mx.array, token_type_ids: mx.array) -> mx.array:
        words = self.word_embeddings(input_ids)
        position = self.position_embeddings(
            mx.broadcast_to(mx.arange(input_ids.shape[1]), input_ids.shape)
        )
        token_types = self.token_type_embeddings(token_type_ids)

        embeddings = position + words + token_types
        return self.norm(embeddings)


class Bert(nn.Module):
    def __init__(self, config: ModelConfig):
        self.embeddings = BertEmbeddings(config)
        self.encoder = TransformerEncoder(
            num_layers=config.num_hidden_layers,
            dims=config.dim,
            num_heads=config.num_attention_heads,
        )
        self.pooler = nn.Linear(config.dim, config.dim)

    def __call__(
        self,
        input_ids: mx.array,
        token_type_ids: mx.array,
        attention_mask: mx.array = None,
    ) -> tuple[mx.array, mx.array]:
        x = self.embeddings(input_ids, token_type_ids)

        if attention_mask is not None:
            # convert 0's to -infs, 1's to 0's, and make it broadcastable
            attention_mask = mx.log(attention_mask)
            attention_mask = mx.expand_dims(attention_mask, (1, 2))

        y = self.encoder(x, attention_mask)
        return y, mx.tanh(self.pooler(y[:, 0]))

class GteModel:
    def __init__(self) -> None:
        model_path = PATH_GTE
        if not os.path.exists(model_path):
            snapshot_download(repo_id="vegaluisjose/mlx-rag", local_dir=model_path)
            snapshot_download(repo_id="thenlper/gte-large", allow_patterns=["vocab.txt", "*.json"], local_dir=model_path)
        with open(f"{model_path}/config.json") as f:
            model_config = ModelConfig(**json.load(f))
        self.model = Bert(model_config)
        self.model.load_weights(f"{model_path}/model.npz")
        self.tokenizer = BertTokenizer.from_pretrained(model_path)

    def __call__(self, input_text: List[str]) -> mx.array:
        tokens = self.tokenizer(input_text, return_tensors="np", padding=True)
        tokens = {key: mx.array(v) for key, v in tokens.items()}
        last_hidden_state, _ = self.model(**tokens)
        embeddings = average_pool(
            last_hidden_state, tokens["attention_mask"].astype(mx.float32)
        )
        embeddings = embeddings / mx.linalg.norm(embeddings, ord=2, axis=1)[..., None]
        return embeddings

_list_api = [
"""Text to image
```python
from gradio_client import Client
client = Client("stabilityai/stable-diffusion-3-medium")
result = client.predict(
		prompt="{prompt}",
		negative_prompt="ugly, low quality",
		seed=0,
		randomize_seed=True,
		width=1024,
		height=1024,
		guidance_scale=5,
		num_inference_steps=28,
		api_name="/infer"
)
print('<|api_output|>'+result[0])
```
""",
"""Text to speech
```python
from gradio_client import Client
client = Client("parler-tts/parler_tts_mini")
result = client.predict(
        text="{prompt}",
        description="",
        api_name="/gen_tts"
)
print('<|api_output|>'+result)
```
""",
"""Transcribe youtube video
```python
from gradio_client import Client
client = Client("rajesh1729/youtube-video-transcription-with-whisper")
result = client.predict(
        url="{prompt}",
        api_name="/get_summary"
)
print('<|api_output|>'+result)
```
""",
]

class VDB:
    def __init__(self, list_api=None, n_line=1):
        self.embed = GteModel()
        if list_api is None:
            self.list_api = _list_api
            list_src = _list_api if n_line < 0 else ['\n'.join(s.split('\n')[:n_line]) for s in _list_api]
            self.list_embed = mx.concatenate([self.embed(i) for i in list_src])
        else:
            self.list_api = list_api['phi']
            self.list_embed = mx.array(np.squeeze(list_api.with_format(type='numpy', columns=['gte'])['gte']))
    def __call__(self, text, n_topk=1):
        query_embed = self.embed(text)
        scores = mx.matmul(query_embed, self.list_embed.T)
        list_idx = mx.argsort(scores)[:,:-1-n_topk:-1].tolist()
        return [[self.list_api[j] for j in i] for i in list_idx]
