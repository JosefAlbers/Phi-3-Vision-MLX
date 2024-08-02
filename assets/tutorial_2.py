import glob
import json
from types import SimpleNamespace
from transformers import AutoProcessor
from huggingface_hub import snapshot_download

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from typing import List, Union
import math

class ClipAttention(nn.Module):
    def __init__(self, dims, num_heads, bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dims // num_heads) ** -0.5
        self.q_proj = nn.Linear(dims, dims, bias=bias)
        self.k_proj = nn.Linear(dims, dims, bias=bias)
        self.v_proj = nn.Linear(dims, dims, bias=bias)
        self.out_proj = nn.Linear(dims, dims, bias=bias)

    def __call__(self, x):
        queries, keys, values = (proj(x).reshape(x.shape[0], x.shape[1], self.num_heads, -1).transpose(0, 2, 1, 3) for proj in (self.q_proj, self.k_proj, self.v_proj))
        output = mx.fast.scaled_dot_product_attention(queries, keys, values, scale=self.scale)
        return self.out_proj(output.transpose(0, 2, 1, 3).reshape(x.shape[0], x.shape[1], -1))

class ClipMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.activation_fn = nn.gelu_fast_approx
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def __call__(self, x):
        return self.fc2(self.activation_fn(self.fc1(x)))

class ClipEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = ClipAttention(config.hidden_size, config.num_attention_heads, bias=True)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = ClipMLP(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def __call__(self, x):
        x = x + self.self_attn(self.layer_norm1(x))
        return x + self.mlp(self.layer_norm2(x))

class ClipEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = [ClipEncoderLayer(config) for _ in range(config.num_hidden_layers)]

class ClipEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.class_embedding = mx.zeros(config.hidden_size)
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)

    def __call__(self, x):
        batch_size = x.shape[0]
        patch_embeddings = self.patch_embedding(x)
        patch_embeddings = mx.flatten(patch_embeddings, start_axis=1, end_axis=2)
        embed_dim = patch_embeddings.shape[-1]
        cls_embeddings = mx.broadcast_to(self.class_embedding, (batch_size, 1, embed_dim))
        position_ids = mx.arange(self.num_positions)[None]
        embeddings = mx.concatenate((cls_embeddings, patch_embeddings), axis=1)
        embeddings += self.position_embedding(position_ids)
        return embeddings

class ClipModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = ClipEmbeddings(config)
        self.pre_layrnorm = nn.LayerNorm(config.hidden_size)
        self.encoder = ClipEncoder(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size)

    def __call__(self, x):
        x = self.embeddings(x)
        x = self.pre_layrnorm(x)
        for l in self.encoder.layers[:-1]:
            x = l(x)
        return x[:, 1:]

class ClipVModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vision_model = ClipModel(config)

class Phi3ImageEmbedding(nn.Module):
    CLIP_VIT_LARGE_PATCH14_336_CONFIG = SimpleNamespace(
        hidden_size=1024,
        image_size=336,
        intermediate_size=4096,
        layer_norm_eps=1e-05,
        num_attention_heads=16,
        num_channels=3,
        num_hidden_layers=24,
        patch_size=14,
        )
    def __init__(self, config):
        super().__init__()
        self.img_processor = ClipVModel(self.CLIP_VIT_LARGE_PATCH14_336_CONFIG)
        self.image_dim_out = image_dim_out = config.img_processor['image_dim_out']
        self.glb_GN = mx.zeros([1, 1, image_dim_out * 4])
        self.sub_GN = mx.zeros([1, 1, 1, image_dim_out * 4])
        self.img_projection = [nn.Linear(image_dim_out * 4, config.hidden_size), nn.GELU(), nn.Linear(config.hidden_size, config.hidden_size)]

    def __call__(self, txt_embeds, img_embeds, img_sizes):
        if img_embeds is None:
            return txt_embeds
        B = img_embeds.shape[0]
        positions = mx.array(np.argwhere(np.array(input_ids) < 0))
        img_sizes, positions = (img_sizes // 336).tolist(), positions.tolist()
        img_features = self.img_processor.vision_model(img_embeds.reshape(-1, *img_embeds.shape[2:]).transpose(0, 2, 3, 1))
        img_features = img_features.reshape(B, -1, *img_features.shape[1:])
        C, H = self.image_dim_out, int(img_features.shape[2] ** 0.5)
        output_imgs, output_len = [], []
        for _bs in range(B):
            h, w = img_sizes[_bs]
            B_ = h * w
            def _reshape_and_concatenate(img, shape, tile_shape):
                return mx.concatenate([img.reshape(shape).transpose(0, 1, 3, 2, 4, 5).reshape(tile_shape), mx.tile(self.sub_GN, (1, tile_shape[1], 1, 1))], axis=2).reshape(1, -1, 4 * C)
            glb_img = _reshape_and_concatenate( img_features[_bs, :1], (1, H//2, 2, H//2, 2, C), (1, H//2, H//2, 4*C) )
            sub_img = _reshape_and_concatenate( img_features[_bs, 1:B_+1], (B_, H//2, 2, H//2, 2, C), (1, h*12, w*12, 4*C) )
            x = mx.concatenate([sub_img, self.glb_GN, glb_img], axis=1)
            for l in self.img_projection:
                x = l(x)
            output_imgs.append(x)
            output_len.append(int((h*w + 1) * 144 + 1 + (h + 1) * 12))
        idx = 0
        for i, cnt in enumerate(output_len):
            txt_embeds[positions[idx][0], positions[idx][1] : positions[idx][1] + cnt] = output_imgs[i]
            idx += cnt
        return txt_embeds

class SuRoPE:
    def __init__(self, config):
        self.dim = config.hidden_size // config.num_attention_heads
        self.original_max_position_embeddings = config.original_max_position_embeddings
        self.rope_theta = config.rope_theta
        self.scaling_factor = math.sqrt(1 + math.log(config.max_position_embeddings / config.original_max_position_embeddings) / math.log(config.original_max_position_embeddings))
        self.long_factor = config.rope_scaling["long_factor"]
        self.short_factor = config.rope_scaling["short_factor"]

    def __call__(self, q, k, position_ids=None):
        position_ids = mx.arange(q.shape[2], dtype=mx.float32)[None] # if position_ids is None else position_ids
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

    def __call__(self, x):
        B, L, _ = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = mx.split(qkv, self.chop, axis=-1)
        q = q.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        q, k = self.rope(q,k)
        mask = mx.triu(mx.full((v.shape[2], v.shape[2]), -mx.inf), k=1)
        # o = mx.fast.scaled_dot_product_attention(q,k,v,scale=self.scale,mask=mask)
        w = (q * self.scale) @ k.transpose(0, 1, 3, 2)
        w += mask
        w = mx.softmax(w, axis=-1)
        o = w @ v
        o = o.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(o).astype(qkv.dtype)

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

    def __call__(self, x):
        r = self.self_attn(self.input_layernorm(x))
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r

class Phi3VModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.vision_embed_tokens = Phi3ImageEmbedding(config)
        self.layers = [Phi3DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, input_ids, pixel_values, image_sizes):
        x = self.embed_tokens(input_ids)
        x = self.vision_embed_tokens(x, pixel_values, image_sizes)
        for l in self.layers:
            x = l(x)
        return self.norm(x)

class Phi3VForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = Phi3VModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(self, input_ids, pixel_values=None, image_sizes=None):
        x = self.model(input_ids, pixel_values, image_sizes)
        return self.lm_head(x)

def load(model_id = 'microsoft/Phi-3-vision-128k-instruct'):
    model_path = snapshot_download(model_id)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    with open(f"{model_path}/config.json", "r") as f:
        config = json.load(f)
    model_config = SimpleNamespace(**config)
    model_weight = [(k, v.transpose(0, 2, 3, 1) if "patch_embedding.weight" in k else v) for wf in glob.glob(f"{model_path}/*.safetensors") for k, v in mx.load(wf).items()]
    model = Phi3VForCausalLM(model_config)
    model.load_weights(model_weight)
    mx.eval(model.parameters())
    model.eval()
    return model, processor

model, processor = load()

# Text Only
inputs = processor('Hello world!', return_tensors='np')
input_ids = mx.array(inputs['input_ids'])
logits = model(input_ids)
token = mx.argmax(logits[:, -1, :], axis=-1)
list_tokens = token.tolist()
for i in range(5):
    input_ids = mx.concatenate([input_ids, token[None]], axis=-1)
    logits = model(input_ids)
    token = mx.argmax(logits[:, -1, :], axis=-1)
    list_tokens += token.tolist()
print(processor.tokenizer.decode(list_tokens))
# Output: How are you doing today?

# Text + Image
from PIL import Image
import requests
prompt = f"<|user|>\n<|image_1|>\nWhat is shown in this image?<|end|>\n<|assistant|>\n"
images = [Image.open(requests.get("https://assets-c4akfrf5b4d3f4b7.z01.azurefd.net/assets/2024/04/BMDataViz_661fb89f3845e.png" , stream=True).raw)]
inputs = processor(prompt, images, return_tensors="np")
input_ids, pixel_values, image_sizes = [mx.array(inputs[i]) for i in ['input_ids', 'pixel_values', 'image_sizes']]
logits = model(input_ids, pixel_values, image_sizes)
token = mx.argmax(logits[:, -1, :], axis=-1)
list_tokens = token.tolist()
for i in range(50):
    input_ids = mx.concatenate([input_ids, token[None]], axis=-1)
    logits = model(input_ids)
    token = mx.argmax(logits[:, -1, :], axis=-1)
    list_tokens += token.tolist()
print(processor.tokenizer.decode(list_tokens))
# Output: The image displays a chart with a series of connected dots forming a line that trends upwards, indicating a positive correlation between two variables. The chart is labeled with 'X' on the horizontal axis and 'Y' on the vertical axis,
