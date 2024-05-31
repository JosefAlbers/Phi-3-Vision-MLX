import mlx.core as mx
import mlx.nn as nn
import numpy as np
import math
import json
import glob
import os
import re
import requests
import torch
import datasets
import random
import mlx.optimizers as optim
import matplotlib.pyplot as plt
from huggingface_hub import snapshot_download
from shutil import copy
from mlx.utils import tree_flatten, tree_unflatten
from PIL import Image, ImageOps
from types import SimpleNamespace
from transformers import AutoTokenizer

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
        B, L = x.shape[:2]
        queries, keys, values = (proj(x).reshape(B, L, self.num_heads, -1).transpose(0, 2, 1, 3) for proj in (self.q_proj, self.k_proj, self.v_proj))
        output = mx.fast.scaled_dot_product_attention(queries, keys, values, scale=self.scale)
        return self.out_proj(output.transpose(0, 2, 1, 3).reshape(B, L, -1))

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

class LoRALinear(nn.Module):
    @staticmethod
    def from_linear(
        linear: nn.Linear,
        r: int = 8,
        alpha: float = 16,
        dropout: float = 0.0,
        scale: float = 10.0,
    ):
        output_dims, input_dims = linear.weight.shape
        if isinstance(linear, nn.QuantizedLinear):
            input_dims *= 32 // linear.bits
        lora_lin = LoRALinear(
            input_dims=input_dims,
            output_dims=output_dims,
            r=r,
            alpha=alpha,
            dropout=dropout,
            scale=scale,
        )
        lora_lin.linear = linear
        return lora_lin

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        r: int = 8,
        alpha: float = 16,
        dropout: float = 0.0,
        scale: float = 10.0,
        bias: bool = False,
    ):
        super().__init__()
        self.linear = nn.Linear(input_dims, output_dims, bias=bias)
        self.dropout = nn.Dropout(p=dropout)
        self.scale = scale * (alpha / r)
        scale = 1 / math.sqrt(input_dims)
        self.lora_a = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(input_dims, r),
        )
        self.lora_b = mx.zeros(shape=(r, output_dims))

    def __call__(self, x):
        y = self.linear(x)
        z = (self.dropout(x) @ self.lora_a) @ self.lora_b
        return y + (self.scale * z).astype(x.dtype)

def linear_to_lora_layers(model, num_lora_layers, config):
    def to_lora(layer):
        return LoRALinear.from_linear( layer, r=config["rank"], alpha=config["alpha"], scale=config["scale"], dropout=config["dropout"])
    for l in model.layers[-num_lora_layers:]:
        lora_layers = [(k, to_lora(m)) for k, m in l.named_modules() if k == "self_attn.qkv_proj"]
        l.update_modules(tree_unflatten(lora_layers))

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
        self.wte = nn.Embedding(config.vocab_size, config.hidden_size)
        self.img_processor = ClipVModel(self.CLIP_VIT_LARGE_PATCH14_336_CONFIG)
        self.image_dim_out = image_dim_out = config.img_processor['image_dim_out']
        self.glb_GN = mx.zeros([1, 1, self.image_dim_out * 4])
        self.sub_GN = mx.zeros([1, 1, 1, self.image_dim_out * 4])
        self.img_projection = [nn.Linear(image_dim_out * 4, config.hidden_size), nn.GELU(), nn.Linear(config.hidden_size, config.hidden_size)]
        self.vocab_size = config.vocab_size

    def __call__(self, input_ids, img_embeds, img_sizes, positions):
        bs = img_embeds.shape[0]
        img_features = self.img_processor.vision_model(img_embeds.reshape(-1, *img_embeds.shape[2:]).transpose(0,2,3,1))
        base_feat_height = base_feat_width = int(img_features.shape[1] ** 0.5)
        img_features = img_features.reshape(bs, -1, base_feat_height * base_feat_width, self.image_dim_out)
        C = self.image_dim_out
        H = base_feat_height
        output_imgs = []
        output_len = []
        for _bs in range(bs):
            h, w = (img_sizes[_bs] // 336).tolist()
            B_ = h * w
            glb_img = img_features[_bs, :1].reshape(1,H,H,C).reshape(1,H//2,2,H//2,2,C).transpose(0,1,3,2,4,5).reshape(1,H//2,H//2,4*C)
            glb_img = mx.concatenate([glb_img, mx.tile(self.sub_GN, (1, H//2, 1, 1))], axis=2).reshape(1,-1,4*C)
            sub_img = img_features[_bs, 1:B_+1].reshape(B_,H,H,C).reshape(B_,H//2,2,H//2,2,C).transpose(0,1,3,2,4,5).reshape(B_,-1,4*C).reshape(1, h, w, 12, 12, -1).transpose(0,1,3,2,4,5).reshape(1,h*12,w*12,4*C)
            sub_img = mx.concatenate([sub_img, mx.tile(self.sub_GN, (1, h*12, 1, 1))], axis=2).reshape(1,-1,4*C)
            output_imgs.append(mx.concatenate([sub_img, self.glb_GN, glb_img], axis=1))
            output_len.append(int((h*w+1)*144 + 1 + (h+1)*12))
        img_set_tensor = []
        for x in output_imgs:
            for l in self.img_projection:
                x = l(x)
            img_set_tensor.append(x)
        input_ids = mx.clip(input_ids, 0, self.vocab_size)
        hidden_states = self.wte(input_ids)
        idx = 0
        positions = positions.tolist()
        for i, cnt in enumerate(output_len):
            hidden_states[positions[idx][0], positions[idx][1] : positions[idx][1] + cnt] = img_set_tensor[i]
            idx += cnt
        return hidden_states

class Phi3SuScaledRotaryEmbedding:
    def __init__(self, dim, config):
        self.dim = dim
        self.base = config.rope_theta 
        self.short_factor = config.rope_scaling["short_factor"]
        self.long_factor = config.rope_scaling["long_factor"]
        self.original_max_position_embeddings = config.original_max_position_embeddings
        self.scaling_factor = math.sqrt(1 + math.log(config.max_position_embeddings / config.original_max_position_embeddings) / math.log(config.original_max_position_embeddings))
        self.inv_freq = None

    def __call__(self, q, k, p):
        cos, sin = self._get_cos_sin(p)
        q, k = self._apply_rotary_pos_emb(q, k, cos, sin)
        return q, k

    def _get_cos_sin(self, position_ids):
        seq_len = position_ids.max() + 1
        ext_factors = mx.array(self.long_factor, dtype=mx.float32) if seq_len > self.original_max_position_embeddings else mx.array(self.short_factor, dtype=mx.float32)
        inv_freq_shape = mx.arange(0, self.dim, 2, dtype=mx.float32) / self.dim
        self.inv_freq = 1.0 / (ext_factors * self.base**inv_freq_shape)
        inv_freq_expanded = mx.repeat(self.inv_freq[None, :, None], position_ids.shape[0], axis=0)
        position_ids_expanded = mx.array(position_ids, dtype=mx.float32)[:, None, :]
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(0, 2, 1)  
        emb = mx.concatenate([freqs, freqs], axis=-1)  
        cos = mx.cos(emb) * self.scaling_factor
        sin = mx.sin(emb) * self.scaling_factor
        return cos, sin 

    def _apply_rotary_pos_emb(selkf, q, k, cos, sin):
        def rotate_half(x):
            midpoint = x.shape[-1] // 2  
            x1, x2 = x[..., :midpoint], x[..., midpoint:]  
            return mx.concatenate([-x2, x1], axis = -1) 
        cos = mx.expand_dims(cos, axis=1)  
        sin = mx.expand_dims(sin, axis=1)
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed

class Phi3Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim = config.hidden_size
        self.n_heads = n_heads = config.num_attention_heads
        self.n_kv_heads = n_kv_heads = config.num_key_value_heads
        self.num_hidden_layers = config.num_hidden_layers
        self.head_dim = head_dim = config.hidden_size // n_heads
        self.scale = head_dim**-0.5
        op_size = n_heads * head_dim + 2 * (n_kv_heads * head_dim)
        self.qkv_proj = nn.Linear(dim, op_size, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)
        self.rotary_emb = Phi3SuScaledRotaryEmbedding(self.head_dim, config)
        self.use_quantized_cache=getattr(config, "use_quantized_cache", False)
    def __call__(self, x, cache):
        B, L, D = x.shape
        qkv = self.qkv_proj(x)
        query_pos = self.n_heads * self.head_dim
        queries, keys, values = mx.split(qkv, [query_pos, query_pos + self.n_kv_heads * self.head_dim], axis=-1)
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        if self.use_quantized_cache:
            past_key_values_length = cache[0][0].shape[0] if cache is not None else 0  
        else:
            past_key_values_length = cache[0].shape[2] if cache is not None else 0        
        position_ids = mx.arange(past_key_values_length, past_key_values_length+L, dtype=mx.float32)[None]
        queries, keys = self.rotary_emb(queries, keys, position_ids)
        if past_key_values_length > 0:
            if self.use_quantized_cache:
                key_cache = mx.dequantize(*cache[0], group_size=32).reshape((B, self.n_kv_heads, -1, self.head_dim))
                value_cache = mx.dequantize(*cache[1], group_size=32).reshape((B, self.n_kv_heads, -1, self.head_dim))
            else:
                key_cache, value_cache = cache
            keys = mx.concatenate([key_cache, keys], axis=2)
            values = mx.concatenate([value_cache, values], axis=2)
        scores = (queries * self.scale) @ keys.transpose(0, 1, 3, 2)
        scores += mx.triu(mx.full((L+past_key_values_length, L+past_key_values_length), -mx.inf), k=1)[None, None, past_key_values_length:, :]
        scores = mx.softmax(scores, axis=-1)
        output = scores @ values
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        if self.use_quantized_cache:
            return self.o_proj(output), (mx.quantize(keys.reshape((-1,32*96)), group_size=32), mx.quantize(values.reshape((-1,32*96)), group_size=32))
        else:
            return self.o_proj(output), (keys, values)

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

    def __call__(self, x, cache):
        r, cache = self.self_attn(self.input_layernorm(x), cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r, cache

class Phi3VModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vision_embed_tokens = Phi3ImageEmbedding(config)
        self.layers = [Phi3DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    def __call__(self, input_ids, pixel_values, image_sizes, positions, cache):
        if pixel_values is None:
            x = self.vision_embed_tokens.wte(input_ids).astype(mx.float32)
        else:
            x = self.vision_embed_tokens(input_ids, pixel_values, image_sizes, positions)
        cache = [None] * len(self.layers) if cache is None else cache
        for i, l in enumerate(self.layers):
            x, cache[i] = l(x, cache[i])
        return self.norm(x), cache

class Phi3VLModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = Phi3VModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    def __call__(self, input_ids, pixel_values=None, image_sizes=None, positions=None, cache=None):
        x, cache = self.model(input_ids, pixel_values, image_sizes, positions, cache)
        return self.lm_head(x), cache

    @property
    def layers(self): # `for linear_to_lora
        return self.model.layers

def interpolate_336(input):
    def get_weights_and_indices(scale, out_size, in_size):
        def cubic(x):
            abs_x = np.abs(x)
            abs_x2 = abs_x ** 2
            abs_x3 = abs_x ** 3
            f = ((1.5 * abs_x3 - 2.5 * abs_x2 + 1) * (abs_x <= 1) +
                (-0.5 * abs_x3 + 2.5 * abs_x2 - 4 * abs_x + 2) * ((abs_x > 1) & (abs_x <= 2)))
            return f
        kernel_radius = 2
        kernel_width = kernel_radius * 2
        out_coordinates = np.linspace(0, in_size - 1, out_size)
        in_coordinates = out_coordinates / scale
        left_indices = np.floor(in_coordinates - 0.5).astype(np.int32)
        right_indices = left_indices + 1
        left_indices = np.clip(left_indices, 0, in_size - 1)
        right_indices = np.clip(right_indices, 0, in_size - 1)
        weights = np.zeros((out_size, kernel_width), dtype=np.float32)
        indices = np.zeros((out_size, kernel_width), dtype=np.int32)
        for i in range(out_size):
            indices[i, 0] = left_indices[i]
            indices[i, 1] = right_indices[i]
            weights[i, 0] = cubic(in_coordinates[i] - left_indices[i])
            weights[i, 1] = cubic(right_indices[i] - in_coordinates[i])

            weight_sum = weights[i].sum()
            if weight_sum != 0:
                weights[i] /= weight_sum

        return weights, indices
    N, C, H, W = input.shape
    out_hw = 336
    output = np.zeros((N, C, out_hw, out_hw), dtype=input.dtype)
    h_weights, h_indices = get_weights_and_indices(out_hw / H, out_hw, H)
    w_weights, w_indices = get_weights_and_indices(out_hw / W, out_hw, W)
    for n in range(N):
        for c in range(C):
            for i in range(out_hw):
                for j in range(out_hw):
                    h_kernel = input[n, c, h_indices[i]]
                    w_kernel = h_kernel[:, w_indices[j]]
                    output[n, c, i, j] = np.sum(h_weights[i][:, None] * w_weights[j] * w_kernel)

    return output

class Phi3VImageProcessor:
    def __init__(self):
        self.num_crops=16
        self.image_mean=np.array([0.48145466, 0.4578275, 0.40821073])
        self.image_std=np.array([0.26862954, 0.26130258, 0.27577711])
    
    def __call__(self, images):
        images = [image.convert('RGB') for image in images]
        def HD_transform(img, hd_num=16):
            width, height = img.size
            trans = False
            if width < height:
                img = img.transpose(Image.TRANSPOSE)
                trans = True
                width, height = img.size

            ratio = (width/ height)
            scale = 1
            while scale*np.ceil(scale/ratio) <= hd_num:
                scale += 1
            scale -= 1
            new_w = int(scale * 336)
            new_h = int(new_w / ratio)

            img = img.resize([new_w, new_h], Image.BILINEAR)
            def padding_336(b):
                width, height = b.size
                diff_height = int(np.ceil(height / 336) * 336) - height
                top_padding = int(diff_height/2)
                bottom_padding = diff_height - top_padding
                b = ImageOps.expand(b, border=(0, top_padding, 0, bottom_padding), fill=(255, 255, 255))
                return b
            img = padding_336(img)
            if trans:
                img = img.transpose(Image.TRANSPOSE)
            return img
        elems = [HD_transform(im, hd_num = self.num_crops) for im in images] 
        img_processor = lambda img: ((np.array(img) / 255.0 - self.image_mean) / self.image_std).transpose(2,0,1)
        hd_images = [img_processor(im) for im in elems]
        # global_image = [interpolate_336(im[None]) for im in hd_images] # or 
        global_image = [torch.nn.functional.interpolate(torch.from_numpy(im[None]), size=(336, 336), mode='bicubic').numpy() for im in hd_images]
        shapes = [[im.shape[1], im.shape[2]] for im in hd_images]
        num_img_tokens = [int((h//336*w//336+1)*144 + 1 + (h//336+1)*12) for h, w in shapes]
        hd_images_reshape = [im
            .reshape(1, 3, h//336, 336, w//336, 336)
            .transpose(0,2,4,1,3,5)
            .reshape(-1, 3, 336, 336)
            for im, (h, w) in zip(hd_images, shapes)]
        hd_images_reshape = [np.concatenate([_global_image, _im], axis=0) for _global_image, _im in zip(global_image, hd_images_reshape)]
        def pad_to_max_num_crops_tensor(images, max_crops=5):
            B, _, H, W = images.shape
            if B < max_crops:
                pad = np.zeros((max_crops - B, 3, H, W))
                images = np.concatenate([images, pad], axis=0)
            return images
        image_transformed = [pad_to_max_num_crops_tensor(im, self.num_crops+1) for im in hd_images_reshape]
        image_transformed = np.stack(image_transformed, axis=0)
        return {"pixel_values": image_transformed, "image_sizes": shapes, "num_img_tokens": num_img_tokens}

class Phi3VProcessor:
    def __init__(self, local_dir):
        self.tokenizer = AutoTokenizer.from_pretrained(local_dir)
        self.img_processor = Phi3VImageProcessor()
    def __call__(self, images, texts):
        if images is None:
            return {'input_ids': mx.array(self.tokenizer.encode(texts))[None]}
        image_inputs = self.img_processor(images)
        return self._merge(image_inputs, texts)
    def _merge(self, images, texts):
        pattern = r"<\|image_\d+\|>"
        prompt_chunks = self.tokenizer(re.split(pattern, texts)).input_ids
        num_img_tokens = images['num_img_tokens']
        images, image_sizes = images['pixel_values'], images['image_sizes']
        image_tags = re.findall(pattern, texts) 
        image_ids = [int(s.split("|")[1].split("_")[-1]) for s in image_tags]
        image_ids_pad = [[-iid]*num_img_tokens[iid-1] for iid in image_ids]
        image_ids_pad = image_ids_pad + [[]] if len(prompt_chunks) > len(image_ids_pad) else image_ids_pad
        input_ids = []
        for chunk, pad in zip(prompt_chunks, image_ids_pad):
            input_ids.extend(chunk)
            input_ids.extend(pad)
        input_ids = np.array(input_ids)[None]
        positions = np.argwhere(input_ids < 0)
        return {"input_ids": mx.array(input_ids),
                "pixel_values": mx.array(images), 
                "image_sizes": mx.array(image_sizes),
                "positions": mx.array(positions)}

def _get_cfg(json_path, **kwargs):
    try:
        with open(json_path, "r") as f:
            cfg = SimpleNamespace(**(json.load(f)|kwargs))
        return cfg
    except:
        return False

def _get_wt(model_path, model_cfg):
    if getattr(model_cfg, 'sanitized', False):
        return [(k, v) for wf in glob.glob(f"{model_path}/*.safetensors") for k, v in mx.load(wf).items()]
    return [(k.replace('model.embed_tokens.weight', 'model.vision_embed_tokens.wte.weight'), v.transpose(0, 2, 3, 1) if "patch_embedding.weight" in k else v) for wf in glob.glob(f"{model_path}/*.safetensors") for k, v in mx.load(wf).items()]

def load(model_path='phi3v', adapter_path=None, **kwargs):
    if not os.path.exists(model_path):
        snapshot_download(repo_id="microsoft/Phi-3-vision-128k-instruct", allow_patterns=["*.safetensors", "*.json"], local_dir=model_path)
    model_cfg = _get_cfg(f"{model_path}/config.json", **kwargs)
    model = Phi3VLModel(model_cfg)
    nn.quantize(model, model_cfg.quantized['group_size'], model_cfg.quantized['bits']) if getattr(model_cfg, 'quantized', False) else None
    model.load_weights(_get_wt(model_path, model_cfg))
    if adapter_path:
        lora_cfg = _get_cfg(f"{adapter_path}/adapter_config.json")
        linear_to_lora_layers(model, lora_cfg.lora_layers, lora_cfg.lora_parameters)
        model.load_weights(f'{adapter_path}/adapters.safetensors', strict=False)
    mx.eval(model.parameters())
    model.eval()
    return model, Phi3VProcessor(model_path)

def generate(model, processor, prompt, images=None, max_tokens=10):
    logits, cache = model(**processor(images, prompt))
    token = mx.argmax(logits[:, -1, :], axis=-1)
    list_tokens = token.tolist()
    for _ in range(max_tokens-1):
        logits, cache = model(input_ids=token[None], cache=cache)
        token = mx.argmax(logits[:, -1, :], axis=-1)
        list_tokens += token.tolist()
        if list_tokens[-1] == processor.tokenizer.eos_token_id:
            break
    return processor.tokenizer.decode(list_tokens)
    
def quantize(from_path='phi3v', to_path='quantized_phi3v', q_group_size=64, q_bits=4):
    if not os.path.exists(from_path):
        snapshot_download(repo_id="microsoft/Phi-3-vision-128k-instruct", allow_patterns=["*.safetensors", "*.json"], local_dir=from_path)
    model_cfg = _get_cfg(f"{from_path}/config.json")
    model = Phi3VLModel(model_cfg)
    model.load_weights(_get_wt(from_path, model_cfg))
    nn.quantize(model, q_group_size, q_bits)
    quantization_config = {"group_size": q_group_size, "bits": q_bits}
    quantized_weights = dict(tree_flatten(model.parameters()))
    del model
    os.makedirs(to_path, exist_ok=True)
    for f in glob.glob(f"{from_path}/*.json"):
        copy(f, to_path)
    with open(f"{to_path}/config.json", "w") as f:
        json.dump(vars(model_cfg)|{'quantized':quantization_config, 'sanitized':True}, f)
    mx.save_safetensors(f'{to_path}/quantized_model.safetensors', quantized_weights)


def train_lora(lora_layers=1, lora_rank=16, epochs=3, lr=1e-4, warmup=.5):
    def _get_lora(lora_layers, lora_rank, adapter_path='adapters'):
        lora_cfg = {
            "adapter_path": adapter_path,
            "lora_layers": lora_layers,
            "lora_parameters": {"rank": lora_rank, "alpha": lora_rank, "dropout": 0.0, "scale": 1.0},
        }
        os.makedirs(adapter_path, exist_ok=True)
        with open(f'{adapter_path}/adapter_config.json', "w") as f:
            json.dump(lora_cfg, f, indent=4)
        return lora_cfg, adapter_path
    
    lora_cfg, adapter_path = _get_lora(lora_layers, lora_rank)
    model, processor = load()
    model.freeze()    
    linear_to_lora_layers(model, lora_cfg['lora_layers'], lora_cfg['lora_parameters'])
    model.train()

    def _prompt(example):
        _question = example['input'].rsplit(' A: ', 1)[0].strip()
        _summary = example['summary'].strip().split('\n', 1)[0].strip()
        _prompt = f"<|user|>\n{_question}<|end|>\n<|assistant|>\n{_summary}<|end|>"
        _tokens = processor.tokenizer.encode(_prompt)
        _idx = _tokens.index(32001)
        example['input_tokens'] = _tokens
        example['idx_assistant'] = _idx
        return example

    ds = datasets.load_dataset("JosefAlbers/akemiH_MedQA_Reason", split='train').take(10) # `debug
    ds = ds.map(_prompt).select_columns(['input_tokens', 'idx_assistant'])
    ds = datasets.concatenate_datasets([ds]*epochs)
    steps = len(ds)

    def _mask(list_tokens, idx_assistant):
        list_masked = []
        for denom in range(3, 11):
            num_to_remove = (idx_assistant - 6) // denom
            indices_to_remove = random.sample(range(3, idx_assistant-3), num_to_remove) 
            list_masked.append( [ 0 if i in indices_to_remove else list_tokens[i] for i in range(len(list_tokens)) ] )
        return list_masked

    def _get_batch(i):
        idx_assistant = ds[i]['idx_assistant']
        list_tokens = ds[i]['input_tokens']
        list_tokens = _mask(list_tokens, idx_assistant)
        input_tokens = mx.array(list_tokens)
        ground_truth = input_tokens[:, idx_assistant+1:]
        return input_tokens, ground_truth, idx_assistant

    def _loss(model, batch):
        input_tokens, ground_truth, idx_assistant = batch
        logit_output, _ = model(input_tokens)
        logit_output = logit_output[:, idx_assistant:-1, :].astype(mx.float32)
        loss_ce = nn.losses.cross_entropy(logit_output, ground_truth, reduction='mean')
        return loss_ce
        
    distil_loss_value_and_grad = nn.value_and_grad(model, _loss)
    def _get_lr_schedule(lr, steps, warmup):
        n_warmup = int(steps*warmup)
        return mx.concatenate([mx.linspace(1e-6, lr, n_warmup), mx.linspace(lr, 1e-6, steps - n_warmup + 1)[1:]])
    lr_schedule = _get_lr_schedule(lr, steps, warmup)
    optimizer=optim.AdamW(learning_rate=lr_schedule[0])
    state = [model.state, optimizer.state]
    best_loss = math.inf
    sum_loss = 0.0
    sum_every = 3
    train_log = {'step_i':[], 'step_loss':[], 'avg_i':[], 'avg_loss':[]}

    for i in range(steps):
        batch_i = _get_batch(i)
        lvalue, grad = distil_loss_value_and_grad(model, batch_i)
        optimizer.learning_rate = lr_schedule[i]
        optimizer.update(model, grad)
        mx.eval(state, lvalue)
        step_loss = lvalue.item()
        print('- Step loss', i, ':', step_loss)
        train_log['step_i'].append(i)
        train_log['step_loss'].append(step_loss)
        sum_loss += step_loss
        if (i+1) % sum_every == 0:
            avg_loss = sum_loss/sum_every
            sum_loss = 0.0
            train_log['avg_i'].append(i)
            train_log['avg_loss'].append(avg_loss)
            print('Avg loss', i, ':', avg_loss)
            if avg_loss < best_loss:
                best_loss = avg_loss
                mx.save_safetensors(f'{adapter_path}/adapters.safetensors', dict(tree_flatten(model.trainable_parameters())))
    
    plt.figure()
    plt.subplot(211)
    plt.plot(train_log['step_i'], train_log['step_loss'], color='b', alpha=0.5)
    plt.plot(train_log['avg_i'], train_log['avg_loss'], color='r')
    plt.subplot(212)
    plt.plot(lr_schedule)
    plt.savefig(f'{adapter_path}/train_log.png')
        
    del model
    del processor

def recall():
    model, processor = load(adapter_path='adapters')
    def _get_alphabet(text):
        return "".join([char for char in text if char.isalpha()]).upper()[0]

    def _recall(example):
        question = example['input']
        _question = question.rsplit(' A: ', 1)[0].strip()
        prompt_recall = f"<|user|>\n{_question}<|end|>\n<|assistant|>"
        example['recall'] = generate(model, processor, prompt_recall, max_tokens=30)
        prompt_answer = f"<|user|>\n{question}<|end|>\n<|assistant|>\nThe correct answer is"
        example['attempt'] = _get_alphabet(generate(model, processor, prompt_answer, max_tokens=3))
        _summary = example['summary'].strip().split('\n', 1)[0].strip()
        example['result'] = f"Question: {_question}\n- Taught: {_summary}\n- Recall: {example['recall']}\n- Answer: {example['output']}\n- Attenmpt: {example['attempt']}\n- Correct: {example['output'] == example['attempt']}"
        return example
    
    ds = datasets.load_dataset("JosefAlbers/akemiH_MedQA_Reason", split='train').take(10)
    ds = ds.map(_recall)
    print('\n'.join(ds['result']))
        
    del model
    del processor

"""
Example:

model, processor = load() # `vanilla
model, processor = load(model_path='quantized_phi3v') # `q_model
model, processor = load(use_quantized_cache=True) # `q_cache
model, processor = load(adapter_path='adapters') # `lora

print(generate(model, processor, "<|user|>\n<|image_1|>\nWhat is shown in this image?<|end|>\n<|assistant|>\n", [Image.open(requests.get("https://assets-c4akfrf5b4d3f4b7.z01.azurefd.net/assets/2024/04/BMDataViz_661fb89f3845e.png" , stream=True).raw)]))
print(generate(model, processor,  "<|user|>Write an exciting sci-fi.<|end|>\n<|assistant|>\n"))
"""