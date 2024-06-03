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
import time

class LoRALinear(nn.Module): # copied from mlx-examples (https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/tuner/lora.py)
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
        nrmlz = lambda img: ((np.array(img) / 255.0 - self.image_mean) / self.image_std).transpose(2,0,1)
        hd_images = [nrmlz(im) for im in elems]
        global_image = [torch.nn.functional.interpolate(torch.from_numpy(im[None]), size=(336, 336), mode='bicubic').numpy() for im in hd_images]
        shapes = [[im.shape[1], im.shape[2]] for im in hd_images]
        num_img_tokens = [int((h//336*w//336+1)*144 + 1 + (h//336+1)*12) for h, w in shapes]
        hd_images_reshape = [im
            .reshape(1, 3, h//336, 336, w//336, 336)
            .transpose(0,2,4,1,3,5)
            .reshape(-1, 3, 336, 336)
            for im, (h, w) in zip(hd_images, shapes)]
        hd_images_reshape = [np.concatenate([_global_image, _im], axis=0) for _global_image, _im in zip(global_image, hd_images_reshape)]
        def pad_to_max_num_crops_tensor(images, max_crops=17):
            B, _, H, W = images.shape
            if B < max_crops:
                pad = np.zeros((max_crops - B, 3, H, W))
                images = np.concatenate([images, pad], axis=0)
            return images
        image_transformed = [pad_to_max_num_crops_tensor(im) for im in hd_images_reshape]
        image_transformed = np.stack(image_transformed, axis=0)
        return {"pixel_values": image_transformed, "image_sizes": shapes, "num_img_tokens": num_img_tokens}

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

    def __call__(self, txt_embeds, img_embeds, img_sizes, positions):
        B = img_embeds.shape[0]
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

class Phi3SuScaledRotaryEmbedding:
    def __init__(self, dim, config, **kwargs):
        self.inv_freq_short = 1.0 / (mx.array(config.rope_scaling["short_factor"], dtype=mx.float32) * config.rope_theta**(mx.arange(0, dim, 2, dtype=mx.float32) / dim))
        self.inv_freq_long = 1.0 / (mx.array(config.rope_scaling["long_factor"], dtype=mx.float32) * config.rope_theta**(mx.arange(0, dim, 2, dtype=mx.float32) / dim))
        self.original_max_position_embeddings = config.original_max_position_embeddings
        self.scaling_factor = math.sqrt(1 + math.log(config.max_position_embeddings / config.original_max_position_embeddings) / math.log(config.original_max_position_embeddings))

    def _get_cos_sin(self, offset, L, pids):
        def _get_pids(offset, L, pids_tensor):
            if offset < 1:
                return pids
            last_column = pids_tensor[:, -1][:,None]
            start_indices = last_column + offset - pids_tensor.shape[1] + 2
            range_tensor = torch.arange(L)[None].expand(start_indices.shape[0], -1)
            final_pids_tensor = start_indices + range_tensor
            return mx.array(final_pids_tensor)

        position_ids = _get_pids(offset, L, pids) if pids is not None else mx.arange(offset, offset+L, dtype=mx.float32)[None]
        inv_freq = self.inv_freq_long if position_ids.max()+1 > self.original_max_position_embeddings else self.inv_freq_short
        inv_freq_expanded = mx.repeat(inv_freq[None, :, None], position_ids.shape[0], axis=0)
        position_ids_expanded = position_ids[:, None, :]
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(0, 2, 1)  
        emb = mx.concatenate([freqs, freqs], axis=-1)  
        cos = mx.cos(emb) * self.scaling_factor
        sin = mx.sin(emb) * self.scaling_factor
        return mx.expand_dims(cos, axis=1), mx.expand_dims(sin, axis=1) 

    def __call__(self, q, k=None, offset=0, pids=None):
        def _rotate_half(x):
            midpoint = x.shape[-1] // 2  
            x1, x2 = x[..., :midpoint], x[..., midpoint:]  
            return mx.concatenate([-x2, x1], axis = -1) 
        cos, sin = self._get_cos_sin(offset, q.shape[2], pids)
        return (q * cos) + (_rotate_half(q) * sin) if k is None else (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)

def _get_mask_4d(past_key_values_length, L, mask):
    mask_4d = mx.triu(mx.full((L+past_key_values_length, L+past_key_values_length), -mx.inf), k=1)[None, None]
    if mask is not None:
        pad_len = L+past_key_values_length - mask.shape[-1]
        mask = mx.pad(mask, ((0,0),(0,pad_len)), 1)
        mask = mx.expand_dims(mask, (1,2))
        mask = mask*mask.transpose(0,1,3,2)
        mask = mx.where(mask==1, 0, -np.inf)
        mask_4d += mask
        mask_4d = mx.repeat(mask_4d, 32, axis=1)
    mask_4d = mask_4d[:,:,past_key_values_length:,:]
    return mask_4d

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
        self.rope = Phi3SuScaledRotaryEmbedding(self.head_dim, config)
        self.use_quantized_cache=getattr(config, "use_quantized_cache", False)
    def __call__(self, x, cache, pids=None, mask=None):
        B, L, D = x.shape
        qkv = self.qkv_proj(x)
        query_pos = self.n_heads * self.head_dim
        queries, keys, values = mx.split(qkv, [query_pos, query_pos + self.n_kv_heads * self.head_dim], axis=-1)
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        if cache is not None:
            if self.use_quantized_cache:
                key_cache = mx.dequantize(*cache[0], group_size=32).reshape((B, self.n_kv_heads, -1, self.head_dim))
                value_cache = mx.dequantize(*cache[1], group_size=32).reshape((B, self.n_kv_heads, -1, self.head_dim))
            else:
                key_cache, value_cache = cache
            past_key_values_length = key_cache.shape[2]
            queries, keys = self.rope(queries, keys, offset=past_key_values_length, pids=pids)
            keys = mx.concatenate([key_cache, keys], axis=2)
            values = mx.concatenate([value_cache, values], axis=2)
        else:
            past_key_values_length = 0
            queries, keys = self.rope(queries, keys, offset=0, pids=pids)
        scores = (queries * self.scale) @ keys.transpose(0, 1, 3, 2)
        scores += _get_mask_4d(past_key_values_length, L, mask)
        scores = mx.softmax(scores, axis=-1)
        output = scores @ values
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        if self.use_quantized_cache:
            return self.o_proj(output), (mx.quantize(keys.reshape((B*self.n_kv_heads,-1)), group_size=32), mx.quantize(values.reshape((B*self.n_kv_heads,-1)), group_size=32))
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

    def __call__(self, x, cache, pids, mask):
        r, cache = self.self_attn(self.input_layernorm(x), cache, pids, mask)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r, cache

class Phi3VModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.vision_embed_tokens = Phi3ImageEmbedding(config)
        self.layers = [Phi3DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    def __call__(self, input_ids, pixel_values, image_sizes, positions, cache, pids, mask):
        x = self.embed_tokens(input_ids)
        if pixel_values is not None:
            x = self.vision_embed_tokens(x, pixel_values, image_sizes, positions)
        cache = [None] * len(self.layers) if cache is None else cache
        for i, l in enumerate(self.layers):
            x, cache[i] = l(x, cache[i], pids, mask)
        return self.norm(x), cache

class Phi3VLModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = Phi3VModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    def __call__(self, input_ids, pixel_values=None, image_sizes=None, positions=None, cache=None, pids=None, mask=None):
        x, cache = self.model(input_ids, pixel_values, image_sizes, positions, cache, pids, mask)
        return self.lm_head(x), cache
    @property
    def layers(self):
        return self.model.layers

class Phi3VProcessor:
    def __init__(self, local_dir):
        self.tokenizer = AutoTokenizer.from_pretrained(local_dir)
        self.img_processor = Phi3VImageProcessor()
    def _tokenize(self, texts):
        if isinstance(texts, str):
            return {'input_ids': mx.array(self.tokenizer(texts).input_ids)[None]}
        input_ids = self.tokenizer(texts).input_ids
        max_length = max(len(sublist) for sublist in input_ids)
        position_ids = mx.array([[1]*(max_length-len(sublist)) + list(range(len(sublist))) for sublist in input_ids])
        attention_masks = mx.array([[0]*(max_length-len(sublist)) + [1]*len(sublist) for sublist in input_ids])
        input_ids = mx.array([[0]*(max_length-len(sublist)) + sublist for sublist in input_ids])
        return {'input_ids':input_ids, 'pids':position_ids, 'mask':attention_masks}
    def __call__(self, images, texts):
        if images is None:
            return self._tokenize(texts)
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

def linear_to_lora_layers(model, lora_layers, config):
    if isinstance(lora_layers, int):
        lora_layers = model.layers[-lora_layers:]
    elif isinstance(lora_layers, list):
        lora_layers = [model.layers[i] for i in lora_layers]
    else:
        raise ValueError("Invalid type for lora_layers. Expected int (number of layers) or list (layer indices or names).")
    def to_lora(layer):
        return LoRALinear.from_linear( layer, r=config["rank"], alpha=config["alpha"], scale=config["scale"], dropout=config["dropout"])
    for l in lora_layers:
        lora_layers = [(k, to_lora(m)) for k, m in l.named_modules() if k == "self_attn.qkv_proj"]
        l.update_modules(tree_unflatten(lora_layers))

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
    return [(k, v.transpose(0, 2, 3, 1) if "patch_embedding.weight" in k else v) for wf in glob.glob(f"{model_path}/*.safetensors") for k, v in mx.load(wf).items()]

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

def generate(model, processor, prompt, images=None, max_tokens=100, verbose=True, stream=False):
    tic = time.perf_counter()
    dict_input = processor(images, prompt)
    logits, cache = model(**dict_input)
    prompt_time = time.perf_counter() - tic
    tic = time.perf_counter()
    token = mx.argmax(logits[:, -1, :], axis=-1)[:,None]
    list_tokens = [token]
    mask=dict_input.get('mask', None)
    pids=dict_input.get('pids', None)
    for i in range(max_tokens-1):
        logits, cache = model(input_ids=token, cache=cache, mask=mask, pids=pids)
        token = mx.argmax(logits[:, -1, :], axis=-1)[:,None]
        list_tokens.append(token)
    list_tokens = mx.concatenate(list_tokens, axis=1)

    result = [processor.tokenizer.decode(i) for i in list_tokens.tolist()]
    for i, gen in enumerate(result):
        print(f'\n< Generated text for prompt #{i} >') if len(result) > 1 else None
        print(gen)
    if verbose:
        gen_time = time.perf_counter() - tic
        prompt_tps =  dict_input['input_ids'].size / prompt_time
        gen_tps = (list_tokens.size - 1) / gen_time
        print(f"\nPrompt: {prompt_tps:.3f} tokens-per-sec")
        print(f"Generation: {gen_tps:.3f} tokens-per-sec")
    return result
    
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
        json.dump(vars(model_cfg)|{'quantized':quantization_config, 'sanitized':True}, f, indent=4)
    mx.save_safetensors(f'{to_path}/quantized_model.safetensors', quantized_weights)

class TrainingCallback:
    def __init__(self, lora_cfg, lr_schedule, sum_every=3):
        self.lora_cfg = lora_cfg
        self.adapter_path = lora_cfg['adapter_path']
        self.lr_schedule = lr_schedule
        self.sum_every = sum_every
        self.current_step = 0
        self.sum_loss = .0
        self.best_loss = math.inf
        self.train_log = {'step_i': [], 'step_loss': [], 'avg_i': [], 'avg_loss': []}
        self.start_time = time.perf_counter()

    def __call__(self, model, lvalue):
        self.current_step += 1
        step_loss = lvalue.item()
        print(f'- Step loss at step {self.current_step}: {step_loss}')
        self.train_log['step_i'].append(self.current_step)
        self.train_log['step_loss'].append(step_loss)
        self.sum_loss += step_loss
        
        if self.current_step % self.sum_every == 0:
            avg_loss = self.sum_loss / self.sum_every
            self.sum_loss = 0.0
            self.train_log['avg_i'].append(self.current_step)
            self.train_log['avg_loss'].append(avg_loss)
            print(f'Avg loss at step {self.current_step}: {avg_loss}')
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                mx.save_safetensors(f'{self.adapter_path}/adapters.safetensors', dict(tree_flatten(model.trainable_parameters())))

    def end_log(self):
        train_log = self.train_log
        train_log['train_time'] = time.perf_counter() - self.start_time
        with open(f'{self.adapter_path}/adapter_config.json', "w") as f:
            json.dump(self.lora_cfg, f, indent=4)
        with open(f'{self.adapter_path}/adapter_train_log.json', "w") as f:
            json.dump(train_log, f, indent=4)
        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1.plot(train_log['step_i'], train_log['step_loss'], color='b', alpha=0.5, label='Step Loss')
        ax1.plot(train_log['avg_i'], train_log['avg_loss'], color='r', label='Avg Loss')
        ax1.set_title('Training Loss Curves')
        ax1.legend()
        ax2.plot(self.lr_schedule)
        ax2.ticklabel_format(axis='y', style='sci')
        ax2.set_title('Learning Rate Schedule')
        plt.tight_layout() 
        fig.savefig(f'train_log.png')
        print(f"Training log saved to {self.adapter_path}")
        print(f"Total training time: {train_log['train_time']:.2f} seconds")

def train_lora(lora_layers=5, lora_rank=16, epochs=10, lr=1e-4, warmup=.5, mask_ratios=[.0], adapter_path='adapters', dataset_path="JosefAlbers/akemiH_MedQA_Reason"): # or mask_ratios=[.0, .1, .3, .5]
    
    model, processor = load()

    def _prompt(example):
        _question = example['input'].rsplit(' A: ', 1)[0].strip()
        _summary = example['summary'].strip().split('\n', 1)[0].strip()
        _prompt = f"<|user|>\n{_question}<|end|>\n<|assistant|>\n{_summary}<|end|>"
        _tokens = processor.tokenizer.encode(_prompt)
        _idx = _tokens.index(32001)
        example['input_tokens'] = _tokens
        example['idx_assistant'] = _idx
        return example

    def _mask(input_tokens, idx_assistant, mask_ratios):
        mask_range = range(max((i for i, num in enumerate(input_tokens) if num < 0), default=0)+3, idx_assistant-3)
        list_masks = []
        for ratio in mask_ratios:
            if ratio > 0:
                list_masks.append( [ 0 if i in random.sample(mask_range, int(len(mask_range)*ratio)) else 1 for i in range(len(input_tokens)) ] )
            else:
                list_masks.append([1]*len(input_tokens))
        list_tokens = [input_tokens]*len(mask_ratios)
        loss_scale = [10.**(-10*i) for i in mask_ratios]
        return mx.array(list_tokens), mx.array(list_masks), mx.array(loss_scale)

    def _get_batch(i):
        idx_assistant = ds[i]['idx_assistant']
        list_tokens = ds[i]['input_tokens']
        input_tokens, input_mask, loss_scale = _mask(list_tokens, idx_assistant, mask_ratios)
        ground_truth = input_tokens[:, idx_assistant+1:]
        return input_tokens, ground_truth, idx_assistant, input_mask, loss_scale

    def _loss(model, batch):
        input_tokens, ground_truth, idx_assistant, input_mask, loss_scale = batch
        logit_output, _ = model(input_tokens, mask = input_mask)
        logit_output = logit_output[:, idx_assistant:-1, :].astype(mx.float32)
        loss_ce = nn.losses.cross_entropy(logit_output, ground_truth, reduction='none')
        loss_ce = loss_ce.mean(axis=-1)
        loss_ce = (loss_ce * loss_scale).sum()
        return loss_ce
        
    def _set_lora(lora_layers, lora_rank, adapter_path):
        lora_cfg = {
            "adapter_path": adapter_path,
            "lora_layers": lora_layers,
            "lora_parameters": {"rank": lora_rank, "alpha": lora_rank, "dropout": 0.0, "scale": 1.0},
        }
        os.makedirs(adapter_path, exist_ok=True)
        return lora_cfg

    ds = datasets.load_dataset(dataset_path, split='train').take(10) # `debug
    ds = ds.map(_prompt).select_columns(['input_tokens', 'idx_assistant'])
    ds = datasets.concatenate_datasets([ds]*epochs)
    steps = len(ds)
    lora_cfg = _set_lora(lora_layers, lora_rank, adapter_path)
    model.freeze()    
    linear_to_lora_layers(model, lora_cfg['lora_layers'], lora_cfg['lora_parameters'])
    model.train()
    distil_loss_value_and_grad = nn.value_and_grad(model, _loss)
    def _get_lr_schedule(lr, steps, warmup):
        n_warmup = int(steps*warmup)
        return mx.concatenate([mx.linspace(1e-6, lr, n_warmup), mx.linspace(lr, 1e-6, steps - n_warmup + 1)[1:]])
    lr_schedule = _get_lr_schedule(lr, steps, warmup)
    callback = TrainingCallback(lora_cfg, lr_schedule)
    optimizer=optim.AdamW(learning_rate=lr_schedule[0])
    state = [model.state, optimizer.state]
    for i in range(steps):
        batch_i = _get_batch(i)
        lvalue, grad = distil_loss_value_and_grad(model, batch_i)
        optimizer.learning_rate = lr_schedule[i]
        optimizer.update(model, grad)
        mx.eval(state, lvalue)
        callback(model, lvalue)
    callback.end_log()
    del model
    del processor

def recall(dataset_path="JosefAlbers/akemiH_MedQA_Reason"):
    model, processor = load(adapter_path='adapters')
    def _recall(example):
        def _get_alphabet(text):
            return "".join([char for char in text if char.isalpha()]).upper()[0]
        question = example['input']
        _question = question.rsplit(' A: ', 1)[0].strip()
        prompt_recall = f"<|user|>\n{_question}<|end|>\n<|assistant|>"
        example['recall'] = generate(model, processor, prompt_recall, max_tokens=30, verbose=False)
        prompt_answer = f"<|user|>\n{question}<|end|>\n<|assistant|>\nThe correct answer is"
        example['attempt'] = _get_alphabet(generate(model, processor, prompt_answer, max_tokens=3, verbose=False))
        _summary = example['summary'].strip().split('\n', 1)[0].strip()
        example['result'] = f"Question: {_question}\n- Taught: {_summary}\n- Recall: {example['recall']}\n- Answer: {example['output']}\n- Attenmpt: {example['attempt']}\n- Correct: {example['output'] == example['attempt']}"
        return example
    ds = datasets.load_dataset(dataset_path, split='train').take(10)
    ds = ds.map(_recall)
    num_recall = len(ds.filter(lambda x: x["output"] == x["attempt"]))
    print('\n'.join(ds['result']), f'\n---\nFinal Score: {num_recall/len(ds)}({num_recall}/{len(ds)})')
    del model
    del processor

"""
Examples:

quantize()
train_lora()
recall()

model, processor = load() # `vanilla
model, processor = load(model_path='quantized_phi3v') # `q_model
model, processor = load(use_quantized_cache=True) # `q_cache
model, processor = load(adapter_path='adapters') # `lora
model, processor = load(model_path='quantized_phi3v', use_quantized_cache=True) # `q_model_cache

generate(model, processor, "<|user|>\n<|image_1|>\nWhat is shown in this image?<|end|>\n<|assistant|>\n", [Image.open(requests.get("https://assets-c4akfrf5b4d3f4b7.z01.azurefd.net/assets/2024/04/BMDataViz_661fb89f3845e.png" , stream=True).raw)])
generate(model, processor, "<|user|>Write a sci-fi thriller.<|end|>\n<|assistant|>\n")
generate(model, processor, ["<|user|>Write an executive summary for a communications business plan<|end|>\n<|assistant|>\n", "<|user|>Write a resume.<|end|>\n<|assistant|>\n", "<|user|>Write a mystery horror.<|end|>\n<|assistant|>\n","<|user|>Write a Neurology ICU Admission Note.<|end|>\n<|assistant|>\n"])
"""
