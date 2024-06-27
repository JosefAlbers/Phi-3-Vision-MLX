import math
import json
import glob
import os
import re
import requests
# import torch
import datasets
import random
import textwrap
import subprocess
import mlx.core as mx
import mlx.nn as nn
import numpy as np
import mlx.optimizers as optim
import matplotlib.pyplot as plt
from urllib.parse import urlparse
from io import BytesIO
from pathlib import Path
from huggingface_hub import snapshot_download
from shutil import copy
from mlx.utils import tree_flatten, tree_unflatten
from PIL import Image, ImageOps
from types import SimpleNamespace
from transformers import AutoTokenizer
from gte import VDB
import time
import logging
import inspect
import gradio as gr
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("transformers").setLevel(logging.ERROR)

class Tic:
    def __init__(self):
        self.last_time = time.perf_counter()

    def __call__(self):
        current_time = time.perf_counter()
        elapsed_time = current_time - self.last_time
        self.last_time = current_time
        return elapsed_time

class Streamer:
    def __init__(self, processor, stream):
        self.tokenizer = processor.tokenizer
        self.stream = stream
        self.list_tokens = []
        self.idx_sofar = 0
    def __call__(self, token):
        if not self.stream:
            self.list_tokens.append(token)
            return None
        if token.shape[0] > 1:
            self.list_tokens.append(token)
            self.stream = False
            return None
        self.list_tokens.append(token.item())
        txt = self.tokenizer.decode(self.list_tokens)
        idx_split = txt.rfind(' ', self.idx_sofar)
        if idx_split > 0:
            print(txt[self.idx_sofar:idx_split], end = '', flush=True)
            self.idx_sofar = idx_split
    def end(self):
        if self.stream:
            txt = self.tokenizer.decode(self.list_tokens)
            print(txt[self.idx_sofar:])
            return [txt], len(self.list_tokens)
        else:
            arr_tokens = mx.concatenate(self.list_tokens, axis=1)
            list_txt = [self.tokenizer.decode(i) for i in arr_tokens.tolist()]
            list_txt = [i[:i.find(self.tokenizer.eos_token)+len(self.tokenizer.eos_token)] if self.tokenizer.eos_token in i else i for i in list_txt]
            for i, gen in enumerate(list_txt):
                print(f'\n< Generated text for prompt #{i} >\n{gen}')
            return list_txt, arr_tokens.size

class LogitStopper:
    def __init__(self, max_tokens, early_stop):
        self.step = 0
        self.early_stop = early_stop if isinstance(early_stop, int) and (early_stop < max_tokens) else False
        self.log_prob_sum = 0.0
        self.best_eos_sofar = -mx.inf
        self.log_prob_sum_at_best_eos = 0.0
    def __call__(self, logits):
        if not self.early_stop:
            return False
        if logits.shape[0] > 1:
            self.early_stop = False
            return False
        log_prob = nn.log_softmax(logits)
        log_prob_best = mx.max(log_prob[:,-1,:], axis=-1).item()
        log_prob_eos = log_prob[:,-1,32007].item()

        if log_prob_eos > self.best_eos_sofar:
            self.log_prob_sum_since_last_best_eos = self.log_prob_sum - self.log_prob_sum_at_best_eos
            if ((self.log_prob_sum_since_last_best_eos) < (self.best_eos_sofar)) and (self.step > self.early_stop):
                return True
            else:
                self.best_eos_sofar = log_prob_eos
                self.log_prob_sum_at_best_eos = self.log_prob_sum
        self.log_prob_sum += log_prob_best
        self.step+=1
        return False

class TokenStopper:
    def __init__(self, processor):
        self.tokenizer = processor.tokenizer
        self.eos_id = 32007 # self.tokenizer.eos_token_id
        self.eos_rows = set()
    def __call__(self, token):
        if self.eos_id in token:
            self.eos_rows.update(np.argwhere(np.array(token)==self.eos_id)[:,0].tolist())
            if len(self.eos_rows) == token.shape[0]:
                return True
        return False

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

class Phi3VImageProcessor:
    def __init__(self):
        self.num_crops=16
        self.image_mean=np.array([0.48145466, 0.4578275, 0.40821073])
        self.image_std=np.array([0.26862954, 0.26130258, 0.27577711])
    
    def __call__(self, images):
        def HD_transform(img):
            img = img.convert('RGB')
            w, h = img.size
            trans = False
            if w < h:
                img = img.transpose(Image.TRANSPOSE)
                trans = True
                w, h = img.size
            scale = int(np.sqrt(self.num_crops * w / h))
            img = img.resize([int(scale * 336), int(scale * 336 * h / w)], Image.BILINEAR)
            def pad_to_336(b):
                _, h = b.size
                diff_height = int(np.ceil(h / 336) * 336) - h
                top_padding = int(diff_height/2)
                bottom_padding = diff_height - top_padding
                b = ImageOps.expand(b, border=(0, top_padding, 0, bottom_padding), fill=(255, 255, 255))
                return b
            img = pad_to_336(img)
            img = img.transpose(Image.TRANSPOSE) if trans else img
            img = ((np.array(img) / 255.0 - self.image_mean) / self.image_std).transpose(2,0,1)
            return img
        def pad_to_max_num_crops_tensor(images, max_crops=17):
            B, _, H, W = images.shape
            if B < max_crops:
                pad = np.zeros((max_crops - B, 3, H, W))
                images = np.concatenate([images, pad], axis=0)
            return images
        hd_images = [HD_transform(img) for img in images] 
        shapes = [[im.shape[1], im.shape[2]] for im in hd_images]
        num_img_tokens = [int((h//336*w//336+1)*144 + 1 + (h//336+1)*12) for h, w in shapes]
        # global_image = [torch.nn.functional.interpolate(torch.from_numpy(im[None]), size=(336, 336), mode='bicubic').numpy() for im in hd_images]
        global_image = [self.interpolate_336(im[None]) for im in hd_images]
        hd_images_reshape = [im
            .reshape(1, 3, h//336, 336, w//336, 336)
            .transpose(0,2,4,1,3,5)
            .reshape(-1, 3, 336, 336)
            for im, (h, w) in zip(hd_images, shapes)]
        hd_images_reshape = [np.concatenate([_global_image, _im], axis=0) for _global_image, _im in zip(global_image, hd_images_reshape)]
        image_transformed = np.stack([pad_to_max_num_crops_tensor(im) for im in hd_images_reshape], axis=0)
        return {"pixel_values": image_transformed, "image_sizes": shapes, "num_img_tokens": num_img_tokens}

    @staticmethod
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

class Phi3Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim = config.hidden_size
        self.n_heads = n_heads = config.num_attention_heads
        self.n_kv_heads = n_kv_heads = config.num_key_value_heads
        self.num_hidden_layers = config.num_hidden_layers
        self.head_dim = head_dim = config.hidden_size // n_heads
        self.scale = head_dim**-0.5
        self.chop_1 = chop_1 = self.n_heads * self.head_dim
        self.chop_2 = chop_1 + self.n_kv_heads * self.head_dim
        op_size = n_heads * head_dim + 2 * (n_kv_heads * head_dim)
        self.qkv_proj = nn.Linear(dim, op_size, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)
    def __call__(self, x, cache, cos, sin, mask):
        @mx.compile
        def _rotate_half(x, cos, sin):
            midpoint = x.shape[-1] // 2  
            x1, x2 = x[..., :midpoint], x[..., midpoint:]
            return (x * cos) + (mx.concatenate([-x2, x1], axis = -1) * sin)
        B, L, _ = x.shape
        qkv = self.qkv_proj(x)
        queries, keys, values = mx.split(qkv, [self.chop_1, self.chop_2], axis=-1)
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        queries = _rotate_half(queries, cos, sin)
        keys = _rotate_half(keys, cos, sin)
        keys, values = cache(keys, values)
        scores = (queries * self.scale) @ keys.transpose(0, 1, 3, 2)
        scores += mask
        scores = mx.softmax(scores, axis=-1)
        output = scores @ values
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)

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

    def __call__(self, x, cache, cos, sin, mask):
        r = self.self_attn(self.input_layernorm(x), cache, cos, sin, mask)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r

class SuRoPE:
    def __init__(self, config, L_all, pids):
        dim = config.hidden_size // config.num_attention_heads
        scaling_factor = math.sqrt(1 + math.log(config.max_position_embeddings / config.original_max_position_embeddings) / math.log(config.original_max_position_embeddings))
        su_factor = config.rope_scaling["long_factor"] if L_all > config.original_max_position_embeddings else config.rope_scaling["short_factor"]
        if pids is None:
            position_ids = mx.arange(L_all, dtype=mx.float32)[None]
        else:
            extended_pids = pids[:, -1][:, None] + 1 + mx.arange(L_all-pids.shape[1], dtype=mx.float32)[None, :]
            position_ids = mx.concatenate([pids, extended_pids], axis=1)
        position_ids_expanded = position_ids[:, None, :]
        inv_freq = 1.0 / (mx.array(su_factor, dtype=mx.float32) * config.rope_theta**(mx.arange(0, dim, 2, dtype=mx.float32) / dim))
        inv_freq_expanded = mx.repeat(inv_freq[None, :, None], position_ids.shape[0], axis=0)        
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(0, 2, 1)  
        emb = mx.concatenate([freqs, freqs], axis=-1)  
        self.cos = mx.expand_dims(mx.cos(emb) * scaling_factor, axis=1)
        self.sin = mx.expand_dims(mx.sin(emb) * scaling_factor, axis=1) 

    def __call__(self, past_L, new_L):
        return self.cos[:,:,past_L:past_L+new_L,:], self.sin[:,:,past_L:past_L+new_L,:]

class KVCache:
    def __init__(self, config, x, max_tokens):
        self.max_tokens = max_tokens
        self.use_quantized_cache = getattr(config, "use_quantized_cache", False)
        self.offset = 0
        shape = (2, x.shape[0], config.num_key_value_heads, x.shape[1]+max_tokens, config.hidden_size // config.num_key_value_heads)
        if self.use_quantized_cache:
            self.kv = None
        else:
            self.kv = mx.zeros(shape, mx.float32)
    def __call__(self, keys, values):
        if self.max_tokens  < 1:
            return keys, values
        B, N, L, D = keys.shape
        new_offset = self.offset + L
        if self.use_quantized_cache: 
            if self.kv is not None:
                k_cache = mx.dequantize(*self.kv[0], group_size=32).reshape((B, N, -1, D))
                v_cache = mx.dequantize(*self.kv[1], group_size=32).reshape((B, N, -1, D))
                keys = mx.concatenate([k_cache, keys], axis=2)
                values = mx.concatenate([v_cache, values], axis=2)
            self.kv = (mx.quantize(keys.reshape((B*N,-1)), group_size=32), mx.quantize(values.reshape((B*N,-1)), group_size=32))
            self.offset = new_offset
            return keys, values
        else:
            self.kv[0,:,:,self.offset:new_offset,:] = keys
            self.kv[1,:,:,self.offset:new_offset,:] = values
            self.offset = new_offset
            return self.kv[0,:,:,:new_offset,:], self.kv[1,:,:,:new_offset,:]

class Mask4D:
    def __init__(self, L_all, mask):
        mask_4d = mx.triu(mx.full((L_all, L_all), -mx.inf), k=1)[None, None]
        if mask is not None:
            pad_len = L_all - mask.shape[-1]
            mask = mx.pad(mask, ((0,0),(0,pad_len)), 1)
            mask = mx.expand_dims(mask, (1,2))
            mask = mask*mask.transpose(0,1,3,2)
            mask = mx.where(mask==1, 0, -np.inf)
            mask_4d += mask
            mask_4d = mx.repeat(mask_4d, 32, axis=1)
        self.mask_4d = mask_4d
    def __call__(self, past_L, L):
        return self.mask_4d[:,:,past_L:L+past_L,:L+past_L]

class Phi3VModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.vision_embed_tokens = Phi3ImageEmbedding(config)
        self.layers = [Phi3DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.use_quantized_cache= getattr(config, "use_quantized_cache", False)
        self.config = config
        self.num_hidden_layers = config.num_hidden_layers
    def __call__(self, input_ids, pixel_values, image_sizes, positions, cache, pids, mask, max_tokens):
        x = self.embed_tokens(input_ids)
        if pixel_values is not None:
            x = self.vision_embed_tokens(x, pixel_values, image_sizes, positions)
        if cache is None:
            cache = [KVCache(self.config, x, max_tokens) for _ in range(self.num_hidden_layers)]
            self.masker = Mask4D(x.shape[1]+max_tokens, mask)
            self.roper = SuRoPE(self.config, x.shape[1]+max_tokens, pids)
        past_L, new_L = cache[0].offset, x.shape[1]
        mask = self.masker(past_L, new_L)
        cos, sin = self.roper(past_L, new_L)
        for i, l in enumerate(self.layers):
            x = l(x, cache[i], cos, sin, mask)
        return self.norm(x), cache

class Phi3VLModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = Phi3VModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    def __call__(self, input_ids, pixel_values=None, image_sizes=None, positions=None, cache=None, pids=None, mask=None, max_tokens=0):
        x, cache = self.model(input_ids, pixel_values, image_sizes, positions, cache, pids, mask, max_tokens)
        return self.lm_head(x), cache
    @property
    def layers(self):
        return self.model.layers

def _linear_to_lora_layers(model, lora_layers, config):
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

def _load(model_path='phi3v', adapter_path=None, **kwargs):
    model_cfg = _get_cfg(f"{model_path}/config.json", **kwargs)
    model = Phi3VLModel(model_cfg)
    nn.quantize(model, model_cfg.quantized['group_size'], model_cfg.quantized['bits']) if getattr(model_cfg, 'quantized', False) else None
    model.load_weights(_get_wt(model_path, model_cfg))
    if adapter_path:
        lora_cfg = _get_cfg(f"{adapter_path}/adapter_config.json")
        _linear_to_lora_layers(model, lora_cfg.lora_layers, lora_cfg.lora_parameters)
        model.load_weights(f'{adapter_path}/adapters.safetensors', strict=False)
    mx.eval(model.parameters())
    model.eval()
    return model, Phi3VProcessor(model_path)

def _quantize(from_path='phi3v', to_path='quantized_phi3v', q_group_size=64, q_bits=4):
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
    
def _load_image(image_source): # copied from https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/utils.py
    if isinstance(image_source, BytesIO):
        try:
            return Image.open(image_source)
        except IOError as e:
            raise ValueError(f"Failed to load image from BytesIO with error: {e}")
    elif image_source.startswith(("http://", "https://")):
        try:
            response = requests.get(image_source, stream=True)
            response.raise_for_status()
            return Image.open(response.raw)
        except Exception as e:
            raise ValueError(
                f"Failed to load image from URL: {image_source} with error {e}"
            )
    elif Path(image_source).is_file():
        try:
            return Image.open(image_source)
        except IOError as e:
            raise ValueError(f"Failed to load image {image_source} with error: {e}")
    else:
        raise ValueError(
            f"The image {image_source} must be a valid URL or existing file."
        )

def _get_api_output_path(process):
    if '<|api_output|>' in process.stdout:
        return process.stdout.strip().split('<|api_output|>', 1)[1]
    else:
        return None

def _apply_chat_template(prompt, images, verbose, apply_chat_template=True):
    if apply_chat_template is False:
        print(f'### Prompt ###\n{prompt}\n### Images ###\n{images}\n### Output ###') if verbose else None
        return prompt, images
    if images is not None:
        images = [_load_image(i) for i in images] if isinstance(images, list) else [_load_image(images)]
        img_prompt = '\n'.join([f'<|image_{i+1}|>' for i in range(len(images))]) + '\n'
    else:
        img_prompt = ''
    prompt = [prompt] if isinstance(prompt, str) else prompt
    prompt = [f"<|user|>\n{img_prompt}{i}<|end|>\n<|assistant|>\n" for i in prompt]
    if verbose:
        prompt_str = "\n".join(map(str.strip, prompt)).strip()
        images_str = "\n".join(map(str, images)) if images else "None"
        print(f'### Prompt ###\n{prompt_str}\n### Images ###\n{images_str}\n### Output ###')
    prompt = prompt[0] if len(prompt) == 1 else prompt
    return prompt, images

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

def _generate(model, processor, prompt, images=None, max_tokens=1000, verbose=True, return_tps=False, early_stop=False, stream=True):
    if images is not None and isinstance(prompt, list):
        raise ValueError('Images cannot be provided when prompt is a list')
    tic = Tic()
    logit_stopper = LogitStopper(max_tokens, early_stop)
    token_stopper = TokenStopper(processor)
    streamer = Streamer(processor, stream)
    dict_input = processor(images, prompt)
    logits, cache = model(**dict_input, max_tokens=max_tokens)
    token = mx.argmax(logits[:, -1, :], axis=-1)[:,None]
    mx.eval(token)
    streamer(token)
    mask, pids = dict_input.get('mask', None), dict_input.get('pids', None)
    prompt_time = tic()
    for i in range(max_tokens-1):
        logits, cache = model(input_ids=token, cache=cache, mask=mask, pids=pids)
        token = mx.argmax(logits[:, -1, :], axis=-1)[:,None]
        mx.eval(token)
        streamer(token)
        if logit_stopper(logits):
            break
        if token_stopper(token):
            break
    result, gen_len = streamer.end()
    gen_time = tic()
    prompt_len = dict_input['input_ids'].size 
    prompt_tps = prompt_len / prompt_time
    gen_tps = (gen_len - 1) / gen_time
    if verbose:
        print(f"\nPrompt: {prompt_tps:.2f} tokens-per-sec ({prompt_len} tokens / {prompt_time:.1f} sec)")
        print(f"Generation: {gen_tps:.2f} tokens-per-sec ({gen_len} tokens / {gen_time:.1f} sec)")
    if return_tps:
        return prompt_tps, gen_tps
    return result
    
def _execute(code_string, file_prefix=0):
    code_string = '\n'.join(re.findall(r"```python\n(.*?)```", code_string, re.DOTALL))
    if len(code_string) < 1:
        return None, None, None, None
    code_string = re.sub(r'plt\.savefig\(.*?\)', 'plt.show()', code_string)
    plot_path = f'{file_prefix}.png' if 'plt.show()' in code_string else None
    code_to_run = code_string.replace("plt.show()", f"plt.savefig('{plot_path}')")
    process = subprocess.run(["python", "-c", code_to_run], capture_output=True, text=True)
    output_path = None
    stderr = process.stderr.strip()
    if len(stderr) < 1:
        output_path = plot_path if plot_path else _get_api_output_path(process)
        stderr = None    
    return code_string.strip(), output_path, process.stdout.strip(), stderr

def load_text(file_path):
    file_path = file_path.strip()
    parsed_url = urlparse(file_path)
    if parsed_url.scheme in ('http', 'https'):
        response = requests.get(file_path)
        if response.status_code == 200:
            return_text = response.text
        else:
            raise Exception(f"Failed to retrieve URL: {file_path}, Status code: {response.status_code}")
    else:
        path = Path(file_path)
        if path.is_file():
            return_text = path.read_text()
        else:
            return_text = file_path
    return return_text.replace('"', "'")

def chatui(agent=None):
    agent = Agent() if agent is None else agent
    def add_message(history, message):
        for x in message["files"]:
            history.append(((x,), None))
        if message["text"] is not None:
            history.append((message["text"], None))
        return history, gr.MultimodalTextbox(value=None, interactive=False)

    def bot(history):
        def _get_input(history):
            return history[-1][0], [i[0][0] for i in history[agent.user_since:-1]] if agent.user_since+1 < len(history) else None
        agent_input = _get_input(history)
        agent_output = agent(*agent_input)
        responses, files = agent_output['responses'], agent_output['files']
        if responses is not None:
            for response in responses:
                response = response[:response.find('<|end|>')] if '<|end|>' in response else response
                lines = response.splitlines()
                non_empty_lines = [line for line in lines if line.strip()]
                response = '\n'.join(non_empty_lines)
                history.append((None, response))
        if files is not None:
            for file in files:
                if file is not None:
                    history.append((None, (file,)))
        agent.user_since = len(history)
        return history

    def reset():
        agent.end()
        return []

    with gr.Blocks(css="footer{display:none !important}") as demo:
        chatbot = gr.Chatbot(
            [],
            elem_id="chatbot",
            bubble_full_width=False,
            height='80vh'
        )

        chat_input = gr.MultimodalTextbox(interactive=True, file_types=["image"], placeholder="Enter message or upload file...", show_label=False)

        close_btn = gr.Button("Reset", variant="stop")

        chat_msg = chat_input.submit(add_message, [chatbot, chat_input], [chatbot, chat_input])
        bot_msg = chat_msg.then(bot, chatbot, chatbot, api_name="bot_response")
        bot_msg.then(lambda: gr.MultimodalTextbox(interactive=True), None, [chat_input])

        close_btn.click(reset, None, chatbot) 

    demo.queue()
    demo.launch(inbrowser=True, inline=True)

def train_lora(lora_layers=5, lora_rank=16, epochs=10, lr=1e-4, warmup=.5, mask_ratios=[.0], adapter_path='adapters', dataset_path="JosefAlbers/akemiH_MedQA_Reason"): # or mask_ratios=[.0, .1, .3, .5]
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

    def _get_lr_schedule(lr, steps, warmup):
        n_warmup = int(steps*warmup)
        return mx.concatenate([mx.linspace(1e-6, lr, n_warmup), mx.linspace(lr, 1e-6, steps - n_warmup + 1)[1:]])

    model, processor = load()
    ds = datasets.load_dataset(dataset_path, split='train').take(10) # `debug
    ds = ds.map(_prompt).select_columns(['input_tokens', 'idx_assistant'])
    ds = datasets.concatenate_datasets([ds]*epochs)
    steps = len(ds)
    lora_cfg = _set_lora(lora_layers, lora_rank, adapter_path)
    model.freeze()    
    _linear_to_lora_layers(model, lora_cfg['lora_layers'], lora_cfg['lora_parameters'])
    model.train()
    distil_loss_value_and_grad = nn.value_and_grad(model, _loss)
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

def test_lora(dataset_path="JosefAlbers/akemiH_MedQA_Reason"):
    model, processor = load(adapter_path='adapters')
    def _get_alphabet(text):
        try:
            return "".join([char for char in text if char.isalpha()]).upper()[0]
        except:
            return ""
    def _recall(example):
        question = example['input']
        _question = question.rsplit(' A: ', 1)[0].strip()
        prompt_recall = f"<|user|>\n{_question}<|end|>\n<|assistant|>"
        example['recall'] = _generate(model, processor, prompt_recall, max_tokens=30, verbose=False)[0]
        prompt_answer = f"<|user|>\n{question}<|end|>\n<|assistant|>\nThe correct answer is"
        example['attempt'] = _get_alphabet(_generate(model, processor, prompt_answer, max_tokens=3, verbose=False)[0])
        _summary = example['summary'].strip().split('\n', 1)[0].strip()
        example['result'] = f"Question: {_question}\n- Taught: {_summary}\n- Recall: {example['recall']}\n- Answer: {example['output']}\n- Attempt: {example['attempt']}\n- Correct: {example['output'] == example['attempt']}"
        return example
    ds = datasets.load_dataset(dataset_path, split='train').take(10)
    ds = ds.map(_recall)
    num_recall = len(ds.filter(lambda x: x["output"] == x["attempt"]))
    print('\n'.join(ds['result']), f'\n---\nFinal Score: {num_recall/len(ds)}({num_recall}/{len(ds)})')
    del model
    del processor


def benchmark():
    prompts = [
        ('Write a cosmic horror.', ),
        ('What is shown in this image?', 'https://assets-c4akfrf5b4d3f4b7.z01.azurefd.net/assets/2024/04/BMDataViz_661fb89f3845e.png'),
        ([
            "Write an executive summary for a communications business plan",
            "Write a resume.", 
            "Write a mystery horror.",
            "Write a Neurology ICU Admission Note."
            ], None),
    ]
    results = {
        'vanilla': [],
        'q_model': [],
        'q_cache': [],
        'lora': [],
    }

    for i, prompt in enumerate(prompts):
        for method in results:
            kwargs = {'return_tps': True}
            if method == 'q_model':
                kwargs['quantize_model'] = True
            elif method == 'q_cache':
                kwargs['quantize_cache'] = True
            elif method == 'lora':
                kwargs['adapter_path'] = 'adapters'

            prompt_tps, gen_tps = generate(*prompt, max_tokens=100, early_stop=100, **kwargs)
            results[method].append([i, prompt_tps, gen_tps])

    with open('benchmark.json', 'w') as f:
        json.dump(results, f, indent=4)

def add_code(prompt, codes):
    return prompt if codes is None else [f'{prompt}\n\n```python\n{code}\n```\n' for code in codes]

def load(quantize_model=False, quantize_cache=False, adapter_path=None, **kwargs):
    if not os.path.exists('phi3v'):
        snapshot_download(repo_id="microsoft/Phi-3-vision-128k-instruct", allow_patterns=["*.safetensors", "*.json"], local_dir='phi3v')
        _quantize()
        train_lora(1,1,1)
    model_path='quantized_phi3v' if quantize_model is True else 'phi3v'
    return _load(model_path=model_path, use_quantized_cache=quantize_cache, adapter_path=adapter_path)

def get_api(prompt, n_topk=1, verbose=True):
    vdb = VDB()
    codes = vdb(prompt)
    prompt = [prompt] if isinstance(prompt, str) else prompt
    codes = [code.format(prompt=prompt[i].split('<|api_input|>')[1]) for i, sublist in enumerate(codes) for code in sublist]
    print('Obtained api codes:\n', codes) if verbose is True else None
    return codes

def generate(prompt, images=None, preload=None, quantize_model=False, quantize_cache=False, adapter_path=None, max_tokens=1000, verbose=True, return_tps=False, early_stop=False, stream=True, apply_chat_template = True):
    if '<|api_input|>' in prompt:
        return get_api(prompt)
    preload = load(quantize_model, quantize_cache, adapter_path) if preload is None else preload
    return _generate(*preload, *_apply_chat_template(prompt, images, verbose, apply_chat_template), max_tokens=max_tokens, verbose=verbose, return_tps=return_tps, early_stop=early_stop, stream=stream)

def execute(code_strings, file_prefix=0, verbose=True):
    results = [_execute(code_string, f'{file_prefix}_{i}') for i, code_string in enumerate(code_strings)]
    print('Execution results:', results) if verbose is True else None
    return {k: [r[i] for r in results] for i, k in enumerate(['codes', 'files', 'souts', 'serrs'])}

class Agent:
    _default_toolchain = """
        prompt = add_code(prompt, codes)
        responses = generate(prompt, images)
        files, codes = execute(responses, step)
        """
    def __init__(self, toolchain=None, enable_api=True, **kwargs):
        toolchain = self._default_toolchain if toolchain is None else toolchain
        self.set_toolchain(toolchain)
        self.kwargs = kwargs|{'preload':load(**kwargs)}
        self.enable_api = enable_api
        self.reset()
        
    def __call__(self, prompt:str, images=None):
        prompt = prompt.replace('"', '<|api_input|>') if self.enable_api else prompt
        self.ongoing.update({'prompt':prompt})
        if images is not None:
            self.ongoing.update({'images':images})
        for tool in self.toolchain:
            _returned = tool['fxn'](*[self.ongoing.get(i, None) for i in tool['args']], **{k:v for k,v in self.kwargs.items() if k in inspect.signature(tool['fxn']).parameters.keys()})
            if isinstance(_returned, dict):
                self.ongoing.update({k:_returned[k] for k in tool['out']})
            else:
                self.ongoing.update({k:_returned for k in tool['out']})
        self.log_step()
        return {i:self.ongoing.get(i, None) for i in self.list_outs}

    def reset(self):
        self.log = []
        self.ongoing = {'step':0}
        self.user_since = 0

    def log_step(self):
        self.log.append({**self.ongoing})
        with open(f'agent_log.json', "w") as f:
            json.dump(self.log, f, indent=4)
        self.ongoing = {k:None if v==[None] else v for k,v in self.ongoing.items()}
        self.ongoing['step']+=1
    
    def end(self):
        self.ongoing.update({'END':'END'})
        self.log_step()
        self.reset()

    def set_toolchain(self, s):
        def _parse_toolchain(s):
            s = s.strip().rstrip(')')
            out_part, fxn_part = s.split('=')
            fxn_name, args_part = fxn_part.split('(')
            
            return {
                'fxn': eval(fxn_name.strip()),
                'args': [arg.strip() for arg in args_part.split(',')],
                'out': [out.strip() for out in out_part.split(',')]
            }

        def _parse_return(s):
            if 'return ' not in s:
                return ['responses', 'files']
            return [i.strip() for i in s.split('return ')[1].split(',')]

        self.toolchain = [_parse_toolchain(i) for i in s.split('\n') if '=' in i]
        self.list_outs = _parse_return(s)
