import math
import json
import glob
import os
import re
import requests
import datasets
import random
import subprocess
from functools import partial
import mlx.core as mx
import mlx.nn as nn
import numpy as np
import mlx.optimizers as optim
import matplotlib.pyplot as plt
from urllib.parse import urlparse
from io import BytesIO
from pathlib import Path
from huggingface_hub import snapshot_download, InferenceClient
from shutil import copy
from mlx.utils import tree_flatten, tree_unflatten
from PIL import Image #, ImageOps
from types import SimpleNamespace
# from transformers import AutoTokenizer
import time
import logging
import inspect
import gradio as gr
from gradio_client import Client

from gte import VDB
from phi import Phi3FProcessor, Phi3VProcessor, Phi3ForCausalLM, Phi3VForCausalLM, LoRALinear

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("transformers").setLevel(logging.ERROR)

PATH_ADAPTERS = 'adapters'
PATH_ORIGINAL_PHI3_VISION  = 'models/phi3_v'
PATH_QUANTIZED_PHI3_VISION = 'models/phi3_v_Q'
PATH_ORIGINAL_PHI3_BLIND   = 'models/phi3_mini_128k'
PATH_QUANTIZED_PHI3_BLIND  = 'models/phi3_mini_128k_Q'
ID_EOS = 32007
ID_ASS = 32001

class Tic:
    def __init__(self):
        self.last_time = time.perf_counter()

    def __call__(self):
        current_time = time.perf_counter()
        elapsed_time = current_time - self.last_time
        self.last_time = current_time
        return elapsed_time

class Streamer:
    def __init__(self, processor, stream, mute):
        self.tokenizer = processor.tokenizer
        self.mute = mute
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
            list_txt = [self.tokenizer.decode(i[:i.index(ID_EOS)+1] if ID_EOS in i else i) for i in arr_tokens.tolist()]
            if self.mute is False:
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
        log_prob = nn.log_softmax(logits[:,-1,:])
        log_prob_best = mx.max(log_prob, axis=-1).item()
        log_prob_eos = log_prob[:,ID_EOS].item()
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
        self.eos_id = ID_EOS
        self.eos_rows = set()
    def __call__(self, token):
        if self.eos_id in token:
            self.eos_rows.update(np.argwhere(np.array(token)==self.eos_id)[:,0].tolist())
            if len(self.eos_rows) == token.shape[0]:
                return True
        return False

class TrainingCallback:
    def __init__(self, lora_cfg, lr_schedule, batch_indices, sum_every=3):
        self.batch_indices = batch_indices
        self.lora_cfg = lora_cfg
        self.adapter_path = lora_cfg['adapter_path']
        self.lr_schedule = lr_schedule
        self.sum_every = min(sum_every, len(batch_indices))
        self.current_step = 0
        self.sum_loss = .0
        self.best_loss = math.inf
        self.train_log = {'step_i': [], 'step_loss': [], 'avg_i': [], 'avg_loss': []}
        self.start_time = time.perf_counter()
        os.makedirs(self.adapter_path, exist_ok=True)

    def __call__(self, model, lvalue):
        self.current_step += 1
        step_loss = lvalue.item()
        print(f'- Step loss at step {self.current_step}: {step_loss:.2f}')
        self.train_log['step_i'].append(self.current_step)
        self.train_log['step_loss'].append(step_loss)
        self.sum_loss += step_loss

        if self.current_step % self.sum_every == 0:
            avg_loss = self.sum_loss / self.sum_every
            self.sum_loss = 0.0
            self.train_log['avg_i'].append(self.current_step)
            self.train_log['avg_loss'].append(avg_loss)
            print(f'Avg loss at step {self.current_step}: {avg_loss:.2f}')
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
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
        ax1.plot(train_log['step_i'], train_log['step_loss'], color='b', alpha=0.5, label='Step Loss')
        ax1.plot(train_log['avg_i'], train_log['avg_loss'], color='r', label='Avg Loss')
        ax1.set_title('Training Loss Curves')
        ax1.legend()
        ax2.plot(self.lr_schedule)
        ax2.ticklabel_format(axis='y', style='sci')
        ax2.set_title('Learning Rate Schedule')
        batch_indices = np.array(self.batch_indices)
        batch_numbers = np.arange(len(batch_indices))
        x = np.repeat(batch_numbers, [len(sublist) for sublist in batch_indices])
        y = np.concatenate(batch_indices)
        ax3.scatter(x,y, color='b', marker='.', alpha=0.5)
        ax3.set_title('Batch Indices')
        plt.tight_layout()
        fig.savefig(f'train_log_{self.current_step}_steps_in_{train_log["train_time"]:.0f}_sec.png')
        print(f"Training log saved to {self.adapter_path}")
        print(f"Total training time: {train_log['train_time']:.2f} seconds")

# class LoRALinear(nn.Module): # copied from mlx-examples (https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/tuner/lora.py)
#     @staticmethod
#     def from_linear(
#         linear: nn.Linear,
#         r: int = 8,
#         alpha: float = 16,
#         dropout: float = 0.0,
#         scale: float = 10.0,
#     ):
#         output_dims, input_dims = linear.weight.shape
#         if isinstance(linear, nn.QuantizedLinear):
#             input_dims *= 32 // linear.bits
#         lora_lin = LoRALinear(
#             input_dims=input_dims,
#             output_dims=output_dims,
#             r=r,
#             alpha=alpha,
#             dropout=dropout,
#             scale=scale,
#         )
#         lora_lin.linear = linear
#         return lora_lin

#     def __init__(
#         self,
#         input_dims: int,
#         output_dims: int,
#         r: int = 8,
#         alpha: float = 16,
#         dropout: float = 0.0,
#         scale: float = 10.0,
#         bias: bool = False,
#     ):
#         super().__init__()
#         self.linear = nn.Linear(input_dims, output_dims, bias=bias)
#         self.dropout = nn.Dropout(p=dropout)
#         self.scale = scale * (alpha / r)
#         scale = 1 / math.sqrt(input_dims)
#         self.lora_a = mx.random.uniform(
#             low=-scale,
#             high=scale,
#             shape=(input_dims, r),
#         )
#         self.lora_b = mx.zeros(shape=(r, output_dims))

#     def __call__(self, x):
#         y = self.linear(x)
#         z = (self.dropout(x) @ self.lora_a) @ self.lora_b
#         return y + (self.scale * z).astype(x.dtype)

# class ClipAttention(nn.Module):
#     def __init__(self, dims, num_heads, bias=True):
#         super().__init__()
#         self.num_heads = num_heads
#         self.scale = (dims // num_heads) ** -0.5
#         self.q_proj = nn.Linear(dims, dims, bias=bias)
#         self.k_proj = nn.Linear(dims, dims, bias=bias)
#         self.v_proj = nn.Linear(dims, dims, bias=bias)
#         self.out_proj = nn.Linear(dims, dims, bias=bias)

#     def __call__(self, x):
#         B, L = x.shape[:2]
#         queries, keys, values = (proj(x).reshape(B, L, self.num_heads, -1).transpose(0, 2, 1, 3) for proj in (self.q_proj, self.k_proj, self.v_proj))
#         output = mx.fast.scaled_dot_product_attention(queries, keys, values, scale=self.scale)
#         return self.out_proj(output.transpose(0, 2, 1, 3).reshape(B, L, -1))

# class ClipMLP(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.activation_fn = nn.gelu_fast_approx
#         self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
#         self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

#     def __call__(self, x):
#         return self.fc2(self.activation_fn(self.fc1(x)))

# class ClipEncoderLayer(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.self_attn = ClipAttention(config.hidden_size, config.num_attention_heads, bias=True)
#         self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
#         self.mlp = ClipMLP(config)
#         self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

#     def __call__(self, x):
#         x = x + self.self_attn(self.layer_norm1(x))
#         return x + self.mlp(self.layer_norm2(x))

# class ClipEncoder(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.layers = [ClipEncoderLayer(config) for _ in range(config.num_hidden_layers)]

# class ClipEmbeddings(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
#         self.embed_dim = config.hidden_size
#         self.image_size = config.image_size
#         self.patch_size = config.patch_size
#         self.class_embedding = mx.zeros(config.hidden_size)
#         self.patch_embedding = nn.Conv2d(
#             in_channels=config.num_channels,
#             out_channels=self.embed_dim,
#             kernel_size=self.patch_size,
#             stride=self.patch_size,
#             bias=False,
#         )
#         self.num_patches = (self.image_size // self.patch_size) ** 2
#         self.num_positions = self.num_patches + 1
#         self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)

#     def __call__(self, x):
#         batch_size = x.shape[0]
#         patch_embeddings = self.patch_embedding(x)
#         patch_embeddings = mx.flatten(patch_embeddings, start_axis=1, end_axis=2)
#         embed_dim = patch_embeddings.shape[-1]
#         cls_embeddings = mx.broadcast_to(self.class_embedding, (batch_size, 1, embed_dim))
#         position_ids = mx.arange(self.num_positions)[None]
#         embeddings = mx.concatenate((cls_embeddings, patch_embeddings), axis=1)
#         embeddings += self.position_embedding(position_ids)
#         return embeddings

# class ClipModel(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.embeddings = ClipEmbeddings(config)
#         self.pre_layrnorm = nn.LayerNorm(config.hidden_size)
#         self.encoder = ClipEncoder(config)
#         self.post_layernorm = nn.LayerNorm(config.hidden_size)

#     def __call__(self, x):
#         x = self.embeddings(x)
#         x = self.pre_layrnorm(x)
#         for l in self.encoder.layers[:-1]:
#             x = l(x)
#         return x[:, 1:]

# class ClipVModel(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.vision_model = ClipModel(config)

# class Phi3FProcessor:
#     def __init__(self, local_dir, return_mx=True):
#         self.tokenizer = AutoTokenizer.from_pretrained(local_dir)
#         self.return_mx = return_mx

#     def _tokenize(self, texts):
#         if isinstance(texts, str):
#             return {'input_ids': mx.array(self.tokenizer(texts).input_ids)[None]}
#         input_ids = self.tokenizer(texts).input_ids
#         max_length = max(len(sublist) for sublist in input_ids)
#         position_ids =[[1]*(max_length-len(sublist)) + list(range(len(sublist))) for sublist in input_ids]
#         attention_masks = [[0]*(max_length-len(sublist)) + [1]*len(sublist) for sublist in input_ids]
#         input_ids = [[0]*(max_length-len(sublist)) + sublist for sublist in input_ids]
#         if self.return_mx:
#             input_ids = mx.array(input_ids)
#             position_ids = mx.array(position_ids)
#             attention_masks = mx.array(attention_masks)
#         return {'input_ids':input_ids, 'pids':position_ids, 'mask':attention_masks}

#     def __call__(self, texts, images=None):
#         if images is not None:
#             print(f'WARNING: You are using phi3_mini_128k. Use phi3_v for VLM tasks.')
#         return self._tokenize(texts)

# class Phi3VProcessor(Phi3FProcessor):
#     def __init__(self, local_dir, return_mx=True):
#         super().__init__(local_dir, return_mx)
#         self.img_processor = Phi3VImageProcessor()

#     def __call__(self, texts, images=None):
#         if images is None:
#             return self._tokenize(texts)
#         image_inputs = self.img_processor(images)
#         return self._merge(image_inputs, texts)

#     def _merge(self, images, texts):
#         pattern = r"<\|image_\d+\|>"
#         prompt_chunks = self.tokenizer(re.split(pattern, texts)).input_ids
#         num_img_tokens = images['num_img_tokens']
#         images, image_sizes = images['pixel_values'], images['image_sizes']
#         image_tags = re.findall(pattern, texts)
#         image_ids = [int(s.split("|")[1].split("_")[-1]) for s in image_tags]
#         image_ids_pad = [[-iid]*num_img_tokens[iid-1] for iid in image_ids]
#         image_ids_pad = image_ids_pad + [[]] if len(prompt_chunks) > len(image_ids_pad) else image_ids_pad
#         input_ids = []
#         for chunk, pad in zip(prompt_chunks, image_ids_pad):
#             input_ids.extend(chunk)
#             input_ids.extend(pad)
#         input_ids = np.array(input_ids)[None]
#         positions = np.argwhere(input_ids < 0)
#         return {"input_ids": mx.array(input_ids),
#                 "pixel_values": mx.array(images),
#                 "image_sizes": mx.array(image_sizes),
#                 "positions": mx.array(positions)}

# class Phi3VImageProcessor:
#     def __init__(self):
#         self.num_crops=16
#         self.image_mean=np.array([0.48145466, 0.4578275, 0.40821073])
#         self.image_std=np.array([0.26862954, 0.26130258, 0.27577711])

#     def __call__(self, images):
#         def HD_transform(img):
#             img = img.convert('RGB')
#             w, h = img.size
#             trans = False
#             if w < h:
#                 img = img.transpose(Image.TRANSPOSE)
#                 trans = True
#                 w, h = img.size
#             scale = int(np.sqrt(self.num_crops * w / h))
#             img = img.resize([int(scale * 336), int(scale * 336 * h / w)], Image.BILINEAR)
#             def pad_to_336(b):
#                 _, h = b.size
#                 diff_height = int(np.ceil(h / 336) * 336) - h
#                 top_padding = int(diff_height/2)
#                 bottom_padding = diff_height - top_padding
#                 b = ImageOps.expand(b, border=(0, top_padding, 0, bottom_padding), fill=(255, 255, 255))
#                 return b
#             img = pad_to_336(img)
#             img = img.transpose(Image.TRANSPOSE) if trans else img
#             img = ((np.array(img) / 255.0 - self.image_mean) / self.image_std).transpose(2,0,1)
#             return img
#         def pad_to_max_num_crops_tensor(images, max_crops=17):
#             B, _, H, W = images.shape
#             if B < max_crops:
#                 pad = np.zeros((max_crops - B, 3, H, W))
#                 images = np.concatenate([images, pad], axis=0)
#             return images
#         hd_images = [HD_transform(img) for img in images]
#         shapes = [[im.shape[1], im.shape[2]] for im in hd_images]
#         num_img_tokens = [int((h//336*w//336+1)*144 + 1 + (h//336+1)*12) for h, w in shapes]
#         # global_image = [torch.nn.functional.interpolate(torch.from_numpy(im[None]), size=(336, 336), mode='bicubic').numpy() for im in hd_images]
#         global_image = [self.interpolate_336(im[None]) for im in hd_images]
#         hd_images_reshape = [im
#             .reshape(1, 3, h//336, 336, w//336, 336)
#             .transpose(0,2,4,1,3,5)
#             .reshape(-1, 3, 336, 336)
#             for im, (h, w) in zip(hd_images, shapes)]
#         hd_images_reshape = [np.concatenate([_global_image, _im], axis=0) for _global_image, _im in zip(global_image, hd_images_reshape)]
#         image_transformed = np.stack([pad_to_max_num_crops_tensor(im) for im in hd_images_reshape], axis=0)
#         return {"pixel_values": image_transformed, "image_sizes": shapes, "num_img_tokens": num_img_tokens}

#     @staticmethod
#     def interpolate_336(input):
#         def get_weights_and_indices(scale, out_size, in_size):
#             def cubic(x):
#                 abs_x = np.abs(x)
#                 abs_x2 = abs_x ** 2
#                 abs_x3 = abs_x ** 3
#                 f = ((1.5 * abs_x3 - 2.5 * abs_x2 + 1) * (abs_x <= 1) +
#                     (-0.5 * abs_x3 + 2.5 * abs_x2 - 4 * abs_x + 2) * ((abs_x > 1) & (abs_x <= 2)))
#                 return f
#             kernel_radius = 2
#             kernel_width = kernel_radius * 2
#             out_coordinates = np.linspace(0, in_size - 1, out_size)
#             in_coordinates = out_coordinates / scale
#             left_indices = np.floor(in_coordinates - 0.5).astype(np.int32)
#             right_indices = left_indices + 1
#             left_indices = np.clip(left_indices, 0, in_size - 1)
#             right_indices = np.clip(right_indices, 0, in_size - 1)
#             weights = np.zeros((out_size, kernel_width), dtype=np.float32)
#             indices = np.zeros((out_size, kernel_width), dtype=np.int32)
#             for i in range(out_size):
#                 indices[i, 0] = left_indices[i]
#                 indices[i, 1] = right_indices[i]
#                 weights[i, 0] = cubic(in_coordinates[i] - left_indices[i])
#                 weights[i, 1] = cubic(right_indices[i] - in_coordinates[i])
#                 weight_sum = weights[i].sum()
#                 if weight_sum != 0:
#                     weights[i] /= weight_sum
#             return weights, indices
#         N, C, H, W = input.shape
#         out_hw = 336
#         output = np.zeros((N, C, out_hw, out_hw), dtype=input.dtype)
#         h_weights, h_indices = get_weights_and_indices(out_hw / H, out_hw, H)
#         w_weights, w_indices = get_weights_and_indices(out_hw / W, out_hw, W)
#         for n in range(N):
#             for c in range(C):
#                 for i in range(out_hw):
#                     for j in range(out_hw):
#                         h_kernel = input[n, c, h_indices[i]]
#                         w_kernel = h_kernel[:, w_indices[j]]
#                         output[n, c, i, j] = np.sum(h_weights[i][:, None] * w_weights[j] * w_kernel)
#         return output

# class Phi3ImageEmbedding(nn.Module):
#     CLIP_VIT_LARGE_PATCH14_336_CONFIG = SimpleNamespace(
#         hidden_size=1024,
#         image_size=336,
#         intermediate_size=4096,
#         layer_norm_eps=1e-05,
#         num_attention_heads=16,
#         num_channels=3,
#         num_hidden_layers=24,
#         patch_size=14,
#         )
#     def __init__(self, config):
#         super().__init__()
#         self.img_processor = ClipVModel(self.CLIP_VIT_LARGE_PATCH14_336_CONFIG)
#         self.image_dim_out = image_dim_out = config.img_processor['image_dim_out']
#         self.glb_GN = mx.zeros([1, 1, image_dim_out * 4])
#         self.sub_GN = mx.zeros([1, 1, 1, image_dim_out * 4])
#         self.img_projection = [nn.Linear(image_dim_out * 4, config.hidden_size), nn.GELU(), nn.Linear(config.hidden_size, config.hidden_size)]

#     def __call__(self, txt_embeds, img_embeds, img_sizes, positions):
#         B = img_embeds.shape[0]
#         img_sizes, positions = (img_sizes // 336).tolist(), positions.tolist()
#         img_features = self.img_processor.vision_model(img_embeds.reshape(-1, *img_embeds.shape[2:]).transpose(0, 2, 3, 1))
#         img_features = img_features.reshape(B, -1, *img_features.shape[1:])
#         C, H = self.image_dim_out, int(img_features.shape[2] ** 0.5)
#         output_imgs, output_len = [], []
#         for _bs in range(B):
#             h, w = img_sizes[_bs]
#             B_ = h * w
#             def _reshape_and_concatenate(img, shape, tile_shape):
#                 return mx.concatenate([img.reshape(shape).transpose(0, 1, 3, 2, 4, 5).reshape(tile_shape), mx.tile(self.sub_GN, (1, tile_shape[1], 1, 1))], axis=2).reshape(1, -1, 4 * C)
#             glb_img = _reshape_and_concatenate( img_features[_bs, :1], (1, H//2, 2, H//2, 2, C), (1, H//2, H//2, 4*C) )
#             sub_img = _reshape_and_concatenate( img_features[_bs, 1:B_+1], (B_, H//2, 2, H//2, 2, C), (1, h*12, w*12, 4*C) )
#             x = mx.concatenate([sub_img, self.glb_GN, glb_img], axis=1)
#             for l in self.img_projection:
#                 x = l(x)
#             output_imgs.append(x)
#             output_len.append(int((h*w + 1) * 144 + 1 + (h + 1) * 12))
#         idx = 0
#         for i, cnt in enumerate(output_len):
#             txt_embeds[positions[idx][0], positions[idx][1] : positions[idx][1] + cnt] = output_imgs[i]
#             idx += cnt
#         return txt_embeds

# class Phi3Attention(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         dim = config.hidden_size
#         self.n_heads = n_heads = config.num_attention_heads
#         self.n_kv_heads = n_kv_heads = config.num_key_value_heads
#         self.num_hidden_layers = config.num_hidden_layers
#         self.head_dim = head_dim = config.hidden_size // n_heads
#         self.scale = head_dim**-0.5
#         self.chop_1 = chop_1 = self.n_heads * self.head_dim
#         self.chop_2 = chop_1 + self.n_kv_heads * self.head_dim
#         op_size = n_heads * head_dim + 2 * (n_kv_heads * head_dim)
#         self.qkv_proj = nn.Linear(dim, op_size, bias=False)
#         self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)

#     def __call__(self, x, cache, cos, sin, mask):
#         @mx.compile
#         def _rotate_half(x, cos, sin):
#             midpoint = x.shape[-1] // 2
#             x1, x2 = x[..., :midpoint], x[..., midpoint:]
#             return (x * cos) + (mx.concatenate([-x2, x1], axis = -1) * sin)
#         B, L, _ = x.shape
#         qkv = self.qkv_proj(x)
#         queries, keys, values = mx.split(qkv, [self.chop_1, self.chop_2], axis=-1)
#         queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
#         keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
#         values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
#         queries = _rotate_half(queries, cos, sin)
#         keys = _rotate_half(keys, cos, sin)
#         keys, values = cache(keys, values)
#         scores = (queries * self.scale) @ keys.transpose(0, 1, 3, 2)
#         scores += mask
#         scores = mx.softmax(scores, axis=-1)
#         output = scores @ values
#         output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
#         return self.o_proj(output)

# class Phi3MLP(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.gate_up_proj = nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias=False)
#         self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

#     def __call__(self, x):
#         x = self.gate_up_proj(x)
#         gate, x = mx.split(x, 2, axis=-1)
#         return self.down_proj(nn.silu(gate) * x)

# class Phi3DecoderLayer(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.self_attn = Phi3Attention(config)
#         self.mlp = Phi3MLP(config)
#         self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
#         self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

#     def __call__(self, x, cache, cos, sin, mask):
#         r = self.self_attn(self.input_layernorm(x), cache, cos, sin, mask)
#         h = x + r
#         r = self.mlp(self.post_attention_layernorm(h))
#         return h + r

# class SuRoPE:
#     def __init__(self, config, L_all, pids):
#         dim = config.hidden_size // config.num_attention_heads
#         scaling_factor = math.sqrt(1 + math.log(config.max_position_embeddings / config.original_max_position_embeddings) / math.log(config.original_max_position_embeddings))
#         su_factor = config.rope_scaling["long_factor"] if L_all > config.original_max_position_embeddings else config.rope_scaling["short_factor"]
#         if pids is None:
#             position_ids = mx.arange(L_all, dtype=mx.float32)[None]
#         else:
#             extended_pids = pids[:, -1][:, None] + 1 + mx.arange(L_all-pids.shape[1], dtype=mx.float32)[None, :]
#             position_ids = mx.concatenate([pids, extended_pids], axis=1)
#         position_ids_expanded = position_ids[:, None, :]
#         inv_freq = 1.0 / (mx.array(su_factor, dtype=mx.float32) * config.rope_theta**(mx.arange(0, dim, 2, dtype=mx.float32) / dim))
#         inv_freq_expanded = mx.repeat(inv_freq[None, :, None], position_ids.shape[0], axis=0)
#         freqs = (inv_freq_expanded @ position_ids_expanded).transpose(0, 2, 1)
#         emb = mx.concatenate([freqs, freqs], axis=-1)
#         self.cos = mx.expand_dims(mx.cos(emb) * scaling_factor, axis=1)
#         self.sin = mx.expand_dims(mx.sin(emb) * scaling_factor, axis=1)

#     def __call__(self, past_L, new_L):
#         return self.cos[:,:,past_L:past_L+new_L,:], self.sin[:,:,past_L:past_L+new_L,:]

# class KVCache:
#     def __init__(self, config, x, max_tokens):
#         self.max_tokens = max_tokens
#         self.use_quantized_cache = getattr(config, "use_quantized_cache", False)
#         self.offset = 0
#         shape = (2, x.shape[0], config.num_key_value_heads, x.shape[1]+max_tokens, config.hidden_size // config.num_key_value_heads)
#         if self.use_quantized_cache or max_tokens < 1:
#             self.kv = None
#         else:
#             self.kv = mx.zeros(shape, mx.float32)

#     def __call__(self, keys, values):
#         if self.max_tokens < 1:
#             return keys, values
#         B, N, L, D = keys.shape
#         new_offset = self.offset + L
#         if self.use_quantized_cache:
#             if self.kv is not None:
#                 k_cache = mx.dequantize(*self.kv[0], group_size=32).reshape((B, N, -1, D))
#                 v_cache = mx.dequantize(*self.kv[1], group_size=32).reshape((B, N, -1, D))
#                 keys = mx.concatenate([k_cache, keys], axis=2)
#                 values = mx.concatenate([v_cache, values], axis=2)
#             self.kv = (mx.quantize(keys.reshape((B*N,-1)), group_size=32), mx.quantize(values.reshape((B*N,-1)), group_size=32))
#             self.offset = new_offset
#             return keys, values
#         else:
#             self.kv[0,:,:,self.offset:new_offset,:] = keys
#             self.kv[1,:,:,self.offset:new_offset,:] = values
#             self.offset = new_offset
#             return self.kv[0,:,:,:new_offset,:], self.kv[1,:,:,:new_offset,:]

# class Mask4D:
#     def __init__(self, L_all, mask):
#         mask_4d = mx.triu(mx.full((L_all, L_all), -mx.inf), k=1)[None, None]
#         if mask is not None:
#             pad_len = L_all - mask.shape[-1]
#             mask = mx.pad(mask, ((0,0),(0,pad_len)), 1)
#             mask = mx.expand_dims(mask, (1,2))
#             mask = mask*mask.transpose(0,1,3,2)
#             mask = mx.where(mask==1, 0, -np.inf)
#             mask_4d += mask
#             mask_4d = mx.repeat(mask_4d, 32, axis=1)
#         self.mask_4d = mask_4d

#     def __call__(self, past_L, L):
#         return self.mask_4d[:,:,past_L:L+past_L,:L+past_L]

# class Phi3F(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
#         self.vision_embed_tokens = None
#         self.layers = [Phi3DecoderLayer(config) for _ in range(config.num_hidden_layers)]
#         self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
#         self.use_quantized_cache= getattr(config, "use_quantized_cache", False)
#         self.config = config
#         self.num_hidden_layers = config.num_hidden_layers

#     def __call__(self, input_ids, pixel_values, image_sizes, positions, cache, pids, mask, max_tokens):
#         x = self.embed_tokens(input_ids)
#         if pixel_values is not None and self.vision_embed_tokens:
#             x = self.vision_embed_tokens(x, pixel_values, image_sizes, positions)
#         if cache is None:
#             cache = [KVCache(self.config, x, max_tokens) for _ in range(self.num_hidden_layers)]
#             self.masker = Mask4D(x.shape[1]+max_tokens, mask)
#             self.roper = SuRoPE(self.config, x.shape[1]+max_tokens, pids)
#         past_L, new_L = cache[0].offset, x.shape[1]
#         mask = self.masker(past_L, new_L)
#         cos, sin = self.roper(past_L, new_L)
#         for i, l in enumerate(self.layers):
#             x = l(x, cache[i], cos, sin, mask)
#         return self.norm(x), cache

# class Phi3V(Phi3F):
#     def __init__(self, config):
#         super().__init__(config)
#         self.vision_embed_tokens = Phi3ImageEmbedding(config)

# class Phi3ForCausalLM(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
#         self.model = Phi3F(config)
#         self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

#     def __call__(self, input_ids, pixel_values=None, image_sizes=None, positions=None, cache=None, pids=None, mask=None, max_tokens=0):
#         x, cache = self.model(input_ids, pixel_values, image_sizes, positions, cache, pids, mask, max_tokens)
#         return self.lm_head(x), cache

#     @property
#     def layers(self):
#         return self.model.layers

# class Phi3VForCausalLM(Phi3ForCausalLM):
#     def __init__(self, config):
#         super().__init__(config)
#         self.model = Phi3V(config)

class Agent:
    """
    A flexible agent class for managing toolchains and executing prompts.

    The Agent class provides a framework for processing prompts through a series of tools
    (functions) defined in a toolchain. It manages the execution flow, handles input and output,
    and maintains a log of operations.

    Attributes:
    -----------
    _default_toolchain : str
        A string defining the default toolchain, which includes adding code to prompts,
        generating responses, and executing code.

    Methods:
    --------
    __init__(self, toolchain=None, enable_api=True, **kwargs)
        Initialize the Agent with a toolchain and other optional parameters.

    __call__(self, prompt:str, images=None)
        Process a given prompt (and optionally images) through the toolchain.

    reset()
        Reset the agent's log and ongoing operations.

    log_step()
        Log the current step of operations.

    end()
        End the current session, log the final step, and reset the agent.

    set_toolchain(s)
        Set a new toolchain for the agent to use.

    Usage:
    ------
    The Agent can be used to process prompts through a defined series of operations:
    1. Initialize an Agent with a custom toolchain or use the default.
    2. Call the Agent with a prompt (and optionally images) to process.
    3. The Agent will execute each tool in the toolchain, passing results between steps.
    4. Results are logged at each step and can be accessed or saved.

    The toolchain is a string defining a series of operations, where each line is of the form:
    'output1, output2, ... = function_name(input1, input2, ...)'

    Example:
    --------
    >>> agent = Agent()
    >>> result = agent("Tell me about this image", images=["path/to/image.jpg"])
    >>> print(result['responses'])

    Notes:
    ------
    - The Agent supports API input handling, which can be enabled/disabled during initialization.
    - The toolchain can be customized to include different functions and processing steps.
    - The Agent maintains a log of all operations, which can be useful for debugging or analysis.
    - The 'enable_api' parameter affects how the Agent handles quotation marks in prompts.

    """
    _default_toolchain = """
        prompt = add_code(prompt, codes)
        responses = generate(prompt, images)
        files, codes = execute(responses, step)
        """
    def __init__(self, toolchain=None, enable_api=True, **kwargs):
        self.kwargs = kwargs if 'preload' in kwargs else kwargs|{'preload':load(**kwargs)}
        self.enable_api = enable_api
        self.set_toolchain(toolchain)
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
        s = self._default_toolchain if s is None else s
        self.toolchain = [_parse_toolchain(i) for i in s.split('\n') if '=' in i]
        self.list_outs = _parse_return(s)

def _linear_to_lora_layers(model, lora_layers, config):
    if isinstance(lora_layers, int):
        lora_layers = model.layers[-lora_layers:]
    elif isinstance(lora_layers, list):
        lora_layers = [model.layers[i] for i in lora_layers]
    else:
        raise ValueError("Invalid type for lora_layers. Expected int (number of layers) or list (layer indices or names).")
    def to_lora(layer):
        return LoRALinear.from_linear(layer, r=config["rank"], alpha=config["alpha"], scale=config["scale"], dropout=config["dropout"])
    for l in lora_layers:
        lora_layers = [(k, to_lora(m)) for k, m in l.named_modules() if k == "self_attn.qkv_proj"]
        l.update_modules(tree_unflatten(lora_layers))

def _setup():
    paths = [
        ("microsoft/Phi-3-mini-128k-instruct", PATH_ORIGINAL_PHI3_BLIND, PATH_QUANTIZED_PHI3_BLIND),
        ("microsoft/Phi-3-vision-128k-instruct", PATH_ORIGINAL_PHI3_VISION, PATH_QUANTIZED_PHI3_VISION)
    ]
    for hub, local, quant in paths:
        snapshot_download(repo_id=hub, allow_patterns=["*.safetensors", "*.json"], local_dir=local)
        _quantize(from_path=local, to_path=quant)
        train_lora(model_path=local, take=1)

def _load(model_path=PATH_ORIGINAL_PHI3_VISION, adapter_path=None, return_mx=True, **kwargs):
    model_cfg = _get_cfg(f"{model_path}/config.json", **kwargs)
    model_arch = model_cfg.architectures[0]
    processor = eval(model_arch[:5]+'Processor')
    processor = processor(model_path, return_mx=return_mx)
    model = eval(model_arch)
    model = model(model_cfg)
    nn.quantize(model, model_cfg.quantized['group_size'], model_cfg.quantized['bits']) if getattr(model_cfg, 'quantized', False) else None
    model.load_weights(_get_wt(model_path, model_cfg))
    if adapter_path:
        lora_cfg = _get_cfg(f"{adapter_path}/adapter_config.json")
        if lora_cfg.model_path != model_path:
            print(f'WARNING: LoRA trained for {lora_cfg.model_path} is being used with {model_path}')
        _linear_to_lora_layers(model, lora_cfg.lora_layers, lora_cfg.lora_parameters)
        model.load_weights(f'{adapter_path}/adapters.safetensors', strict=False)
    mx.eval(model.parameters())
    model.eval()
    return model, processor

def _quantize(from_path=PATH_ORIGINAL_PHI3_VISION, to_path=PATH_QUANTIZED_PHI3_VISION, q_group_size=64, q_bits=4):
    model_cfg = _get_cfg(f"{from_path}/config.json")
    model = eval(model_cfg.architectures[0])
    model = model(model_cfg)
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
            raise ValueError(f"Failed to load image from URL: {image_source} with error {e}")
    elif Path(image_source).is_file():
        try:
            return Image.open(image_source)
        except IOError as e:
            raise ValueError(f"Failed to load image {image_source} with error: {e}")
    else:
        raise ValueError(f"The image {image_source} must be a valid URL or existing file.")

def _get_api_output_path(process, file_prefix):
    if '<|api_output|>' in process.stdout:
        _api_output = process.stdout.strip().split('<|api_output|>', 1)[1]
        _from_path = Path(_api_output)
        if _from_path.is_file():
            _to_path = f'{file_prefix}_{_from_path.name}'
            _from_path.rename(_to_path)
            return _to_path
        else:
            return _api_output
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
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {json_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON in configuration file: {json_path}")
    except Exception as e:
        raise RuntimeError(f"Error loading configuration from {json_path}: {str(e)}")

def _get_wt(model_path, model_cfg):
    if getattr(model_cfg, 'sanitized', False):
        return [(k, v) for wf in glob.glob(f"{model_path}/*.safetensors") for k, v in mx.load(wf).items()]
    return [(k, v.transpose(0, 2, 3, 1) if "patch_embedding.weight" in k else v) for wf in glob.glob(f"{model_path}/*.safetensors") for k, v in mx.load(wf).items()]

def _generate(model, processor, prompt, images=None, max_tokens=1000, verbose=True, return_tps=False, early_stop=False, stream=True, mute=False):
    if images is not None and isinstance(prompt, list):
        raise ValueError('Images cannot be provided when prompt is a list')
    logit_stopper = LogitStopper(max_tokens, early_stop)
    token_stopper = TokenStopper(processor)
    streamer = Streamer(processor, stream, mute)
    dict_input = processor(prompt, images)
    tic = Tic()
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
    code_string = '\n'.join(re.findall(r"```python\n(.*?)```", code_string, re.DOTALL)).strip()
    if len(code_string) < 1:
        return None, None, None, None
    code_string = re.sub(r'plt\.savefig\(.*?\)', 'plt.show()', code_string)
    plot_path = f'{file_prefix}.png' if 'plt.show()' in code_string else None
    code_to_run = code_string.replace("plt.show()", f"plt.savefig('{plot_path}')")
    process = subprocess.run(["python", "-c", code_to_run], capture_output=True, text=True)
    output_path = None
    stdout = process.stdout.strip()
    stderr = process.stderr.strip()
    if len(stderr) < 1:
        output_path = plot_path if plot_path else _get_api_output_path(process, file_prefix)
        stderr = None
    return code_string, output_path, stdout, stderr

def _format_benchmark(json_path='benchmark.json'):
    with open(json_path, "r") as f:
        data = json.load(f)
    tasks = ["Text Generation", "Image Captioning", "Batched Generation"]
    task_indices = {0: "Text Generation", 1: "Image Captioning", 2: "Batched Generation"}
    markdown_table = """
    | Task                  | Vanilla Model | Quantized Model | Quantized Cache | LoRA Adapter |
    |-----------------------|---------------|-----------------|-----------------|--------------|"""
    def format_task_data(task_index):
        vanilla_tps = data["vanilla"][task_index][2]
        q_model_tps = data["q_model"][task_index][2]
        q_cache_tps = data["q_cache"][task_index][2]
        lora_tps = data["lora"][task_index][2]
        return f"\n    | {task_indices[task_index]}{' '*(22-len(task_indices[task_index]))}|  {vanilla_tps:.2f} tps     |  {q_model_tps:.2f} tps      |  {q_cache_tps:.2f} tps       |  {lora_tps:.2f} tps    |"
    for i in range(len(tasks)):
        markdown_table += format_task_data(i)
    print(markdown_table)

def _load_text(file_path):
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

def _score(model, processor, prompts):
    dict_input = processor(prompts)
    logits, _ = model(**dict_input, max_tokens=0)
    logits = nn.log_softmax(logits)
    input_ids = dict_input['input_ids']
    mask = dict_input['mask']
    batch_size, seq_length, vocab_size = logits.shape
    row_indices = mx.arange(batch_size)[:, None]
    col_indices = mx.arange(seq_length - 1)[None, :]
    token_indices = input_ids[:, 1:]
    next_token_logits = logits[row_indices, col_indices, token_indices]
    masked_logits = next_token_logits * mask[:, 1:]
    logit_sums = masked_logits.sum(axis=1)
    return logit_sums

def _choose(model, processor, prompts, appends=None, return_idx=False):
    if isinstance(appends, list):
        prompts = [prompt + str(a) for prompt in prompts for a in appends]
    scores = _score(model, processor, prompts)
    choices = prompts
    if appends is None:
        scores = [scores.argmax().item()]
    elif isinstance(appends, int):
        scores = scores.reshape((-1, appends)).argmax(axis=-1).tolist()
    elif isinstance(appends, list):
        scores = scores.reshape((-1, len(appends))).argmax(axis=-1).tolist()
        choices = appends
    else:
        raise ValueError('appends must be of type None, int, or list')
    if return_idx:
        return scores
    return [choices[i] for i in scores]

def _choose_from(model, processor, prompts, choices='ABCDE'):
    def _ord(s):
        return processor([f' {i}' for i in s])['input_ids'][:,-1]
    options = _ord(choices)
    dict_input = processor(prompts)
    logits, _ = model(**dict_input, max_tokens=0)
    logits = nn.log_softmax(logits[:,-1,:])
    indices = mx.argmax(logits[:, options], axis=-1).tolist()
    return [choices[i] for i in indices]

def mistral_api(prompt, history):
    """
    Example:
    --------
    agent = Agent(toolchain = "responses, history = mistral_api(prompt, history)")
    agent('Write a neurology ICU admission note')
    """
    history = '<s>' if history is None else history
    history += f"[INST] {prompt} [/INST]"
    client = InferenceClient("mistralai/Mistral-7B-Instruct-v0.3", token = os.environ.get('HF_READ_TOKEN', False))
    generate_kwargs = dict(
        temperature=0.9,
        max_new_tokens=1024,
        top_p=0.95,
        repetition_penalty=1.0,
        do_sample=True,
        seed=42,
        stream=False,
        details=False,
        # details=True,
        return_full_text=False,
    )
    result = client.text_generation(history, **generate_kwargs)
    result = result.strip()
    # result = result.generated_text.strip() # if details=True
    history += f" {result}</s> "
    print(f'### Prompt ###\n{prompt}\n### Output ###\n{result}')
    return {'responses':result, 'history':history}

def bark_api(prompt):
    """
    Example:
    --------
    agent = Agent(toolchain = "responses = bark_api(prompt)")
    agent('Write a neurology ICU admission note')
    """
    client = InferenceClient("suno/bark-small", token = os.environ.get('HF_READ_TOKEN', False))
    result = client.text_to_speech(prompt)
    Path("bark.flac").write_bytes(result)
    return prompt

def get_api(prompt, n_topk=1, verbose=True):
    """
    Example:
    --------
    agent = Agent(toolchain = "responses = get_api(prompt)")
    agent('Draw <|api_input|> A perfectly red apple, 32k HDR, studio lighting')
    """
    vdb = VDB()
    prompt = [prompt] if isinstance(prompt, str) else prompt
    codes = vdb([p.split('<|api_input|>')[0] for p in prompt])
    codes = [code.format(prompt=prompt[i].split('<|api_input|>')[1].strip()) for i, sublist in enumerate(codes) for code in sublist]
    if verbose:
        print('### Obtained api codes ###')
        for code in codes:
            print(code)
    return codes

def add_code(prompt, codes):
    """
    Append Python code blocks to a given prompt.

    Parameters:
    -----------
    prompt : str
        The original prompt text.
    codes : list of str or None
        A list of Python code strings to be appended to the prompt.

    Returns:
    --------
    str or list of str
        If codes is None, returns the original prompt.
        Otherwise, returns a list of strings, each containing the original prompt
        followed by a Python code block.
    """
    return prompt if codes is None else [f'{prompt}\n\n```python\n{code}\n```\n' for code in codes]

def chatui(agent=None):
    """
    Create and launch a chat user interface using Gradio.

    This function sets up an interactive chat interface that allows users to communicate with an AI agent.
    It supports text input and file uploads (specifically images) and displays the conversation history.

    This function is also the entry point for the 'phi3v' command-line tool, which can be run directly
    from the terminal after installing the phi-3-vision-mlx package.

    Parameters:
    -----------
    agent : Agent, optional
        An instance of the Agent class to handle the chat logic. If None, a new Agent instance is created.
        Default is None.

    Returns:
    --------
    None
        The function launches a Gradio interface and doesn't return a value.

    Behavior:
    ---------
    1. Initializes the chat agent if not provided.
    2. Defines helper functions for message handling and bot responses:
       - add_message: Adds user messages (text and files) to the chat history.
       - bot: Processes user input through the agent and formats the response.
       - reset: Resets the conversation and clears the chat history.
    3. Creates a Gradio Blocks interface with the following components:
       - Chatbot: Displays the conversation history.
       - MultimodalTextbox: Allows text input and file uploads.
       - Reset button: Clears the conversation.
    4. Sets up event handlers for user input submission and bot responses.
    5. Launches the Gradio interface in the browser.

    Notes:
    ------
    - The interface supports both text and image inputs.
    - Bot responses are processed to remove '<|end|>' tokens and empty lines.
    - The chat history keeps track of user inputs and bot responses, including file uploads.
    - The interface is set to occupy 80% of the viewport height.
    - The Gradio footer is hidden using custom CSS.
    - The interface is launched in-browser and inline.

    Dependencies:
    -------------
    - Requires the Gradio library for creating the user interface.
    - Assumes the existence of an Agent class that handles the chat logic.

    Command-line Usage:
    -------------------
    After installing the phi-3-vision-mlx package, you can run this function directly from the terminal using:

    $ phi3v

    This will launch the chat interface in your default web browser.

    Example:
    --------
    >>> chatui()
    # This will launch the chat interface in the default web browser.

    >>> custom_agent = Agent(custom_params)
    >>> chatui(agent=custom_agent)
    # Launches the chat interface with a custom agent configuration.
    """
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

def train_lora(model_path=PATH_QUANTIZED_PHI3_BLIND, adapter_path=None, lora_layers=1, lora_rank=1, epochs=1, batch_size=1, take=10, lr=1e-4, warmup=.5, mask_ratios=None, dataset_path="JosefAlbers/akemiH_MedQA_Reason"):
    """
    Train a LoRA (Low-Rank Adaptation) model using the specified parameters.

    This function loads a pre-trained model, applies LoRA adaptations, and fine-tunes it on a given dataset.
    It supports various training configurations, including masking strategies and learning rate scheduling.

    Parameters:
    -----------
    model_path : str, optional
        Path to the base model. Defaults to PATH_QUANTIZED_PHI3_BLIND.
    adapter_path : str or None, optional
        Path to save the LoRA adapter. If None, it's set to '{PATH_ADAPTERS}/{model_path}'.
        Defaults to None.
    lora_layers : int, optional
        Number of layers to apply LoRA. Defaults to 1.
    lora_rank : int, optional
        Rank of the LoRA adapter. Defaults to 1.
    epochs : int, optional
        Number of training epochs. Defaults to 1.
    batch_size : int, optional
        Batch size for training. Defaults to 1.
    take : int, optional
        Number of samples to take from the dataset. Defaults to 10.
    lr : float, optional
        Learning rate for the optimizer. Defaults to 1e-4.
    warmup : float, optional
        Fraction of total steps to use for learning rate warmup. Defaults to 0.5.
    mask_ratios : list of float or None, optional
        Ratios for input masking. If None, no masking is applied. Defaults to None.
    dataset_path : str, optional
        Path to the dataset used for training. Defaults to "JosefAlbers/akemiH_MedQA_Reason".

    Returns:
    --------
    None
        The function doesn't return a value but saves the trained LoRA adapter to the specified path.

    Notes:
    ------
    - The function uses several helper methods for data processing, loss calculation, and training.
    - It applies a learning rate schedule with warmup.
    - If mask_ratios are provided, it applies input masking during training.
    - The function uses AdamW optimizer for training.
    - After training, it cleans up by deleting the model and processor to free memory.

    Example:
    --------
    >>> train_lora(lora_layers=5, lora_rank=16, epochs=10,
    ...            take=10, batch_size=2, lr=1e-4, warmup=.5,
    ...            dataset_path="JosefAlbers/akemiH_MedQA_Reason")
    """
    def _prompt(example):
        questions = [i.rsplit(' A: ', 1)[0].strip() for i in example['input']]
        summaries = [i.strip().split('\n', 1)[0].strip() for i in example['summary']]
        prompts = [f"<|user|>\n{q}<|end|>\n<|assistant|>\n{s}<|end|>" for q,s in zip(questions, summaries)]
        example['prompts'] = prompts
        return example

    def _mask(batch):
        if mask_ratios is None:
            return batch, mx.ones(len(batch['input_ids']))
        new_batch = {key: [] for key in batch}
        num_sequences = len(batch['input_ids'])
        num_versions = len(mask_ratios) + 1
        loss_scales = []
        for key in batch:
            if key != 'mask':
                new_batch[key] = [seq for seq in batch[key] for _ in range(num_versions)]
        for i in range(num_sequences):
            input_tokens = batch['input_ids'][i]
            original_mask = batch['mask'][i]
            new_batch['mask'].append(original_mask)
            loss_scales.append(1.0)
            start = max((j for j, num in enumerate(input_tokens) if num < 0), default=0) + 3
            end = input_tokens.index(ID_ASS) - 3 if ID_ASS in input_tokens else len(input_tokens)
            maskable_range = range(start, end)
            maskable_indices = [j for j in maskable_range if original_mask[j] == 1]
            for ratio in mask_ratios:
                masked_attention_mask = original_mask.copy()
                num_to_mask = int(len(maskable_indices) * ratio)
                mask_indices = random.sample(maskable_indices, num_to_mask)
                for idx in mask_indices:
                    masked_attention_mask[idx] = 0
                new_batch['mask'].append(masked_attention_mask)
                loss_scales.append(10.**(-10.*ratio))
        return new_batch, mx.array(loss_scales)

    def _get_batch(i):
        batch = ds[i]
        batch = processor(batch['prompts'])
        batch, loss_scales = _mask(batch)
        splits = [i.index(ID_ASS) for i in batch['input_ids']]
        start_ce = min(splits)
        targets = mx.array(batch['input_ids'])[:,1:]
        loss_masks = mx.arange(targets.shape[1])[None,:] >= mx.array(splits)[:, None]
        inputs = {k:mx.array(v) for k,v in batch.items() if k in ['input_ids', 'pids', 'mask']}
        targets = targets[:, start_ce:]
        loss_masks = loss_masks[:, start_ce:]
        return inputs, targets, loss_masks, start_ce, loss_scales

    def _loss(model, batch):
        inputs, targets, loss_masks, start_ce, loss_scales = batch
        logit_outputs, _ = model(**inputs)
        logit_outputs = logit_outputs[:,:-1].astype(mx.float32)
        logit_outputs = logit_outputs[:,start_ce:]
        loss_ce = nn.losses.cross_entropy(logit_outputs, targets, reduction='none') * loss_masks
        loss_ce = loss_ce.sum(axis=1) / loss_masks.sum(axis = 1)
        loss_ce = (loss_ce * loss_scales).sum() # / targets.shape[0]
        return loss_ce

    def _set_lora(model_path, adapter_path, lora_layers, lora_rank):
        lora_cfg = {
            "model_path": str(model_path),
            "adapter_path": str(adapter_path),
            "lora_layers": lora_layers,
            "lora_parameters": {"rank": lora_rank, "alpha": lora_rank, "dropout": 0.0, "scale": 1.0},
        }
        return lora_cfg

    def _get_lr_schedule(lr, steps, warmup):
        n_warmup = int(steps*warmup)
        return mx.concatenate([mx.linspace(1e-6, lr, n_warmup), mx.linspace(lr, 1e-6, steps - n_warmup + 1)[1:]])

    if adapter_path is None:
        adapter_path = f'{PATH_ADAPTERS}/{model_path}'

    model, processor = _load(model_path, return_mx=False)
    ds = datasets.load_dataset(dataset_path, split='train').take(take)
    ds = ds.map(_prompt, batched=True).select_columns(['prompts'])
    batch_idx = []
    for _ in range(epochs):
        batch_idx +=  [x[i:i+batch_size] for x in [random.sample(range(len(ds)), len(ds))] for i in range(0, len(x) - batch_size + 1, batch_size)]
    lora_cfg = _set_lora(model_path, adapter_path, lora_layers, lora_rank)
    model.freeze()
    _linear_to_lora_layers(model, lora_cfg['lora_layers'], lora_cfg['lora_parameters'])
    model.train()
    distil_loss_value_and_grad = nn.value_and_grad(model, _loss)
    lr_schedule = _get_lr_schedule(lr, len(batch_idx), warmup)
    callback = TrainingCallback(lora_cfg, lr_schedule, batch_idx)
    optimizer=optim.AdamW(learning_rate=lr_schedule[0])
    state = [model.state, optimizer.state]
    for i, idx in enumerate(batch_idx):
        batch_i = _get_batch(idx)
        lvalue, grad = distil_loss_value_and_grad(model, batch_i)
        optimizer.learning_rate = lr_schedule[i]
        optimizer.update(model, grad)
        mx.eval(state, lvalue)
        callback(model, lvalue)
    callback.end_log()
    del model
    del processor

def test_lora(model_path=PATH_QUANTIZED_PHI3_BLIND, adapter_path=True, dataset_path="JosefAlbers/akemiH_MedQA_Reason", take=(0, 10), batch_size=10):
    """
    Test a LoRA (Low-Rank Adaptation) model on a given dataset.

    This function loads a model and its LoRA adapter, processes a dataset, and evaluates the model's
    performance on recall (summarization) and answer generation tasks.

    Parameters:
    -----------
    model_path : str, optional
        Path to the base model. Defaults to PATH_QUANTIZED_PHI3_BLIND.
    adapter_path : bool or str, optional
        Path to the LoRA adapter. If True, it's set to '{PATH_ADAPTERS}/{model_path}'.
        If None, the model without adapter is tested. Defaults to True.
    dataset_path : str, optional
        Path to the dataset to be used for testing. Defaults to "JosefAlbers/akemiH_MedQA_Reason".
    take : tuple of int, optional
        Range of samples to take from the dataset, in the format (start, end). Defaults to (0, 10).
    batch_size : int, optional
        Number of samples to process in each batch. Defaults to 10.

    Returns:
    --------
    None
        The function prints the evaluation results, including generation time, prediction time,
        and final score, but doesn't return any value.

    Notes:
    ------
    - The function uses an internal function _try for generating responses and evaluating them.
    - It performs two tasks: recall (summarization) and answer generation.
    - For recall, it generates a summary and compares it with the true summary.
    - For answer generation, it chooses an answer from options A-E and compares with the correct answer.
    - The function prints comparisons between generated and true responses for the recall task.
    - After completion, it deletes the model and processor to free up memory.

    Example:
    --------
    >>> test_lora(model_path="path/to/model", adapter_path="path/to/adapter",
    ...           dataset_path="dataset/path", take=(0, 10), batch_size=10)
    """
    def _try(example, q_col, q_until, q_format, fxn, a_col, c_col, verbose=True):
        questions = example[q_col]
        if q_until is not None:
            questions = [i.rsplit(q_until, 1)[0].strip() for i in questions]
        prompts_answer = [f"<|user|>\n{i}<|end|>\n<|assistant|>{q_format}" for i in questions]
        attempts = fxn(model, processor, prompts_answer)
        example[a_col] = [i.strip() for i in attempts]
        if c_col is not None and verbose is True:
            print('### Compare ###')
            for i,j in zip(example[a_col], example[c_col]):
                print('LoRA:', i)
                print('True: ', j.strip().split('\n', 1)[0])
                print('---')
        return example

    if adapter_path is True:
        adapter_path = f'{PATH_ADAPTERS}/{model_path}'
    model, processor = _load(model_path=model_path, adapter_path=adapter_path)
    ds = datasets.load_dataset(dataset_path, split='train')
    take = (0, take) if isinstance(take, int) else take
    ds = ds.select(range(*take), keep_in_memory=True)
    _recall_args = {
        'q_col':'input',
        'q_until':' A: ',
        'q_format':'',
        'fxn':partial(_generate, max_tokens=30, verbose=False, mute=True),
        'a_col':'recall',
        'c_col':'summary',
        'verbose':True
    }
    ds = ds.map(_try, batched=True, batch_size=batch_size, fn_kwargs=_recall_args)
    _answer_args = {
        'q_col':'input',
        'q_until':None,
        'q_format':'\nThe correct answer is',
        'fxn':partial(_choose_from, choices='ABCDE'),
        'a_col':'attempt',
        'c_col':'output',
        'verbose':False,
    }
    ds = ds.map(_try, batched=True, batch_size=batch_size, fn_kwargs=_answer_args)
    num_recall = len(ds.filter(lambda x: x["output"] == x["attempt"]))
    print(f'Score: {num_recall/len(ds)}({num_recall}/{len(ds)})')
    del model
    del processor

def benchmark(blind_model=False, json_path='benchmark.json'):
    """
    Perform a benchmark test on different model configurations and save the results.

    This function tests various configurations of a language model (vanilla, quantized model,
    quantized cache, and LoRA) on a set of predefined prompts. It measures the performance
    in terms of tokens per second (TPS) for both prompt processing and text generation.

    Parameters:
    -----------
    blind_model : bool, optional
        If True, uses a 'blind' version of the model (details depend on implementation).
        Defaults to False.

    Returns:
    --------
    None
        The function doesn't return a value but saves the benchmark results to a JSON file
        and prints a formatted version of the results.

    Behavior:
    ---------
    1. Defines a set of test prompts, including text-only and image-text prompts.
    2. Tests four configurations: vanilla, quantized model, quantized cache, and LoRA.
    3. For each configuration:
       - Loads the model with appropriate settings.
       - Processes each prompt and generates text.
       - Measures TPS for prompt processing and text generation.
    4. Saves all results to 'benchmark.json'.
    5. Calls a function to format and print the benchmark results.

    Notes:
    ------
    - The function uses predefined prompts, including a mix of text-only and image-text tasks.
    - It generates 100 tokens for each prompt.
    - The results are stored in a dictionary with keys 'vanilla', 'q_model', 'q_cache', 'lora'.
    - Each result entry contains the prompt index, prompt TPS, and generation TPS.
    - The function cleans up resources by deleting the model after each configuration test.
    - Requires 'generate', 'load', and '_format_benchmark' functions to be defined elsewhere.

    Example:
    --------
    >>> benchmark()
    # This will run the benchmark and save results to 'benchmark.json',
    # then print a formatted version of the results.

    >>> benchmark(blind_model=True)
    # Runs the benchmark using the 'blind' version of the model.
    """
    prompts = [
        ('Write a mystery horror.', ),
        ('What is shown in this image?', 'https://assets-c4akfrf5b4d3f4b7.z01.azurefd.net/assets/2024/04/BMDataViz_661fb89f3845e.png'),
        ([
            "Write an executive summary for a communications business plan",
            "Explain quantum computing.",
            "Write a poem about the first snowfall of the year.",
            "Write a Python function to implement a neural network from scratch, with detailed comments.",
            "Write a resume.",
            "Explain the key concepts of quantum computing and provide a Rust code example demonstrating quantum superposition.",
            "Explain the concept of dark matter and its significance in the universe.",
            "Summarize the major events of the French Revolution.",
            "Describe the water cycle.",
            "Write a Neurology ICU Admission Note.",
            "Describe a bustling alien marketplace on a distant planet with unique goods and creatures."
            "Imagine you have a magic potion that grants one wish. What would you wish for and how would it change your life?",
            "Compose a limerick about a clumsy robot.",
            "Write a JavaScript function to sort an array of objects by a specific property.",
            "Design a database schema for a social media platform, considering user profiles, posts, and interactions.",
            "Implement a basic encryption algorithm in Python.",
        ], None),
    ]
    results = {
        'vanilla': [],
        'q_model': [],
        'q_cache': [],
        'lora': [],
    }
    for method in results:
        kwargs = {'blind_model':blind_model}
        if method == 'q_model':
            kwargs['quantize_model'] = True
        elif method == 'q_cache':
            kwargs['quantize_cache'] = True
        elif method == 'lora':
            kwargs['use_adapter'] = True
        preload = load(**kwargs)
        for i, prompt in enumerate(prompts):
            prompt_tps, gen_tps = generate(*prompt, preload=preload, max_tokens=100, return_tps=True)
            results[method].append([i, prompt_tps, gen_tps])
        del preload
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=4)
    _format_benchmark(json_path)

def load(blind_model=False, quantize_model=False, quantize_cache=False, use_adapter=False, **kwargs):
    """
    Load a Phi-3 model with specified configuration.

    Parameters:
    -----------
    blind_model : bool, optional
        If True, load the language-only model. If False, load the vision model. Default is False.
    quantize_model : bool, optional
        If True, load the quantized version of the model. Default is False.
    quantize_cache : bool, optional
        If True, use quantized cache for the model. Default is False.
    use_adapter : bool, optional
        If True, load and use a LoRA adapter for the model. Default is False.
    **kwargs : dict
        Additional keyword arguments to pass to the model loading function.

    Returns:
    --------
    tuple
        A tuple containing the loaded model and processor.

    Notes:
    ------
    - If the model path doesn't exist, it will call _setup() to download or prepare the model.
    - The function uses predefined paths (PATH_*) to locate model files.
    """
    if blind_model:
        if quantize_model:
            model_path = PATH_QUANTIZED_PHI3_BLIND
        else:
            model_path = PATH_ORIGINAL_PHI3_BLIND
    else:
        if quantize_model:
            model_path = PATH_QUANTIZED_PHI3_VISION
        else:
            model_path = PATH_ORIGINAL_PHI3_VISION
    if use_adapter:
        adapter_path = f'{PATH_ADAPTERS}/{model_path}'
    else:
        adapter_path = None
    if not os.path.exists(model_path):
        _setup()
    return _load(model_path=model_path, use_quantized_cache=quantize_cache, adapter_path=adapter_path)

def generate(prompt, images=None, preload=None, blind_model=False, quantize_model=False, quantize_cache=False, use_adapter=False, max_tokens=1000, verbose=True, return_tps=False, early_stop=False, stream=True, apply_chat_template=True):
    """
    Generate text based on a given prompt, optionally with image input.

    Parameters:
    -----------
    prompt : str
        The input prompt for text generation.
    images : list of str or None, optional
        List of image paths or URLs to process along with the prompt.
    preload : tuple or None, optional
        A pre-loaded model and processor tuple. If None, a model will be loaded.
    blind_model : bool, optional
        If True, use the language-only model. Default is False.
    quantize_model : bool, optional
        If True, use the quantized version of the model. Default is False.
    quantize_cache : bool, optional
        If True, use quantized cache for the model. Default is False.
    use_adapter : bool, optional
        If True, use a LoRA adapter with the model. Default is False.
    max_tokens : int, optional
        Maximum number of tokens to generate. Default is 1000.
    verbose : bool, optional
        If True, print additional information during generation. Default is True.
    return_tps : bool, optional
        If True, return tokens per second information. Default is False.
    early_stop : bool or int, optional
        If True or an integer, stop generation early under certain conditions.
    stream : bool, optional
        If True, stream the generated text. Default is True.
    apply_chat_template : bool, optional
        If True, apply a chat template to the prompt. Default is True.

    Returns:
    --------
    str or tuple
        Generated text, or a tuple containing generated text and additional information
        if return_tps is True.

    Notes:
    ------
    - If '<|api_input|>' is in the prompt, it will call get_api() instead.
    - The function can handle both text-only and text-image inputs.
    """
    if '<|api_input|>' in prompt:
        return get_api(prompt)
    if preload is None:
        preload = load(blind_model=blind_model, quantize_model=quantize_model, quantize_cache=quantize_cache, use_adapter=use_adapter)
    return _generate(*preload, *_apply_chat_template(prompt, images, verbose, apply_chat_template), max_tokens=max_tokens, verbose=verbose, return_tps=return_tps, early_stop=early_stop, stream=stream)

def execute(code_strings, file_prefix=0, verbose=True):
    """
    Execute one or more Python code strings and capture the results.

    Parameters:
    -----------
    code_strings : str or list of str
        A single code string or a list of code strings to execute.
    file_prefix : int or str, optional
        A prefix to use for naming output files. Default is 0.
    verbose : bool, optional
        If True, print execution results. Default is True.

    Returns:
    --------
    dict
        A dictionary containing lists of execution results:
        - 'codes': The input code strings
        - 'files': Names of any files generated during execution
        - 'souts': Standard output from each execution
        - 'serrs': Standard error from each execution

    Notes:
    ------
    - Each code string is executed in a separate environment.
    - The function captures standard output, standard error, and any generated files.
    - If verbose is True, execution results are printed to the console.
    """
    code_strings = [code_strings] if isinstance(code_strings, str) else code_strings
    results = [_execute(code_string, f'{file_prefix}_{i}') for i, code_string in enumerate(code_strings)]
    if verbose is True:
        print('### Execution ###')
        for result in results:
            for r in result:
                print(r)
    return {k: [r[i] for r in results] for i, k in enumerate(['codes', 'files', 'souts', 'serrs'])}
