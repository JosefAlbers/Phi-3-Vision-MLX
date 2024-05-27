from transformers import AutoProcessor 
from PIL import Image
import requests
import numpy as np
import math
import json
import glob
from types import SimpleNamespace
import mlx.nn as nn
import mlx.core as mx
from clip import VisionModel

CLIP_VIT_LARGE_PATCH14_336_CONFIG = SimpleNamespace(
  attention_dropout=0.0,
  dropout=0.0,
  hidden_act="quick_gelu",
  hidden_size=1024,
  image_size=336,
  initializer_factor=1.0,
  initializer_range=0.02,
  intermediate_size=4096,
  layer_norm_eps=1e-05,
  num_attention_heads=16,
  num_channels=3,
  num_hidden_layers=24,
  patch_size=14,
  projection_dim=768 
)

class Phi3ImageEmbedding(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        hidden_size = config.n_embd if hasattr(config, 'n_embd') else config.hidden_size
        self.wte = nn.Embedding(config.vocab_size, config.hidden_size)
        clip_config = CLIP_VIT_LARGE_PATCH14_336_CONFIG
        self.img_processor = VisionModel(clip_config)
        image_dim_out = config.img_processor['image_dim_out']
        self.num_img_tokens = config.img_processor['num_img_tokens']
        self.image_dim_out = image_dim_out
        self.img_sizes = None
        self.use_hd_transform = kwargs.get('use_hd_transform', False)
        self.with_learnable_separator = kwargs.get('with_learnable_separator', False)
        self.hd_transform_order = kwargs.get('hd_transform_order', 'glb_sub')
        self.glb_GN = mx.zeros([1, 1, self.image_dim_out * 4])
        self.sub_GN = mx.zeros([1, 1, 1, self.image_dim_out * 4])
        dim_projection = hidden_size
        depth = 2
        layers = [nn.Linear(image_dim_out * 4, dim_projection)]
        for _ in range(1, depth):
            layers.extend([nn.GELU(),
                            nn.Linear(dim_projection, dim_projection)])
        self.img_projection = layers
        self.vocab_size = config.vocab_size
        self.img_features = None
        self.layer_idx = config.img_processor.get('layer_idx', -2)
        self.type_feature = config.img_processor.get('type_feature', 'patch')

    def get_img_features(self, img_embeds):
        LAYER_IDX = self.layer_idx
        TYPE_FEATURE = self.type_feature
        img_processor_output = self.img_processor(mx.array(img_embeds).astype(mx.float32).transpose(0,2,3,1), output_hidden_states=True)
        img_feature = img_processor_output[-1][LAYER_IDX]
        patch_feature = img_feature[:, 1:]
        return np.array(mx.array(patch_feature).astype(mx.float32))

    def __call__(self, input_ids, img_embeds, img_sizes):
        input_ids = np.array(input_ids)
        img_embeds = np.array(img_embeds)
        img_sizes = np.array(img_sizes)
        MAX_INPUT_ID = int(1e9)
        positions = np.argwhere((input_ids < 0) & (input_ids > -MAX_INPUT_ID))
        select = False
        if len(positions.tolist()) > 0:
            g_values = abs(input_ids[positions[:, 0], positions[:, 1]])
            if self.use_hd_transform and img_sizes is not None and len(img_sizes):
                hd_transform = True
                bs = img_embeds.shape[0]
                img_features = self.get_img_features(img_embeds.reshape(-1, *img_embeds.shape[2:]))
                base_feat_height = base_feat_width = int(img_features.shape[1] ** 0.5)
                img_features = img_features.reshape(bs, -1, base_feat_height * base_feat_width, self.image_dim_out)
                C = self.image_dim_out
                H = base_feat_height
                output_imgs = []
                output_len = []
                for _bs in range(bs):
                    h, w = img_sizes[_bs]
                    h = h // 336 
                    w = w // 336
                    B_ = h * w
                    global_img_feature = img_features[_bs, :1]
                    glb_img = global_img_feature.reshape(1,H,H,C).reshape(1,H//2,2,H//2,2,C).transpose(0,1,3,2,4,5).reshape(1,H//2,H//2,4*C)
                    temp_glb_GN = np.tile(np.array(self.sub_GN.astype(mx.float32)), (1, H//2, 1, 1))
                    glb_img = np.concatenate([glb_img, temp_glb_GN], axis=2).reshape(1,-1,4*C)
                    sub_img = img_features[_bs, 1:]
                    sub_img = sub_img[:B_]
                    sub_img = sub_img.reshape(B_,H,H,C).reshape(B_,H//2,2,H//2,2,C).transpose(0,1,3,2,4,5).reshape(B_,-1,4*C)
                    sub_img = sub_img.reshape(1, h, w, 12, 12, -1).transpose(0,1,3,2,4,5).reshape(1,h*12,w*12,4*C)
                    temp_sub_GN = np.tile(np.array(self.sub_GN.astype(mx.float32)), (1, h*12, 1, 1))
                    sub_img = np.concatenate([sub_img, temp_sub_GN], axis=2).reshape(1,-1,4*C)
                    output_imgs.append(np.concatenate([sub_img, np.array(self.glb_GN.astype(mx.float32)), glb_img], axis=1))
                    temp_len = int((h*w+1)*144 + 1 + (h+1)*12)
                    output_len.append(temp_len)
                num_img_tokens = output_len
                img_set_tensor = []
                for _output_img in output_imgs:
                    x = mx.array(_output_img)
                    for l in self.img_projection:
                        x = l(x)
                    img_feature_proj = np.array(x.astype(mx.float32))
                    img_set_tensor.append(img_feature_proj)
            elif img_embeds.ndim == 4:
                selected_g_values = g_values[::self.num_img_tokens]
                tt = self.get_img_features(img_embeds).reshape(-1, self.image_dim_out)
                x = tt
                for l in self.img_projection:
                    x = l(x)
                img_set_tensor = x
            elif img_embeds.ndim == 3:
                selected_g_values = g_values[::self.num_img_tokens]
                tt = img_embeds.view(-1, self.image_dim_out)
                x = tt
                for l in self.img_projection:
                    x = l(x)
                img_set_tensor = x
            else:
                raise NotImplementedError
            select = True
        input_ids = np.clip(input_ids, 0, self.vocab_size)
        hidden_states = self.wte(mx.array(input_ids))
        hidden_states = np.array(hidden_states.astype(mx.float32))
        if select:
            if hd_transform:
                idx = 0
                for i, cnt in enumerate(num_img_tokens):
                    hidden_states[positions[idx, 0], positions[idx, 1] : positions[idx, 1] + cnt] = np.array(img_set_tensor[i])
                    idx += cnt
            else:
                idx = 0
                for i, g in enumerate(selected_g_values):
                    cnt = self.num_img_tokens
                    hidden_states[positions[idx, 0], positions[idx, 1] : positions[idx, 1] + cnt] = (
                        img_set_tensor[i * cnt : (i + 1) * cnt]
                        )
                    idx += cnt
        return mx.array(hidden_states)

class Phi3SuScaledRotaryEmbedding:
    def __init__(self, dim, config, device=None):
        self.dim = dim
        self.max_position_embeddings = config.max_position_embeddings
        self.base = config.rope_theta 
        self.short_factor = config.rope_scaling["short_factor"]
        self.long_factor = config.rope_scaling["long_factor"]
        self.original_max_position_embeddings = config.original_max_position_embeddings
        self.inv_freq = None

    def __call__(self, position_ids):
        seq_len = position_ids.max() + 1
        ext_factors = mx.array(self.long_factor, dtype=mx.float32) if seq_len > self.original_max_position_embeddings else mx.array(self.short_factor, dtype=mx.float32)
        inv_freq_shape = mx.arange(0, self.dim, 2, dtype=mx.float32) / self.dim
        self.inv_freq = 1.0 / (ext_factors * self.base**inv_freq_shape)
        inv_freq_expanded = mx.repeat(self.inv_freq[None, :, None], position_ids.shape[0], axis=0)
        position_ids_expanded = mx.array(position_ids)[:, None, :]
        freqs = mx.matmul(inv_freq_expanded, position_ids_expanded).transpose(0, 2, 1)  
        emb = mx.concatenate([freqs, freqs], axis=-1)  
        scale = self.max_position_embeddings / self.original_max_position_embeddings
        scaling_factor = math.sqrt(1 + math.log(scale) / math.log(self.original_max_position_embeddings))
        cos = mx.cos(emb) * scaling_factor
        sin = mx.sin(emb) * scaling_factor
        return cos, sin 

def rotate_half(x):
    midpoint = x.shape[-1] // 2  
    x1, x2 = x[..., :midpoint], x[..., midpoint:]  
    return mx.concatenate([-x2, x1], axis = -1) 

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = mx.expand_dims(cos, axis=unsqueeze_dim)  
    sin = mx.expand_dims(sin, axis=unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class Attention(nn.Module):
    def __init__(self, args):
        super().__init__()
        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads
        self.num_hidden_layers = args.num_hidden_layers
        self.head_dim = head_dim = args.hidden_size // n_heads
        self.scale = head_dim**-0.5
        op_size = n_heads * head_dim + 2 * (n_kv_heads * head_dim)
        self.qkv_proj = nn.Linear(dim, op_size, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)
        self.rotary_emb = Phi3SuScaledRotaryEmbedding(self.head_dim, args)
    def __call__(self, x, cache):
        B, L, D = x.shape
        qkv = self.qkv_proj(x)
        query_pos = self.n_heads * self.head_dim
        queries, keys, values = mx.split(qkv, [query_pos, query_pos + self.n_kv_heads * self.head_dim], axis=-1)
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        past_key_values_length = cache[0].shape[2] if cache is not None else 0        
        position_ids = mx.arange(past_key_values_length, past_key_values_length+L, dtype=mx.float32)[None]
        cos, sin = self.rotary_emb(position_ids)
        queries, keys = apply_rotary_pos_emb(queries, keys, cos, sin)
        if past_key_values_length > 0:
            key_cache, value_cache = cache
            keys = mx.concatenate([key_cache, keys], axis=2)
            values = mx.concatenate([value_cache, values], axis=2)
        scores = (queries * self.scale) @ keys.transpose(0, 1, 3, 2)
        scores += mx.triu(mx.full((L+past_key_values_length, L+past_key_values_length), -mx.inf), k=1)[None, None, past_key_values_length:, :]
        scores = mx.softmax(scores, axis=-1)
        output = (scores @ values)
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
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
        self.self_attn = Attention(config)
        self.mlp = Phi3MLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, x, cache):
        r, cache = self.self_attn(self.input_layernorm(x), cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r, cache

class Phi3V(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vision_embed_tokens = Phi3ImageEmbedding(config, **{'embedding_cls': config.embd_layer['embedding_cls'], **config.embd_layer})
        self.layers = [Phi3DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    def __call__(self, input_ids, pixel_values, image_sizes, cache):
        if pixel_values is None:
            x = self.vision_embed_tokens.wte(input_ids).astype(mx.float32)
        else:
            x = self.vision_embed_tokens(input_ids, pixel_values, image_sizes)
        cache = [None] * len(self.layers) if cache is None else cache
        for i, l in enumerate(self.layers):
            x, cache[i] = l(x, cache[i])
        return self.norm(x), cache

class CausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = Phi3V(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    def __call__(self, input_ids, pixel_values, image_sizes, cache):
        x, cache = self.model(input_ids, pixel_values, image_sizes, cache)
        return self.lm_head(x), cache

def load(model_id):
    from huggingface_hub import snapshot_download
    model_path = snapshot_download(repo_id=model_id)
    with open(f"{model_path}/config.json", "r") as f:
        config = json.load(f)
    model_config = SimpleNamespace(**config)
    weights = {}
    for wf in glob.glob(str(f"{model_path}/*.safetensors")):
        weights.update(mx.load(wf))
    weights = {k.replace('model.embed_tokens.weight','model.vision_embed_tokens.wte.weight'): v.transpose(0, 2, 3, 1) if "patch_embedding.weight" in k else v for k, v in weights.items()}
    model = CausalLM(model_config)
    model.load_weights(list(weights.items()))
    mx.eval(model.parameters())
    model.eval()
    return model

def phi3v(prompt, images=None):
    model_id = "microsoft/Phi-3-vision-128k-instruct" 
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True) 
    model = load(model_id)
    if images is None:
        input_ids, pixel_values, image_sizes = mx.array(processor.tokenizer.encode(prompt))[None], None, None
    else:
        inputs = processor(prompt, images, return_tensors="np")
        input_ids, pixel_values, image_sizes = [mx.array(inputs[i]) for i in ['input_ids', 'pixel_values', 'image_sizes']]
    logits, cache = model(input_ids, pixel_values, image_sizes, cache=None)
    token = mx.argmax(logits[:, -1, :], axis=-1)
    list_tokens = token.tolist()
    for i in range(50):
        logits, cache = model(input_ids=token[None], pixel_values=None, image_sizes=None, cache=cache)
        token = mx.argmax(logits[:, -1, :], axis=-1)
        list_tokens += token.tolist()
        if list_tokens[-1] == processor.tokenizer.eos_token_id:
            break
    print(processor.tokenizer.decode(list_tokens))

"""
Example 1:

phi3v("<|user|>Write a short funny sci-fi.<|end|>\n<|assistant|>\n")

```output
Title: The Great Galactic Gaffe

In the year 3021, the intergalactic space station, Orion-7, was the hub of all space travel and exploration. The station was a bustling metropol
```

Example 2:

prompt = f"<|user|>\n<|image_1|>\nWhat is shown in this image?<|end|>\n<|assistant|>\n"
images = [Image.open(requests.get("https://assets-c4akfrf5b4d3f4b7.z01.azurefd.net/assets/2024/04/BMDataViz_661fb89f3845e.png" , stream=True).raw)]
phi3v(prompt, images)

```output
The image displays a bar chart showing the percentage of respondents who agree with various statements about their preparedness for meetings. The chart has five vertical bars, each representing a different statement, with percentages ranging from 75% to 7
```
"""
