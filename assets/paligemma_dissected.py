from huggingface_hub import login
import os
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from transformers import AutoProcessor
import glob
from huggingface_hub import snapshot_download
import json
from PIL import Image
import requests
from types import SimpleNamespace

CFG_L = dict(rms_norm_eps = 1e-6, rope_base = 10000.0, attn_bias = False,)
CFG_V = dict(image_size = 224, num_channels = 3, layer_norm_eps = 1e-6, attn_bias = True)

class PGemmaModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vision_tower = VisionModel(config.vision_config)
        self.language_model = LanguageModel(config.text_config)
        self.multi_modal_projector = Projector(config)

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        dims = config.hidden_size
        bias = config.attn_bias
        self.n_heads = n_heads = config.num_attention_heads
        head_dim = dims // n_heads
        self.n_kv_heads = n_kv_heads = getattr(config, 'num_key_value_heads', n_heads)
        self.scale = head_dim**-0.5
        self.q_proj = nn.Linear(dims, n_heads * head_dim, bias=bias)
        self.k_proj = nn.Linear(dims, n_kv_heads * head_dim, bias=bias)
        self.v_proj = nn.Linear(dims, n_kv_heads * head_dim, bias=bias)
        self.o_proj = nn.Linear(n_heads * head_dim, dims, bias=bias)
        if getattr(config, 'rope_base', False):
            self.rope = nn.RoPE(head_dim, base = config.rope_base)
        else:
            self.rope = lambda x, *args, **kwargs: x

    def __call__(self, x, mask=None, cache = None):
        B, L, _ = x.shape
        queries = self.q_proj(x).reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = self.k_proj(x).reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = self.v_proj(x).reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        if cache is not None:
            key_cache, value_cache = cache
            queries = self.rope(queries, offset=key_cache.shape[2])
            keys = self.rope(keys, offset=key_cache.shape[2])
            keys = mx.concatenate([key_cache, keys], axis=2)
            values = mx.concatenate([value_cache, values], axis=2)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)
        output = mx.fast.scaled_dot_product_attention(queries, keys, values, scale=self.scale, mask=mask)
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output), (keys, values)

class Projector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear = nn.Linear(config.vision_config.hidden_size, config.vision_config.projection_dim, bias=True)

    def __call__(self, x):
        return self.linear(x)

class VisionEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_embedding = nn.Conv2d(in_channels=config.num_channels, out_channels=config.hidden_size, kernel_size=config.patch_size, stride=config.patch_size)
        
        self.num_patches = num_patches = (config.image_size // config.patch_size) ** 2 # -> 256
        self.position_embedding = nn.Embedding(num_patches, config.hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        return mx.flatten(self.patch_embedding(x), start_axis=1, end_axis=2) + self.position_embedding(mx.arange(self.num_patches)[None, :])

class GELU(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(nn.gelu_approx(self.fc1(x)))

class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = Attention(config)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = GELU(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def __call__(self, x: mx.array) -> mx.array:
        r, _ = self.self_attn(self.layer_norm1(x))
        h = x + r
        r = self.mlp(self.layer_norm2(h))
        return h + r

class VisionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = VisionEmbeddings(config)
        self.layers = [EncoderLayer(config) for _ in range(config.num_hidden_layers)]
        self.post_layernorm = nn.LayerNorm(config.hidden_size)

    def __call__(self, x):
        x = self.embeddings(x)
        for l in self.layers:
            x = l(x)
        x = self.post_layernorm(x[0])
        return x

class RMSNorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.weight = mx.ones((config.hidden_size,))
        self.eps = config.rms_norm_eps

    def __call__(self, x):
        return mx.fast.rms_norm(x, 1.0 + self.weight, self.eps)

class GeGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)

    def __call__(self, x) -> mx.array:
        return self.down_proj(nn.gelu(self.gate_proj(x)) * self.up_proj(x))

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = Attention(config)
        self.mlp = GeGLU(config)
        self.input_layernorm = RMSNorm(config)
        self.post_attention_layernorm = RMSNorm(config)

    def __call__(self, x, mask = None, cache = None):
        r, cache = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r
        return out, cache

class LanguageModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.scale = config.hidden_size**0.5
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [TransformerBlock(config=config) for _ in range(config.num_hidden_layers)]
        self.norm = RMSNorm(config)

    def __call__(self, input_ids, inputs_embeds=None, attention_mask_4d=None, cache=None):
        cache = [None] * len(self.layers) if cache is None else cache
        h = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        h = h * self.scale
        for e, layer in enumerate(self.layers):
            h, cache[e] = layer(h, attention_mask_4d, cache[e])
        attention_mask_4d = nn.MultiHeadAttention.create_additive_causal_mask(input_ids.shape[1]) if attention_mask_4d is None else attention_mask_4d
        return self.embed_tokens.as_linear(self.norm(h)), cache

def _get_cfg(json_path, **kwargs): # `copied from main.py
    try:
        with open(json_path, "r") as f:
            cfg = SimpleNamespace(**(json.load(f)|kwargs))
        return cfg
    except:
        return False

def _get_wt(model_path, model_cfg): # `modified from main.py
    if getattr(model_cfg, 'sanitized', False):
        return [(k, v) for wf in glob.glob(f"{model_path}/*.safetensors") for k, v in mx.load(wf).items()]
    else: 
        model_type = getattr(model_cfg, 'model_type', False)
        if model_type == 'paligemma': #
            return [(k.replace('vision_tower.vision_model.', 'vision_tower.').replace('language_model.model.', 'language_model.').replace('encoder.layers.', 'layers.').replace('self_attn.out_proj.','self_attn.o_proj.'), v.transpose(0, 2, 3, 1) if "patch_embedding.weight" in k else v) for wf in glob.glob(f"{model_path}/*.safetensors") for k, v in mx.load(wf).items()] # `paligemma
        elif mode_type == 'phi3_v':
            return [(k, v.transpose(0, 2, 3, 1) if "patch_embedding.weight" in k else v) for wf in glob.glob(f"{model_path}/*.safetensors") for k, v in mx.load(wf).items()] # `phi3v
        else:
            raise ValueError('Model type not supported')
    
def load_parts(model_id="google/paligemma-3b-mix-224", adapter_path=None, **kwargs): # `modified from main.py
    model_path = snapshot_download(repo_id=model_id, revision=None, allow_patterns=["*.safetensors", "*.json"], token = os.getenv('HF_TOKEN')) #
    model_cfg = _get_cfg(f"{model_path}/config.json", **kwargs)
    model_cfg.vision_config = SimpleNamespace(**(CFG_V|model_cfg.vision_config)) #
    model_cfg.text_config = SimpleNamespace(**(CFG_L|model_cfg.text_config)) #
    model = PGemmaModel(model_cfg) #
    nn.quantize(model, model_cfg.quantized['group_size'], model_cfg.quantized['bits']) if getattr(model_cfg, 'quantized', False) else None
    model.load_weights(_get_wt(model_path, model_cfg))
    if adapter_path:
        lora_cfg = _get_cfg(f"{adapter_path}/adapter_config.json")
        linear_to_lora_layers(model, lora_cfg.lora_layers, lora_cfg.lora_parameters)
        model.load_weights(f'{adapter_path}/adapters.safetensors', strict=False)
    mx.eval(model.parameters())
    model.eval()
    return AutoProcessor.from_pretrained(model_id), model.language_model, model.vision_tower, model.multi_modal_projector, model_cfg # 

def assemble(input_ids, inputs_embeds, image_features, attention_mask, config): # `new for paligemma
    inputs_embeds, image_features, attention_mask = [mx.array(i) for i in (inputs_embeds, image_features, attention_mask)]
    final_embedding = mx.zeros_like(inputs_embeds)
    text_mask = (input_ids != config.image_token_index) & (input_ids != config.pad_token_id)
    text_mask_expanded = mx.repeat(mx.expand_dims(text_mask, -1), final_embedding.shape[-1], axis=-1)
    final_embedding = mx.where(text_mask_expanded, inputs_embeds, final_embedding)
    image_mask = input_ids == config.image_token_index
    image_mask_expanded = mx.repeat(mx.expand_dims(image_mask, -1), final_embedding.shape[-1], axis=-1)
    final_embedding = mx.where(image_mask_expanded, mx.pad(image_features, ((0,0), (0,input_ids.shape[1] - image_features.shape[1]), (0,0))), final_embedding)
    attention_mask_expanded = mx.expand_dims(attention_mask, (1, 2)) 
    final_attention_mask_4d = attention_mask_expanded * attention_mask_expanded.transpose(0, 1, 3, 2)
    mx.repeat(final_attention_mask_4d, config.text_config.num_key_value_heads, axis=1)
    return mx.array(final_embedding), mx.array(final_attention_mask_4d)

# `load each component of a model separately
processor, language_model, vision_model, projector, config = load_parts()

# `process components
_processed = processor('Caption: ', Image.open(requests.get("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg", stream=True).raw), return_tensors="np")
input_ids, pixel_values, _attention_mask = [mx.array(_processed[key]) for key in ["input_ids", "pixel_values", "attention_mask"]]
_inputs_embeds = language_model.embed_tokens(input_ids)
_hidden_state = vision_model(pixel_values.transpose(0, 2, 3, 1))
_image_features = projector(_hidden_state[None])/(config.hidden_size**0.5)

# `assemble components
inputs_embeds, attention_mask_4d = assemble(input_ids, _inputs_embeds, _image_features, _attention_mask, config)

# `generate
logits, cache = language_model(input_ids, inputs_embeds, attention_mask_4d, None)
token = mx.argmax(logits[:, -1, :], axis=-1)
list_tokens = token.tolist()
for _ in range(100):
    logits, cache = language_model(token[None], None, None, cache)
    token = mx.argmax(logits[:, -1, :], axis=-1)
    list_tokens += token.tolist()
    if list_tokens[-1] == processor.tokenizer.eos_token_id:
        break

print(processor.tokenizer.decode(list_tokens))