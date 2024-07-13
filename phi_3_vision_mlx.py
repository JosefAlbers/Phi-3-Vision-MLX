import glob
import inspect
import json
import logging
import math
import os
import random
import re
import subprocess
import time
from functools import partial
from io import BytesIO
from pathlib import Path
from shutil import copy
from types import SimpleNamespace
from urllib.parse import urlparse

import datasets
import gradio as gr
import matplotlib.pyplot as plt
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import requests
from huggingface_hub import snapshot_download
from mlx.utils import tree_flatten, tree_unflatten
from PIL import Image

from api import bark_api, mistral_api
from gte import VDB
from phi import (LoRALinear, Phi3ForCausalLM, Phi3FProcessor, Phi3VForCausalLM,
                 Phi3VProcessor, Tic, TrainingCallback)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("transformers").setLevel(logging.ERROR)

PATH_ADAPTERS = 'adapters'
PATH_ORIGINAL_PHI3_VISION  = 'models/phi3_v'
PATH_QUANTIZED_PHI3_VISION = 'models/phi3_v_Q'
PATH_ORIGINAL_PHI3_BLIND   = 'models/phi3_mini_128k'
PATH_QUANTIZED_PHI3_BLIND  = 'models/phi3_mini_128k_Q'
ID_EOS = 32007
ID_ASS = 32001

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
            list_txt = self.tokenizer.batch_decode([(i[:i.index(ID_EOS)+1] if ID_EOS in i else i) for i in arr_tokens.tolist()])
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
    def __init__(self, processor, batch_size):
        self.tokenizer = processor.tokenizer
        self.eos_id = ID_EOS
        self.batch_size = batch_size
        self.eos_rows = mx.ones(batch_size)
    def __call__(self, token):
        if self.eos_id in token:
            self.eos_rows *= token.squeeze()!=self.eos_id
            if self.eos_rows.sum() < 1:
                return True
        return False

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

def _linear_to_lora_layers(model, lora_targets, lora_layers, lora_config):
    if isinstance(lora_layers, int):
        lora_layers = model.layers[-lora_layers:]
    elif isinstance(lora_layers, list):
        lora_layers = [model.layers[i] for i in lora_layers]
    else:
        raise ValueError("Invalid type for lora_layers. Expected int (number of layers) or list (layer indices or names).")
    def to_lora(layer):
        return LoRALinear.from_linear(layer, r=lora_config["rank"], alpha=lora_config["alpha"], scale=lora_config["scale"], dropout=lora_config["dropout"])
    for l in lora_layers:
        lora_layers = [(k, to_lora(m)) for k, m in l.named_modules() if k in lora_targets]
        l.update_modules(tree_unflatten(lora_layers))

def _setup():
    paths = [
        ("microsoft/Phi-3-mini-128k-instruct", PATH_ORIGINAL_PHI3_BLIND, PATH_QUANTIZED_PHI3_BLIND),
        ("microsoft/Phi-3-vision-128k-instruct", PATH_ORIGINAL_PHI3_VISION, PATH_QUANTIZED_PHI3_VISION)
    ]
    for hub, local, quant in paths:
        raw = snapshot_download(repo_id=hub, allow_patterns=["*.safetensors", "*.json"])
        _sanitize(from_path=raw, to_path=local)
        _quantize(from_path=raw, to_path=quant)

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
        _linear_to_lora_layers(model, lora_cfg.lora_targets, lora_cfg.lora_layers, lora_cfg.lora_parameters)
        model.load_weights(f'{adapter_path}/adapters.safetensors', strict=False)
    mx.eval(model.parameters())
    model.eval()
    return model, processor

def _sanitize(from_path, to_path):
    model_cfg = _get_cfg(f"{from_path}/config.json")
    model = eval(model_cfg.architectures[0])
    model = model(model_cfg)
    model.load_weights(_get_wt(from_path, model_cfg))
    sanitized_weights = dict(tree_flatten(model.parameters()))
    del model
    os.makedirs(to_path, exist_ok=True)
    for f in glob.glob(f"{from_path}/*.json"):
        copy(f, to_path)
    with open(f"{to_path}/config.json", "w") as f:
        json.dump(vars(model_cfg)|{'sanitized':True}, f, indent=4)
    mx.save_safetensors(f'{to_path}/sanitized_model.safetensors', sanitized_weights)


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

def _load_image(image_source): # https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/utils.py
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
        print(f'*** Prompt ***\n{prompt}\n*** Images ***\n{images}\n*** Output ***') if verbose else None
        return prompt, images
    if images is not None:
        images = [_load_image(i) for i in images] if isinstance(images, list) else [_load_image(images)]
        img_prompt = '\n'.join([f'<|image_{i+1}|>' for i in range(len(images))]) + '\n'
    else:
        img_prompt = ''
    prompt = [prompt] if isinstance(prompt, str) else prompt
    prompt = [f"<|user|>\n{img_prompt}{i.strip()}<|end|>\n<|assistant|>\n" for i in prompt]
    if verbose:
        prompt_str = "\n".join(map(str.strip, prompt)).strip()
        images_str = "\n".join(map(str, images)) if images else "None"
        print(f'*** Prompt ***\n{prompt_str}\n*** Images ***\n{images_str}\n*** Output ***')
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
    streamer = Streamer(processor, stream, mute)
    dict_input = processor(prompt, images)
    token_stopper = TokenStopper(processor, dict_input['input_ids'].shape[0])
    tic = Tic()
    logits, cache = model(**dict_input, max_tokens=max_tokens)
    token = mx.argmax(logits[:, -1, :], axis=-1)[:,None]
    mx.eval(token, logits, cache)
    streamer(token)
    mask, pids = dict_input.get('mask', None), dict_input.get('pids', None)
    prompt_time = tic()
    for i in range(max_tokens-1):
        logits, cache = model(input_ids=token, cache=cache, mask=mask, pids=pids)
        token = mx.argmax(logits[:, -1, :], axis=-1)[:,None]
        mx.eval(token, logits, cache)
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

def _get_adapter_path(model_path):
    print(f'{PATH_ADAPTERS}/{Path(model_path).name}')
    return f'{PATH_ADAPTERS}/{Path(model_path).name}'

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

def _choose_from(model, processor, prompt, choices='ABCDE'):
    def _ord(s):
        return processor([f' {i}' for i in s])['input_ids'][:,-1]
    options = _ord(choices)
    dict_input = processor(prompt)
    logits, _ = model(**dict_input, max_tokens=0)
    logits = nn.log_softmax(logits[:,-1,:])
    indices = mx.argmax(logits[:, options], axis=-1).tolist()
    return [choices[i] for i in indices]

def _preprocess(s):
    for i in ['<|system|>', '<|user|>', '<|end|>']:
        s = s.replace(f'{i} ', f'{i}\n').replace(f'{i}\n\n', f'{i}\n')
    s = s.replace('<|end|><|assistant|>', '<|end|>\n<|assistant|>')
    return s

def _already(array_2d, array_1d):
    if array_2d.shape[1] < array_1d.shape[0]:
        return mx.ones(array_2d.shape[0])
    return ~mx.all(array_2d[:, -len(array_1d):] == array_1d, axis=1)

def _constrain(model, processor, prompt, constraints, return_full_text=False, mute=False):
    _was_prompt_str = False
    if isinstance(prompt, str):
        _was_prompt_str = True
        prompt = [prompt]
    prompt = [_preprocess(s) for s in prompt]
    len_ps = [len(p) for p in prompt]
    for constraint in constraints:
        constraint_text = constraint[1]
        id_constraint = mx.array(processor.tokenizer.encode(constraint_text, add_special_tokens=False)[1:])
        dict_input = processor(prompt)
        logits, cache = model(**dict_input, max_tokens=constraint[0] + id_constraint.shape[0]+10)
        logits = nn.log_softmax(logits, axis=-1)
        token = mx.argmax(logits[:, -1, :], axis=-1)[:,None]
        running_score = mx.max(logits[:, -1, :], axis=-1)[:,None]
        _score_0 = logits[:, -1, id_constraint[0]]
        tiled_id_constraint = mx.tile(id_constraint, (token.shape[0], 1))
        logits_rest, cache = model(input_ids=tiled_id_constraint, cache=cache, mask=dict_input.get('mask', None), pids=dict_input.get('pids', None), advance_offset=0)
        logits_rest = nn.log_softmax(logits_rest, axis=-1)
        _score_1 = logits_rest[mx.arange(tiled_id_constraint.shape[0])[:,None], mx.arange(tiled_id_constraint.shape[1]-1)[None,:], tiled_id_constraint[:,1:]]
        score_sofar = mx.concatenate([_score_0[:,None], _score_1], axis = 1).mean(axis=1)
        synth_sofar = tiled_id_constraint
        synth_pad = mx.tile(mx.array([ID_EOS]), (tiled_id_constraint.shape[0], 1))
        synth = tiled_id_constraint
        tokens = []
        finished_rows = mx.ones(tiled_id_constraint.shape[0])
        for i in range(constraint[0]):
            tokens.append(token)
            synth = mx.concatenate(tokens+[tiled_id_constraint], axis=1)
            token_plus = mx.concatenate([token, tiled_id_constraint], axis=1)
            logits, cache = model(input_ids=token_plus, cache=cache, mask=dict_input.get('mask', None), pids=dict_input.get('pids', None), advance_offset=1)
            logits = nn.log_softmax(logits)
            token = mx.argmax(logits[:, 0, :], axis=-1)
            finished_rows *= token != ID_EOS
            token = token[:,None]
            _synth_score = logits[mx.arange(tiled_id_constraint.shape[0])[:,None], mx.arange(tiled_id_constraint.shape[1])[None,:], tiled_id_constraint]
            score = mx.concatenate([running_score, _synth_score], axis=1).mean(axis=1)
            running_score = mx.concatenate([running_score, mx.max(logits[:, 0, :], axis=-1)[:, None]], axis=1)
            synth_sofar = mx.concatenate([synth_sofar, synth_pad], axis=1)
            finished_rows *= _already(mx.concatenate(tokens, axis=1), id_constraint)
            if finished_rows.sum() < 1:
                break
            rows_to_update = score > score_sofar
            rows_to_update *= finished_rows
            synth_sofar = mx.where(rows_to_update[:,None], synth, synth_sofar)
            score_sofar = mx.where(rows_to_update, score, score_sofar)
        output = mx.concatenate([dict_input['input_ids'], synth_sofar], axis=1).tolist()
        S = dict_input['input_ids'].shape[1]
        output = [(i[:i.index(ID_EOS,S)] if ID_EOS in i[S:] else i) for i in output]
        output = [[num for num in sublist if num not in (0,1)] for sublist in output]
        output = processor.tokenizer.batch_decode(output)
        output = [_preprocess(s) for s in output]
        prompt = output
    if not return_full_text:
        output = [o[l:] for o,l in zip(output,len_ps)]
    if not mute:
        if len(output) == 1:
            print(output[0])
        else:
            for i,o in enumerate(output):
                print(f'\n< Generated text for prompt #{i} >\n{o}')
    if _was_prompt_str:
        output = output[0]
    return output

def _beam(model, processor, prompt, constraints, return_full_text=False, mute=False):
    def _get_beam(logits, cache, id_constraint, beam_idx=0, n_beam=3):
        _B, _S, _V = logits.shape
        _arg_beam = mx.argpartition(-logits[:, beam_idx, :], kth=n_beam, axis=-1)[:,:n_beam]
        _beam = _arg_beam.reshape(-1)[:,None]
        _beam = mx.concatenate([_beam, mx.tile(id_constraint, (_beam.shape[0], 1))], axis=-1)
        _beam_logits, cache = model(input_ids=_beam, cache=cache, n_beam=n_beam)
        _beam_logits = nn.log_softmax(_beam_logits)
        _beam_score = _beam_logits[mx.arange(_beam_logits.shape[0])[:,None], mx.arange(_beam.shape[1]-1)[None,:], _beam[:,1:]]
        _beam_score = mx.concatenate([logits[mx.arange(_arg_beam.shape[0])[:,None],beam_idx,_arg_beam].reshape(-1)[:,None], _beam_score], axis=1).mean(axis=1)
        _beam_score = _beam_score.reshape(-1,n_beam)
        _argmax_beam = mx.argmax(_beam_score, axis=-1)
        token = _arg_beam[mx.arange(len(_argmax_beam)), _argmax_beam]
        return token, cache
    _was_prompt_str = False
    if isinstance(prompt, str):
        _was_prompt_str = True
        prompt = [prompt]
    prompt = [_preprocess(s) for s in prompt]
    len_ps = [len(p) for p in prompt]
    for constraint in constraints:
        constraint_text = constraint[1]
        id_constraint = mx.array(processor.tokenizer.encode(constraint_text, add_special_tokens=False)[1:])
        dict_input = processor(prompt)
        logits, cache = model(**dict_input, max_tokens=constraint[0] + id_constraint.shape[0]+10)
        logits = nn.log_softmax(logits, axis=-1)
        token = mx.argmax(logits[:, -1, :], axis=-1)[:,None]
        running_score = mx.max(logits[:, -1, :], axis=-1)[:,None]
        _score_0 = logits[:, -1, id_constraint[0]]
        tiled_id_constraint = mx.tile(id_constraint, (token.shape[0], 1))
        logits_rest, cache = model(input_ids=tiled_id_constraint, cache=cache, advance_offset=0)
        logits_rest = nn.log_softmax(logits_rest, axis=-1)
        _score_1 = logits_rest[mx.arange(tiled_id_constraint.shape[0])[:,None], mx.arange(tiled_id_constraint.shape[1]-1)[None,:], tiled_id_constraint[:,1:]]
        score_sofar = mx.concatenate([_score_0[:,None], _score_1], axis = 1).mean(axis=1)
        synth_sofar = tiled_id_constraint
        synth_pad = mx.tile(mx.array([ID_EOS]), (tiled_id_constraint.shape[0], 1))
        synth = tiled_id_constraint
        tokens = []
        finished_rows = mx.ones(tiled_id_constraint.shape[0])
        for i in range(constraint[0]):
            tokens.append(token)
            synth = mx.concatenate(tokens+[tiled_id_constraint], axis=1)
            token_plus = mx.concatenate([token, tiled_id_constraint], axis=1)
            logits, cache = model(input_ids=token_plus, cache=cache, advance_offset=1)
            logits = nn.log_softmax(logits)
            token, cache = _get_beam(logits, cache, id_constraint)
            finished_rows *= token != ID_EOS
            _synth_score = logits[mx.arange(tiled_id_constraint.shape[0])[:,None], mx.arange(tiled_id_constraint.shape[1])[None,:], tiled_id_constraint]
            score = mx.concatenate([running_score, _synth_score], axis=1).mean(axis=1)
            running_score = mx.concatenate([running_score, logits[mx.arange(token.shape[0]),0,token][:,None]], axis=1)
            token = token[:,None]
            synth_sofar = mx.concatenate([synth_sofar, synth_pad], axis=1)
            finished_rows *= _already(mx.concatenate(tokens, axis=1), id_constraint)
            if finished_rows.sum() < 1:
                break
            rows_to_update = score > score_sofar
            rows_to_update *= finished_rows
            synth_sofar = mx.where(rows_to_update[:,None], synth, synth_sofar)
            score_sofar = mx.where(rows_to_update, score, score_sofar)
        output = mx.concatenate([dict_input['input_ids'], synth_sofar], axis=1).tolist()
        S = dict_input['input_ids'].shape[1]
        output = [(i[:i.index(ID_EOS,S)] if ID_EOS in i[S:] else i) for i in output]
        output = [[num for num in sublist if num not in (0,1)] for sublist in output]
        output = processor.tokenizer.batch_decode(output)
        output = [_preprocess(s) for s in output]
        prompt = output
    if not return_full_text:
        output = [o[l:] for o,l in zip(output,len_ps)]
    if not mute:
        if len(output) == 1:
            print(output[0])
        else:
            for i,o in enumerate(output):
                print(f'\n< Generated text for prompt #{i} >\n{o}')
    if _was_prompt_str:
        output = output[0]
    return output

def get_api(prompt, n_topk=1, verbose=True):
    """
    Retrieve and format API code based on input prompts using vector similarity search.

    This function uses a Vector Database (VDB) to find the most relevant API code
    for given prompts. It's designed to work with prompts that may contain the
    '<|api_input|>' delimiter to separate the API request from additional input.

    Parameters:
    -----------
    prompt : str or list of str
        The input prompt(s) to search for relevant API code. If a prompt contains
        '<|api_input|>', the part before it is used for the search, and the part
        after it is used to format the retrieved code.
    n_topk : int, optional
        The number of top matching API codes to retrieve for each prompt. Default is 1.
    verbose : bool, optional
        If True, print the obtained API codes. Default is True.

    Returns:
    --------
    list of str
        A list of formatted API code strings relevant to the input prompt(s).

    Notes:
    ------
    - The function uses a VDB (Vector Database) for similarity search.
    - If multiple prompts are provided, it returns a list of API codes for each prompt.
    - The retrieved API code is formatted with the part of the prompt after '<|api_input|>'.
    - This function is typically used within an Agent's toolchain for API-related tasks.

    Example:
    --------
    >>> agent = Agent(toolchain="responses = get_api(prompt)")
    >>> agent('Draw <|api_input|> A perfectly red apple, 32k HDR, studio lighting')
    # This will retrieve and format API code for image generation based on the given prompt.

    In this example, 'Draw' is used for the API search, and 'A perfectly red apple, 32k HDR,
    studio lighting' is used to format the retrieved API code.
    """
    vdb = VDB()
    prompt = [prompt] if isinstance(prompt, str) else prompt
    codes = vdb([p.split('<|api_input|>')[0] for p in prompt])
    codes = [code.format(prompt=prompt[i].split('<|api_input|>')[1].strip()) for i, sublist in enumerate(codes) for code in sublist]
    if verbose:
        print('*** Obtained API Codes ***')
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

def chat_ui(agent=None):
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
    >>> chat_ui()
    # This will launch the chat interface in the default web browser.

    >>> custom_agent = Agent(custom_params)
    >>> chat_ui(agent=custom_agent)
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

def train_lora(model_path=PATH_QUANTIZED_PHI3_BLIND, adapter_path=None, lora_targets=["self_attn.qkv_proj", "self_attn.o_proj"], lora_layers=1, lora_rank=1, epochs=1, batch_size=1, take=10, lr=1e-4, warmup=.5, mask_ratios=None, dataset_path="JosefAlbers/akemiH_MedQA_Reason"):
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

    def _get_batch(indices):
        batch = [list_prompts[i] for i in indices]
        batch = processor(batch)
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

    def _set_lora(model_path, adapter_path, lora_targets, lora_layers, lora_rank):
        lora_cfg = {
            "model_path": str(model_path),
            "adapter_path": str(adapter_path),
            "lora_layers": lora_layers,
            "lora_targets": lora_targets,
            "lora_parameters": {"rank": lora_rank, "alpha": lora_rank, "dropout": 0.0, "scale": 1.0},
        }
        return lora_cfg

    def _get_lr_schedule(lr, steps, warmup):
        n_warmup = int(steps*warmup)
        return mx.concatenate([mx.linspace(1e-6, lr, n_warmup), mx.linspace(lr, 1e-6, steps - n_warmup + 1)[1:]])

    if adapter_path is None:
        adapter_path = _get_adapter_path(model_path)
    model, processor = _load(model_path, return_mx=False)
    ds = datasets.load_dataset(dataset_path, split='train')
    if take > len(ds):
        raise ValueError(f"Requested {take} samples, but dataset only contains {len(ds)} samples.")
    ds = ds.take(take)
    list_prompts = ds.map(_prompt, batched=True)['prompts']
    batch_idx = []
    for _ in range(epochs):
        batch_idx +=  [x[i:i+batch_size] for x in [random.sample(range(len(ds)), len(ds))] for i in range(0, len(x) - batch_size + 1, batch_size)]
    lora_cfg = _set_lora(model_path, adapter_path, lora_targets, lora_layers, lora_rank)
    model.freeze()
    _linear_to_lora_layers(model, lora_cfg['lora_targets'], lora_cfg['lora_layers'], lora_cfg['lora_parameters'])
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
    Test a LoRA (Low-Rank Adaptation) model on a given dataset using various generation methods.

    This function loads a model and its LoRA adapter (if specified), processes a dataset, and evaluates
    the model's performance on recall (summarization) and answer generation tasks using different methods.

    Parameters:
    -----------
    model_path : str, optional
        Path to the base model. Default is PATH_QUANTIZED_PHI3_BLIND.
    adapter_path : bool or str, optional
        Path to the LoRA adapter. If True, it's set to '{PATH_ADAPTERS}/{model_path}'.
        If None, the model without adapter is tested. Default is True.
    dataset_path : str, optional
        Path to the dataset to be used for testing. Default is "JosefAlbers/akemiH_MedQA_Reason".
    take : tuple of int or int, optional
        Range of samples to take from the dataset. If tuple, in the format (start, end).
        If int, takes (0, take) samples. Default is (0, 10).
    batch_size : int, optional
        Number of samples to process in each batch. Default is 10.

    Returns:
    --------
    None
        The function prints the evaluation results but doesn't return any value.

    Notes:
    ------
    - Performs three tasks: recall of trained texts, answer generation using _choose_from(),
      and answer generation using constrained decoding.
    - For recall, it generates a summary and compares it with the true summary.
    - For answer generation, it uses three methods:
      1. _choose_from(): Chooses an answer from options A-E.
      2. _constrain(): Generates an answer with specific constraints.
      3. _beam(): Generates an answer using beam search with constraints.
    - Prints comparisons between generated and true responses for each task.
    - After completion, prints scores for all answer generation methods.
    - The model and processor are deleted after use to free up memory.

    Example:
    --------
    >>> test_lora(model_path="path/to/model", adapter_path="path/to/adapter",
    ...           dataset_path="dataset/path", take=(0, 10), batch_size=10)
    """
    def _try(example, q_col, q_until, q_format, fxn, a_format, a_col, c_col, verbose=True):
        questions = example[q_col]
        if q_until is not None:
            questions = [i.rsplit(q_until, 1)[0].strip() for i in questions]
        q_format = q_format if q_format else ''
        prompts = [f"<|user|>\n{i}<|end|>\n<|assistant|>{q_format}" for i in questions]
        attempts = fxn(prompt=prompts)
        if a_format:
            answers = [i[i.find(a_format)+len(a_format)].strip() for i in attempts]
        else:
            answers = attempts
        example[a_col] = answers
        if c_col is not None and verbose is True:
            print(f'*** COMPARE ***')
            for i,j,k,l in zip(questions, attempts, answers, example[c_col]):
                print('- PROMPT:', i)
                print('- OUTPUT:', j)
                if a_format:
                    print('- ANSWER:', k)
                print('-  TRUTH:', l.strip().split('\n', 1)[0])
                print('---')
        return example

    def _map(ds, map_args):
        return ds.map(_try, batched=True, batch_size=batch_size, fn_kwargs=map_args, load_from_cache_file=False, new_fingerprint=map_args['a_col'])

    if adapter_path is True:
        adapter_path = _get_adapter_path(model_path)
    model, processor = _load(model_path=model_path, adapter_path=adapter_path)
    ds = datasets.load_dataset(dataset_path, split='train')
    take = (0, take) if isinstance(take, int) else take
    ds = ds.select(range(*take), keep_in_memory=True)
    list_args=[
        {
            'q_col':'input',
            'q_until':' A: ',
            'q_format':None,
            'fxn':partial(_generate, model=model, processor=processor, max_tokens=30, verbose=False, mute=True),
            'a_format':None,
            'a_col':'recall',
            'c_col':'summary',
            'verbose':True
        },
        {
            'q_col':'input',
            'q_until':None,
            'q_format':'\nThe correct answer is',
            'fxn':partial(_choose_from, model=model, processor=processor, choices='ABCDE'),
            'a_format':None,
            'a_col':'attempt',
            'c_col':'output',
            'verbose':True,
        },
        {
            'q_col':'input',
            'q_until':None,
            'q_format':'',
            'fxn':partial(_constrain, model=model, processor=processor, constraints=[(30, ' The correct answer is'), (10, 'X.')], mute=True),
            'a_format':'The correct answer is ',
            'a_col':'constrained_attempt',
            'c_col':'output',
            'verbose':True,
        },
        {
            'q_col':'input',
            'q_until':None,
            'q_format':'',
            'fxn':partial(_beam, model=model, processor=processor, constraints=[(30, ' The correct answer is'), (10, 'X.')], mute=True),
            'a_format':'The correct answer is ',
            'a_col':'beamed_attempt',
            'c_col':'output',
            'verbose':True,
        },
    ]
    for i in list_args:
        ds = _map(ds, i)
    num_chosen = len(ds.filter(lambda x: x["output"] == x["attempt"]))
    print(f'Score w/ _choose_from(): {num_chosen/len(ds)}({num_chosen}/{len(ds)})')
    num_constr = len(ds.filter(lambda x: x["output"] == x["constrained_attempt"]))
    print(f'Score w/ _constrain():   {num_constr/len(ds)}({num_constr}/{len(ds)})')
    num_beamed = len(ds.filter(lambda x: x["output"] == x["beamed_attempt"]))
    print(f'Score w/ _beam():        {num_beamed/len(ds)}({num_beamed}/{len(ds)})')
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
    # Runs the benchmark using the 'blind' version of the model (i.e., Phi-3-Mini-128K)
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
    for i in [
        PATH_ORIGINAL_PHI3_VISION,
        PATH_QUANTIZED_PHI3_VISION,
        PATH_ORIGINAL_PHI3_BLIND,
        PATH_QUANTIZED_PHI3_BLIND
    ]:
        train_lora(model_path=i, take=1)
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
        adapter_path = _get_adapter_path(model_path)
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

def constrain(prompt, constraints=[(30, ' The correct answer is'), (10, 'X.')], images=None, preload=None, blind_model=False, quantize_model=False, quantize_cache=False, use_adapter=False, apply_chat_template=True, use_beam=False):
    """
    Perform constrained decoding on the given prompt using specified constraints.

    This function generates text based on the input prompt while adhering to the specified constraints.
    It supports various model configurations and can handle both text and image inputs.

    Parameters:
    -----------
    prompt : str or list of str
        The input prompt(s) for text generation.
    constraints : list of tuple, optional
        List of constraints in the format [(max_tokens, constraint_text), ...].
        Each constraint specifies the maximum number of tokens to generate before the constraint_text
        must appear. Default is [(30, ' The correct answer is'), (10, 'X.')].
    images : list or None, optional
        List of image inputs for multimodal models. Default is None.
    preload : tuple or None, optional
        A tuple containing (model, processor) if already loaded. If None, the function will load
        the model and processor based on the provided configuration. Default is None.
    blind_model : bool, optional
        If True, uses a model without vision capabilities. Default is False.
    quantize_model : bool, optional
        If True, uses a quantized version of the model for reduced memory usage. Default is False.
    quantize_cache : bool, optional
        If True, uses cache quantization for improved memory efficiency. Default is False.
    use_adapter : bool, optional
        If True, uses a LoRA adapter with the model for fine-tuned behavior. Default is False.
    apply_chat_template : bool, optional
        If True, applies a chat template to the prompt before processing. Default is True.
    use_beam : bool, optional
        If True, uses beam search for generation instead of greedy decoding. Default is False.

    Returns:
    --------
    str or list of str
        The generated text(s) adhering to the specified constraints. Returns a single string if
        the input prompt was a string, otherwise returns a list of strings.

    Notes:
    ------
    - The function preprocesses the prompt and applies the constraints sequentially.
    - It uses either a custom constrained generation algorithm or beam search to ensure the output adheres to the constraints.
    - The output format matches the input format (str or list of str).
    - If apply_chat_template is True, the prompt is processed through a chat template before generation.

    Example:
    --------
    >>> prompt = "Write a Python function to calculate the Fibonacci sequence up to a given number n."
    >>> constraints = ...
    >>> result = constrain(prompt, constraints)
    >>> print(result)
    """
    # constrain ("Write a Python function to calculate the Fibonacci sequence up to a given number n.", [(100, "\n```python\n"), (100, " return "), (200, "\n```")])
    # prompts = [
    #     "A 20-year-old woman presents with menorrhagia for the past several years. She says that her menses “have always been heavy”, and she has experienced easy bruising for as long as she can remember. Family history is significant for her mother, who had similar problems with bruising easily. The patient's vital signs include: heart rate 98/min, respiratory rate 14/min, temperature 36.1°C (96.9°F), and blood pressure 110/87 mm Hg. Physical examination is unremarkable. Laboratory tests show the following: platelet count 200,000/mm3, PT 12 seconds, and PTT 43 seconds. Which of the following is the most likely cause of this patient’s symptoms? A: Factor V Leiden B: Hemophilia A C: Lupus anticoagulant D: Protein C deficiency E: Von Willebrand disease",
    #     "A 25-year-old primigravida presents to her physician for a routine prenatal visit. She is at 34 weeks gestation, as confirmed by an ultrasound examination. She has no complaints, but notes that the new shoes she bought 2 weeks ago do not fit anymore. The course of her pregnancy has been uneventful and she has been compliant with the recommended prenatal care. Her medical history is unremarkable. She has a 15-pound weight gain since the last visit 3 weeks ago. Her vital signs are as follows: blood pressure, 148/90 mm Hg; heart rate, 88/min; respiratory rate, 16/min; and temperature, 36.6℃ (97.9℉). The blood pressure on repeat assessment 4 hours later is 151/90 mm Hg. The fetal heart rate is 151/min. The physical examination is significant for 2+ pitting edema of the lower extremity. Which of the following tests o should confirm the probable condition of this patient? A: Bilirubin assessment B: Coagulation studies C: Hematocrit assessment D: Leukocyte count with differential E: 24-hour urine protein"]
    # constrain(prompts, constraints=[(30, ' The correct answer is'), (10, 'X.')], blind_model=True, quantize_model=True)
    if preload is None:
        model, processor = load(blind_model=blind_model, quantize_model=quantize_model, quantize_cache=quantize_cache, use_adapter=use_adapter)
        preload = model, processor
    if apply_chat_template:
        prompt = _apply_chat_template(prompt, None, False)[0]
    if use_beam:
        return _beam(*preload, prompt=prompt, constraints=constraints)
    return _constrain(*preload, prompt=prompt, constraints=constraints)

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
        print('*** Execution ***')
        for result in results:
            for r in result:
                print(r)
    return {k: [r[i] for r in results] for i, k in enumerate(['codes', 'files', 'souts', 'serrs'])}
