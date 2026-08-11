import os
from pathlib import Path

import requests
from huggingface_hub import InferenceClient

MINIMAX_MODELS = ("MiniMax-M3", "MiniMax-M2.7")

MINIMAX_BASE_URLS = {
    "global_en": "https://api.minimax.io/v1",
    "cn_zh": "https://api.minimaxi.com/v1",
}

def minimax_api(prompt, history, verbose=True, return_dict=True, api_model="MiniMax-M3", region="global_en"):
    """
    Example:
    --------
    agent = Agent(toolchain = "responses, history = minimax_api(prompt, history)")
    agent('Write a neurology ICU admission note')
    """
    history = [] if history is None else history
    history = history + [{"role": "user", "content": prompt}]
    base_url = MINIMAX_BASE_URLS.get(region, MINIMAX_BASE_URLS["global_en"])
    response = requests.post(
        f"{base_url}/chat/completions",
        headers={
            "Authorization": f"Bearer {os.environ.get('MINIMAX_API_KEY', '')}",
            "Content-Type": "application/json",
        },
        json=dict(
            model=api_model,
            messages=history,
            temperature=0.9,
            top_p=0.95,
            max_tokens=8192,
            stream=False,
        ),
    )
    response.raise_for_status()
    result = response.json()["choices"][0]["message"]["content"].strip()
    history = history + [{"role": "assistant", "content": result}]
    if verbose:
        print(f'### Prompt ###\n{prompt}\n### Output ###\n{result}')
    if return_dict:
        return {'responses':result, 'history':history}
    return result

def deepseek_api(prompt, history, verbose=True, return_dict=True, api_model="deepseek-ai/DeepSeek-R1"):
    """
    Chat completion via Hugging Face's free Inference Providers router
    (https://router.huggingface.co/v1/chat/completions, OpenAI-compatible).

    Requires a free HF_TOKEN with the "Make calls to Inference Providers" scope.
    Create one at https://huggingface.co/settings/tokens, then `export HF_TOKEN=...`.
    Default model (deepseek-ai/DeepSeek-R1) is served free-tier via the Novita
    provider; any model listed at https://huggingface.co/inference/models works.

    Example:
    --------
    agent = Agent(toolchain = "responses, history = deepseek_api(prompt, history)")
    agent('Write a neurology ICU admission note')
    """
    token = os.environ.get('HF_TOKEN', '')
    if not token:
        raise RuntimeError("HF_TOKEN not set. Create a free token with 'Make calls to Inference Providers' scope at https://huggingface.co/settings/tokens, then `export HF_TOKEN=...`.")

    # Normalize history into an OpenAI-style messages list. Accepts None, the
    # legacy raw-string format, or an existing messages list.
    if history is None:
        messages = []
    elif isinstance(history, str):
        messages = [{"role": "user", "content": history}] if history else []
    elif isinstance(history, list):
        messages = [dict(m) for m in history]
    else:
        messages = []
    messages = messages + [{"role": "user", "content": prompt}]

    response = requests.post(
        "https://router.huggingface.co/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        json=dict(
            model=api_model,
            messages=messages,
            temperature=0.9,
            top_p=0.95,
            max_tokens=8192,
            stream=False,
        ),
    )
    if not response.ok:
        raise RuntimeError(f"HF router request failed ({response.status_code}): {response.text[:300]}")
    result = response.json()["choices"][0]["message"]["content"].strip()
    history = messages + [{"role": "assistant", "content": result}]
    if verbose:
        print(f'### Prompt ###\n{prompt}\n### Output ###\n{result}')
    if return_dict:
        return {'responses':result, 'history':history}
    return result

def bark_api(prompt):
    """
    Example:
    --------
    agent = Agent(toolchain = "responses = bark_api(prompt)")
    agent('We never really grow up, we only learn how to act in public.')
    """
    client = InferenceClient("suno/bark-small", token = os.environ.get('HF_TOKEN', False))
    result = client.text_to_speech(prompt)
    Path("bark.flac").write_bytes(result)
    return prompt
