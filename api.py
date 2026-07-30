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

def mistral_api(prompt, history, verbose=True, return_dict=True, api_model="mistralai/Mistral-Nemo-Instruct-2407"):
    """
    Example:
    --------
    agent = Agent(toolchain = "responses, history = mistral_api(prompt, history)")
    agent('Write a neurology ICU admission note')
    """
    history = '<s>' if history is None else history
    history += f"[INST] {prompt} [/INST]"
    client = InferenceClient(api_model, token = os.environ.get('HF_READ_TOKEN', False))
    generate_kwargs = dict(
        temperature=0.9,
        max_new_tokens=8192,
        top_p=0.95,
        repetition_penalty=1.0,
        do_sample=True,
        seed=42,
        stream=False,
        details=False,
        return_full_text=False,
    )
    result = client.text_generation(history, **generate_kwargs)
    result = result.strip()
    history += f" {result}</s> "
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
    client = InferenceClient("suno/bark-small", token = os.environ.get('HF_READ_TOKEN', False))
    result = client.text_to_speech(prompt)
    Path("bark.flac").write_bytes(result)
    return prompt
