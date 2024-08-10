import os
from pathlib import Path

from huggingface_hub import InferenceClient

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
