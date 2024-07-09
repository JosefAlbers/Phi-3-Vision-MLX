import os
from pathlib import Path

from huggingface_hub import InferenceClient

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
    agent('We never really grow up, we only learn how to act in public.')
    """
    client = InferenceClient("suno/bark-small", token = os.environ.get('HF_READ_TOKEN', False))
    result = client.text_to_speech(prompt)
    Path("bark.flac").write_bytes(result)
    return prompt
