import os
from pathlib import Path

import requests
from huggingface_hub import InferenceClient

MINIMAX_TTS_DEFAULT_MODEL = "speech-2.8-hd"
MINIMAX_TTS_MODELS = (
    "speech-2.8-hd",
    "speech-2.8-turbo",
    "speech-2.6-hd",
    "speech-2.6-turbo",
    "speech-02-hd",
    "speech-02-turbo",
    "speech-01-hd",
    "speech-01-turbo",
)
MINIMAX_TTS_AUDIO_FORMATS = ("mp3", "wav", "flac", "pcm")
MINIMAX_TTS_BASE_URLS = {
    "global_en": "https://api.minimax.io/v1/t2a_v2",
    "cn_zh": "https://api.minimaxi.com/v1/t2a_v2",
}

def minimax_tts_api(prompt, model=MINIMAX_TTS_DEFAULT_MODEL, region="global_en",
                    voice_setting=None, output_format="mp3", verbose=True,
                    return_dict=True):
    """
    Synthesize speech from text through the MiniMax text-to-audio endpoint.

    Both the global (api.minimax.io) and China (api.minimaxi.com) T2A endpoints
    are supported via the ``region`` argument. The non-streaming response carries
    the audio payload as a hex string in ``data.audio``; ``base_resp.status_code``
    is ``0`` on success.

    Example:
    --------
    agent = Agent(toolchain = "responses = minimax_tts_api(prompt)")
    agent('People say nothing is impossible, but I do nothing every day.')
    """
    base_url = MINIMAX_TTS_BASE_URLS.get(region, MINIMAX_TTS_BASE_URLS["global_en"])
    extension = output_format if output_format in MINIMAX_TTS_AUDIO_FORMATS else "mp3"
    body = dict(
        model=model,
        text=prompt,
        stream=False,
        output_format=extension,
    )
    if voice_setting is not None:
        body["voice_setting"] = voice_setting
    response = requests.post(
        f"{base_url}",
        headers={
            "Authorization": f"Bearer {os.environ.get('MINIMAX_API_KEY', '')}",
            "Content-Type": "application/json",
        },
        json=body,
    )
    response.raise_for_status()
    payload = response.json()
    status_code = payload.get("base_resp", {}).get("status_code", -1)
    if status_code != 0:
        raise RuntimeError(f"MiniMax T2A request failed: {payload}")
    audio_hex = payload.get("data", {}).get("audio")
    audio_bytes = bytes.fromhex(audio_hex)
    Path(f"minimax_tts.{extension}").write_bytes(audio_bytes)
    if verbose:
        print(f'### Prompt ###\n{prompt}\n### Saved ###\nminimax_tts.{extension}')
    if return_dict:
        return {'responses': prompt}
    return prompt

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
