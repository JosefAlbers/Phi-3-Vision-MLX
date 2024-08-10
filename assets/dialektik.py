from pathlib import Path
from datasets import load_dataset, concatenate_datasets
import random
import json
import os
from datetime import datetime
from huggingface_hub import InferenceClient
import phi_3_vision_mlx as pv
import mlx.core as mx
from functools import partial
import fire

PATH_DS = 'JosefAlbers/StampyAI-alignment-research-dataset'
PROMPT_THESIS = "Based on the above bullet points, create a detailed and engaging article that explores the main themes and insights. For each bullet point, provide context, elaborate on the key ideas, and discuss their implications. Ensure the article flows logically, connects related concepts, and presents a coherent narrative."
PROMPT_ANTITHESIS = "Read through the article and write a response that challenges its main ideas. Offer different viewpoints, suggest alternative explanations, and propose new approaches. Keep your response well-structured and relevant to the original content."
PROMPT_SYNTHESIS = """You have an initial article and a response to it:

**Article:**
{thesis}

**Response:**
{antithesis}

Create an improved version of the article that incorporates insights from both the original and the response. Address conflicting ideas and present a more comprehensive view. Add new insights based on this broader perspective. Your final article should be clear, balanced, and offer a deeper understanding of the topic."""

def setup(instruction="\n<|end|>\n<|user|>\nTLDR: Summarize the following text into concise, stand-alone bullet points (max 3-5 bullet points). Each bullet point should be self-contained and provide a clear and complete idea without referencing other bullet points or the original text.", list_source=['agentmodels', 'distill', 'arbital', 'blogs', 'lesswrong', 'youtube', 'arxiv', 'special_docs'], quantize_model=False, batch_size=4, path_ds=PATH_DS):
    model, processor = pv.load(blind_model=True, quantize_model=quantize_model, quantize_cache=False, use_adapter=False)
    def aggregate(example):
        str_md = f"# {example['title']}\n\n{example['text']}"
        example['str_md'] = str_md
        example['len_md'] = processor(str_md)['input_ids'].size
        return example
    def summarize(example):
        markdowns = example['str_md']
        prompts = [f'{m}{instruction}' for m in markdowns]
        summaries = pv.generate(prompts, preload=(model, processor), stream=False, verbose=False, max_tokens=512)
        example['sum_md'] = summaries
        return example
    list_ds = []
    try:
        _ds_prev = load_dataset(path_ds, token=os.getenv("HF_WRITE_TOKEN"), split='train')
        list_source = [i for i in list_source if i not in _ds_prev['source']]
        list_ds.append(_ds_prev)
    except:
        print('Dataset not found.')
    for src in list_source:
        ds = load_dataset('StampyAI/alignment-research-dataset', src, trust_remote_code=True, split='train')
        ds = ds.select_columns(['id', 'source', 'title', 'text', 'url', 'date_published', 'authors', 'summary', 'source_type'])
        ds = ds.map(aggregate)
        ds = ds.filter(lambda example: 600 < example["len_md"] < 6000)
        if batch_size > 1:
            ds = ds.sort('len_md')
        ds = ds.map(summarize, batched=True, batch_size=batch_size)
        ds = ds.filter(lambda example: ('<unk>' not in example['sum_md']) and ('<|end|>' in example['sum_md']))
        list_ds.append(ds)
    ds = concatenate_datasets(list_ds)
    ds.push_to_hub(path_ds, token=os.getenv("HF_WRITE_TOKEN"), private=True)

def load_books(list_source=None, list_exclude=None, path_ds=PATH_DS):
    ds = load_dataset(path_ds, token=os.getenv("HF_READ_TOKEN", None), split='train')
    if list_source:
        list_source = [list_source] if isinstance(list_source, str) else list_source
        ds = ds.filter(lambda example: example['source'] in list_source)
    if list_exclude:
        list_exclude = [list_exclude] if isinstance(list_exclude, str) else list_exclude
        ds = ds.filter(lambda example: not any(word in example['sum_md'] for word in list_exclude))
    print(f"Loaded {len(ds)} from {', '.join(set(ds['source']))}")
    books = ds['sum_md']
    books = [i.split('\n- ') for i in books]
    clean_str = lambda s: s[2:] if s.startswith('- ') else s[:-7] if s.endswith('<|end|>') else s
    books = [[clean_str(s).strip() for s in book] for book in books]
    return books

def pick_books(topic, list_idx, list_books, num_book=3):
    if topic is None:
        return random.sample(range(len(list_books)), num_book)
    list_rand = list_idx if list_idx else random.sample(range(len(list_books)), 100)
    list_text = [list_books[i][0] for i in list_rand]
    embed = pv.GteModel()
    l = embed(list_text)
    q = embed(topic)
    scores = mx.matmul(q, l.T)
    list_idx = mx.argsort(scores)[:,:-1-num_book:-1].tolist()
    list_idx = list_idx[0]
    return [list_rand[i] for i in list_idx]

def get_bullets(topic='AI agents', list_source=None, list_exclude=['MIRI', 'Machine Intelligence Research Institute'], list_idx=None, num_book=3, per_book=3):
    books = load_books(list_source, list_exclude)
    list_idx = pick_books(topic, list_idx, books, num_book)
    print(f"Picked {list_idx}")
    picks = [books[i] for i in list_idx]
    bullets = ''
    for pick in picks:
        pick=pick[:per_book]
        bullets += '- ' + '\n    - '.join(pick) + '\n'
    bullets = bullets.strip()
    print(f'Bullets:\n{bullets}')
    return bullets, list_idx

def save_output(output, file_suffix=None, base_folder='syntheses'):
    file_suffix = f'_{file_suffix}' if file_suffix else ''
    os.makedirs(base_folder, exist_ok=True)
    date_str = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    filename = os.path.join(base_folder, f'{date_str}{file_suffix}.md')
    with open(filename, 'w') as f:
        f.write(output)

def synthesize(topic=None, prompt_thesis=PROMPT_THESIS, prompt_antithesis=PROMPT_ANTITHESIS, prompt_synthesis=PROMPT_SYNTHESIS,
               list_source=None, list_exclude=['MIRI', 'Machine Intelligence Research Institute'],
               list_idx=None, num_book=3, per_book=3, llm_model=None):
    if llm_model is None:
        preload = pv.load(blind_model=True, quantize_model=True)
        generate = partial(pv.generate, preload=preload)
    else:
        generate = partial(pv.mistral_api, api_model=llm_model, history=None, return_dict=False, verbose=False)
    bullets, list_idx = get_bullets(topic, list_source, list_exclude, list_idx, num_book, per_book)
    prompt = f"{bullets}\n\n{prompt_thesis}"
    thesis_output = generate(prompt)
    prompt_anti = f'{thesis_output}\n\n{prompt_antithesis}'
    antithesis_output = generate(prompt_anti)
    prompt_synth = prompt_synthesis.format(thesis=thesis_output, antithesis=antithesis_output)
    synthesis_output = generate(prompt_synth)
    all_output = f'Thesis:\n---\n\n{thesis_output}\n\nAntithesis:\n---\n\n{antithesis_output}\n\nSynthesis:\n---\n\n{synthesis_output}\n\nArguments:\n---\n\ndialektik.synthesize({list_source=}, {list_exclude=},{list_idx=}, {per_book=}, {llm_model=})\n\n{bullets}'
    save_output(all_output)
    return thesis_output, antithesis_output, synthesis_output

if __name__ == "__main__":
    fire.Fire(synthesize)
