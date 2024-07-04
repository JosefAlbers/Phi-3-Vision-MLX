# Phi-3-MLX: Language and Vision Models for Apple Silicon

Phi-3-MLX is a versatile AI framework that leverages both the Phi-3-Vision multimodal model and the recently updated ([July 2, 2024](https://x.com/reach_vb/status/1808056108319179012)) Phi-3-Mini-128K language model, optimized for Apple Silicon using the MLX framework. This project provides an easy-to-use interface for a wide range of AI tasks, from advanced text generation to visual question answering and code execution.

## Recent Updates: Phi-3 Mini Improvements

Microsoft has recently released significant updates to the Phi-3 Mini model, dramatically improving its capabilities:

- Substantially enhanced code understanding in Python, C++, Rust, and TypeScript
- Improved post-training for better-structured output
- Enhanced multi-turn instruction following
- Added support for the `<|system|>` tag
- Improved reasoning and long-context understanding

## Features

- Support for the newly updated Phi-3-Mini-128K (language-only) model
- Integration with Phi-3-Vision (multimodal) model
- Optimized performance on Apple Silicon using MLX
- Batched generation for processing multiple prompts
- Flexible agent system for various AI tasks
- Custom toolchains for specialized workflows
- Model quantization for improved efficiency
- LoRA fine-tuning capabilities
- API integration for extended functionality (e.g., image generation, text-to-speech)

## Quick Start

Install and launch Phi-3-MLX from command line:

```bash
pip install phi-3-vision-mlx
phi3v
```

To instead use the library in a Python script:

```python
from phi_3_vision_mlx import generate
```

## Usage Examples

### 1. Core Functionalities

#### Visual Question Answering

```python
generate('What is shown in this image?', 'https://collectionapi.metmuseum.org/api/collection/v1/iiif/344291/725918/main-image')
```

#### Batch Generation

```python
prompts = [
    "Explain the key concepts of quantum computing and provide a Rust code example demonstrating quantum superposition.",
    "Write a poem about the first snowfall of the year.",
    "Summarize the major events of the French Revolution.",
    "Describe a bustling alien marketplace on a distant planet with unique goods and creatures."
    "Implement a basic encryption algorithm in Python.",
]

# `Phi-3-Vision
generate(prompts, max_tokens=100)
# `Phi-3-Mini-128K
generate(prompts, max_tokens=100, blind_model=True)
```

#### Model and Cache Quantization

```python
# `Model quantization
generate("Explain the implications of quantum entanglement in quantum computing.", quantize_model=True)
# `Cache quantization
generate("Describe the potential applications of CRISPR gene editing in medicine.", quantize_cache=True)
```

#### LoRA Fine-tuning

```python
from phi_3_vision_mlx import train_lora

train_lora(lora_layers=5, lora_rank=16, epochs=10, lr=1e-4, warmup=.5, mask_ratios=[.0], adapter_path='adapters', dataset_path = "JosefAlbers/akemiH_MedQA_Reason")
```

![Alt text](https://raw.githubusercontent.com/JosefAlbers/Phi-3-Vision-MLX/main/assets/train_log.png)

```python
generate("Write a cosmic horror.", adapter_path='adapters')
```

### 2. Agent Interactions

#### Multi-turn Conversations and Context Handling

```python
from phi_3_vision_mlx import Agent

agent = Agent()
agent('Analyze this image and describe the architectural style:', 'https://images.metmuseum.org/CRDImages/rl/original/DP-19531-075.jpg')
agent('What historical period does this architecture likely belong to?')
agent.end()
```

![Alt text](https://raw.githubusercontent.com/JosefAlbers/Phi-3-Vision-MLX/main/assets/vqa.png)

#### Generative Feedback Loop

```python
agent('Plot a Lissajous Curve.')
agent('Modify the code to plot 3:4 frequency')
agent.end()
```

![Alt text](https://raw.githubusercontent.com/JosefAlbers/Phi-3-Vision-MLX/main/assets/coding_agent.png)

#### Extending Capabilities with API Integration

```python
agent('Draw "A perfectly red apple, 32k HDR, studio lighting"')
agent.end()
agent('Speak "People say nothing is impossible, but I do nothing every day."')
agent.end()
```

![Alt text](https://raw.githubusercontent.com/JosefAlbers/Phi-3-Vision-MLX/main/assets/api_agent.png)

### 3. Toolchain Customization

#### Example 1. In-Context Learning

```python
from phi_3_vision_mlx import load_text

# Create tool
def add_text(prompt):
    prompt, path = prompt.split('@')
    return f'{load_text(path)}\n<|end|>\n<|user|>{prompt}'

# Chain tools
toolchain = """
    prompt = add_text(prompt)
    responses = generate(prompt, images)
    """

# Create agent
agent = Agent(toolchain, early_stop=100)

# Run agent
agent('How to inspect API endpoints? @https://raw.githubusercontent.com/gradio-app/gradio/main/guides/08_gradio-clients-and-lite/01_getting-started-with-the-python-client.md')
```

#### Example 2. Retrieval Augmented Coding

```python
from phi_3_vision_mlx import VDB
import datasets

# User proxy
user_input = 'Comparison of Sortino Ratio for Bitcoin and Ethereum.'

# Create tool
def rag(prompt, repo_id="JosefAlbers/sharegpt_python_mlx", n_topk=1):
    ds = datasets.load_dataset(repo_id, split='train')
    vdb = VDB(ds)
    context = vdb(prompt, n_topk)[0][0]
    return f'{context}\n<|end|>\n<|user|>Plot: {prompt}'

# Chain tools
toolchain_plot = """
    prompt = rag(prompt)
    responses = generate(prompt, images)
    files = execute(responses, step)
    """

# Create agent
agent = Agent(toolchain_plot, False)

# Run agent
_, images = agent(user_input)
```

#### Example 3. Multi-Agent Interaction

```python
# Continued from Example 2
agent_writer = Agent(early_stop=100)
agent_writer(f'Write a stock analysis report on: {user_input}', images)
```

## Benchmarks

```python
from phi_3_vision_mlx import benchmark

benchmark()
```

| Task                  | Vanilla Model | Quantized Model | Quantized Cache | LoRA Adapter |
|-----------------------|---------------|-----------------|-----------------|--------------|
| Text Generation       |  8.72 tps     |  55.97 tps       |  7.04 tps      |  8.71 tps    |
| Image Captioning      |  8.04 tps     |  32.48 tps       |  1.77 tps      |  8.00 tps    |
| Batched Generation    | 30.74 tps     | 106.94 tps       | 20.47 tps      | 30.72 tps    |

*(On an M1 Max 64GB)*

## License

This project is licensed under the [MIT License](LICENSE).

## Citation

<a href="https://zenodo.org/doi/10.5281/zenodo.11403221"><img src="https://zenodo.org/badge/806709541.svg" alt="DOI"></a>
