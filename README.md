# Phi-3-MLX: Language and Vision Models for Apple Silicon

Phi-3-MLX is a versatile AI framework that leverages both the Phi-3-Vision multimodal model and the recently updated ([July 2, 2024](https://x.com/reach_vb/status/1808056108319179012)) Phi-3-Mini-128K language model, optimized for Apple Silicon using the MLX framework. This project provides an easy-to-use interface for a wide range of AI tasks, from advanced text generation to visual question answering and code execution.

## Recent Updates: Phi-3 Mini Improvements

Microsoft has recently released significant updates to the Phi-3 Mini model, dramatically improving its capabilities:

- Substantially enhanced code understanding in Python, C++, Rust, and TypeScript
- Improved post-training for better-structured output
- Enhanced multi-turn instruction following
- Added support for the `<|system|>` tag
- Improved reasoning and long-context understanding


| Benchmarks | Original | June 2024 Update |
| :-         | :-       | :-               |
| Instruction Extra Hard | 5.7 | 5.9 |
| Instruction Hard | 5.0 | 5.2 |
| JSON Structure Output | 1.9 | 60.1 |
| XML Structure Output | 47.8 | 52.9 |
| GPQA	| 25.9	| 29.7 |
| MMLU	| 68.1	| 69.7 |
| **Average**	| **25.7**	| **37.3** |

RULER: a retrieval-based benchmark for long context understanding

| Model             | 4K   | 8K   | 16K  | 32K  | 64K  | 128K | Average |
| :-------------------| :------| :------| :------| :------| :------| :------| :---------|
| Original          | 86.7 | 78.1 | 75.6 | 70.3 | 58.9 | 43.3 | **68.8**    |
| June 2024 Update  | 92.4 | 91.1 | 90.8 | 87.9 | 79.8 | 65.6 | **84.6**    |

RepoQA: a benchmark for long context code understanding

| Model             | Python | C++ | Rust | Java | TypeScript | Average |
| :-------------------| :--------| :-----| :------| :------| :------------| :---------|
| Original          | 27     | 29  | 40   | 33   | 33         | **32.4**    |
| June 2024 Update  | 85     | 63  | 72   | 93   | 72         | **77**      |


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

#### Batch Text Generation

```python
# A list of prompts for batch generation
prompts = [
    "Explain the key concepts of quantum computing and provide a Rust code example demonstrating quantum superposition.",
    "Write a poem about the first snowfall of the year.",
    "Summarize the major events of the French Revolution.",
    "Describe a bustling alien marketplace on a distant planet with unique goods and creatures."
    "Implement a basic encryption algorithm in Python.",
]

# Generate responses using Phi-3-Vision (multimodal model)
generate(prompts, max_tokens=100)

# Generate responses using Phi-3-Mini-128K (language-only model)
generate(prompts, max_tokens=100, blind_model=True)
```

#### Model and Cache Quantization

```python
# Model quantization
generate("Describe the water cycle.", quantize_model=True)

# Cache quantization
generate("Explain quantum computing.", quantize_cache=True)
```

#### LoRA Fine-tuning

Training a LoRA Adapter:

```python
from phi_3_vision_mlx import train_lora

train_lora(
    lora_layers=5,  # Number of layers to apply LoRA
    lora_rank=16,   # Rank of the LoRA adaptation
    epochs=10,      # Number of training epochs
    lr=1e-4,        # Learning rate
    warmup=0.5,     # Fraction of steps for learning rate warmup
    dataset_path="JosefAlbers/akemiH_MedQA_Reason"
)
```

![Alt text](https://raw.githubusercontent.com/JosefAlbers/Phi-3-Vision-MLX/main/assets/train_log.png)

Generating Text with LoRA

```python
generate("Describe the potential applications of CRISPR gene editing in medicine.", adapter_path='adapters')
```

Comparing LoRA Adapters

```python
from phi_3_vision_mlx import test_lora

# Test model without LoRA adapter
test_lora(adapter_path=None)
# Output score: 0.6 (6/10)

# Test model with the trained LoRA adapter (using default path)
test_lora(adapter_path=True)
# Output score: 0.8 (8/10)

# Test model with a specific LoRA adapter path
test_lora(adapter_path="/path/to/your/lora/adapter")
```

### 2. Agentic Interactions

#### Multi-turn Conversations

```python
from phi_3_vision_mlx import Agent

# Create an instance of the Agent
agent = Agent()

# First interaction: Analyze an image
agent('Analyze this image and describe the architectural style:', 'https://images.metmuseum.org/CRDImages/rl/original/DP-19531-075.jpg')

# Second interaction: Follow-up question
agent('What historical period does this architecture likely belong to?')

# End the conversation
# This clears the agent's memory and prepares it for a new conversation
agent.end()
```

![Alt text](https://raw.githubusercontent.com/JosefAlbers/Phi-3-Vision-MLX/main/assets/vqa.png)

#### Generative Feedback Loop

```python
# Ask the agent to generate and execute code to create a plot
agent('Plot a Lissajous Curve.')

# Ask the agent to modify the generated code and create a new plot
agent('Modify the code to plot 3:4 frequency')
agent.end()
```

![Alt text](https://raw.githubusercontent.com/JosefAlbers/Phi-3-Vision-MLX/main/assets/coding_agent.png)

#### API Integration

```python
# Request the agent to generate an image
agent('Draw "A perfectly red apple, 32k HDR, studio lighting"')
agent.end()

# Request the agent to convert text to speech
agent('Speak "People say nothing is impossible, but I do nothing every day."')
agent.end()
```

![Alt text](https://raw.githubusercontent.com/JosefAlbers/Phi-3-Vision-MLX/main/assets/api_agent.png)

### 3. Custom Agent Toolchains

#### Example 1. In-Context Learning Agent

```python
from phi_3_vision_mlx import load_text

# Create a custom tool named 'add_text'
def add_text(prompt):
    prompt, path = prompt.split('@')
    return f'{load_text(path)}\n<|end|>\n<|user|>{prompt}'

# Define the toolchain as a string
toolchain = """
    prompt = add_text(prompt)
    responses = generate(prompt, images)
    """

# Create an Agent instance with the custom toolchain
agent = Agent(toolchain, early_stop=100)

# Run the agent
agent('How to inspect API endpoints? @https://raw.githubusercontent.com/gradio-app/gradio/main/guides/08_gradio-clients-and-lite/01_getting-started-with-the-python-client.md')
```

#### Example 2. Retrieval Augmented Coding Agent

```python
from phi_3_vision_mlx import VDB
import datasets

# Simulate user input
user_input = 'Comparison of Sortino Ratio for Bitcoin and Ethereum.'

# Create a custom RAG tool
def rag(prompt, repo_id="JosefAlbers/sharegpt_python_mlx", n_topk=1):
    ds = datasets.load_dataset(repo_id, split='train')
    vdb = VDB(ds)
    context = vdb(prompt, n_topk)[0][0]
    return f'{context}\n<|end|>\n<|user|>Plot: {prompt}'

# Define the toolchain
toolchain_plot = """
    prompt = rag(prompt)
    responses = generate(prompt, images)
    files = execute(responses, step)
    """

# Create an Agent instance with the RAG toolchain
agent = Agent(toolchain_plot, False)

# Run the agent with the user input
_, images = agent(user_input)
```

#### Example 3. Multi-Agent Interaction

```python
# Continued from Example 2 above
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
| Text Generation       |  8.46 tps     |  51.69 tps      |  6.94 tps       |  8.58 tps    |
| Image Captioning      |  7.72 tps     |  33.10 tps      |  1.75 tps       |  7.11 tps    |
| Batched Generation    |  103.47 tps     |  182.83 tps      |  38.72 tps       |  101.02 tps    |

*(On M1 Max 64GB)*

## License

This project is licensed under the [MIT License](LICENSE).

## Citation

<a href="https://zenodo.org/doi/10.5281/zenodo.11403221"><img src="https://zenodo.org/badge/806709541.svg" alt="DOI"></a>
