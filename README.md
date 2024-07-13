# Phi-3-MLX: Language and Vision Models for Apple Silicon

Phi-3-MLX is a versatile AI framework that leverages both the Phi-3-Vision multimodal model and the recently updated ([July 2, 2024](https://x.com/reach_vb/status/1808056108319179012)) Phi-3-Mini-128K language model, optimized for Apple Silicon using the MLX framework. This project provides an easy-to-use interface for a wide range of AI tasks, from advanced text generation to visual question answering and code execution.

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

## 1. Core Functionalities

### Visual Question Answering

```python
generate('What is shown in this image?', 'https://collectionapi.metmuseum.org/api/collection/v1/iiif/344291/725918/main-image')
```

### Batch Text Generation

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

### Model and Cache Quantization

```python
# Model quantization
generate("Describe the water cycle.", quantize_model=True)

# Cache quantization
generate("Explain quantum computing.", quantize_cache=True)
```

### Structured Generation Using Constrained Decoding (WIP)

The `constrain` function allows for structured generation, which can be useful for tasks like code generation, function calling, chain-of-thought prompting, or multiple-choice question answering.

```python
from phi_3_vision_mlx import constrain

# Define the prompt
prompt = "Write a Python function to calculate the Fibonacci sequence up to a given number n."

# Define constraints
constraints = [
    (100, "\n```python\n"), # Start of code block
    (100, " return "),      # Ensure a return statement
    (200, "\n```")],        # End of code block

# Apply constrained decoding using the 'constrain' function from phi_3_vision_mlx.
constrain(prompt, constraints)
```

The `constrain` function can also guide the model to provide reasoning before concluding with an answer. This approach can be especially helpful for multiple-choice questions, such as those in the Massive Multitask Language Understanding (MMLU) benchmark, where the model's thought process is as crucial as its final selection.

```python
prompts = [
    "A 20-year-old woman presents with menorrhagia for the past several years. She says that her menses “have always been heavy”, and she has experienced easy bruising for as long as she can remember. Family history is significant for her mother, who had similar problems with bruising easily. The patient's vital signs include: heart rate 98/min, respiratory rate 14/min, temperature 36.1°C (96.9°F), and blood pressure 110/87 mm Hg. Physical examination is unremarkable. Laboratory tests show the following: platelet count 200,000/mm3, PT 12 seconds, and PTT 43 seconds. Which of the following is the most likely cause of this patient’s symptoms? A: Factor V Leiden B: Hemophilia A C: Lupus anticoagulant D: Protein C deficiency E: Von Willebrand disease",
    "A 25-year-old primigravida presents to her physician for a routine prenatal visit. She is at 34 weeks gestation, as confirmed by an ultrasound examination. She has no complaints, but notes that the new shoes she bought 2 weeks ago do not fit anymore. The course of her pregnancy has been uneventful and she has been compliant with the recommended prenatal care. Her medical history is unremarkable. She has a 15-pound weight gain since the last visit 3 weeks ago. Her vital signs are as follows: blood pressure, 148/90 mm Hg; heart rate, 88/min; respiratory rate, 16/min; and temperature, 36.6℃ (97.9℉). The blood pressure on repeat assessment 4 hours later is 151/90 mm Hg. The fetal heart rate is 151/min. The physical examination is significant for 2+ pitting edema of the lower extremity. Which of the following tests o should confirm the probable condition of this patient? A: Bilirubin assessment B: Coagulation studies C: Hematocrit assessment D: Leukocyte count with differential E: 24-hour urine protein"]

constrain(prompts, constraints=[(30, ' The correct answer is'), (10, 'X.')], blind_model=True, quantize_model=True)
```

The constraints encourage a structured response that includes the thought process, making the output more informative and transparent:

```
< Generated text for prompt #0 >
The most likely cause of this patient's menorrhagia and easy bruising is E: Von Willebrand disease. The correct answer is Von Willebrand disease.

< Generated text for prompt #1 >
The patient's hypertension, edema, and weight gain are concerning for preeclampsia. The correct answer is E: 24-hour urine protein.
(phi) phi %
```

### (Q)LoRA Fine-tuning

Training a (Q)LoRA Adapter

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

Generating Text with (Q)LoRA

```python
generate("Describe the potential applications of CRISPR gene editing in medicine.",
    blind_model=True,
    quantize_model=True,
    use_adapter=True)
```

Comparing (Q)LoRA Adapters

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

## 2. Agent Interactions

### Multi-turn Conversation

```python
from phi_3_vision_mlx import Agent

# Create an instance of the Agent
agent = Agent()

# First interaction: Analyze an image
agent('Analyze this image and describe the architectural style:', 'https://images.metmuseum.org/CRDImages/rl/original/DP-19531-075.jpg')

# Second interaction: Follow-up question
agent('What historical period does this architecture likely belong to?')

# End the conversation: This clears the agent's memory and prepares it for a new conversation
agent.end()
```

![Alt text](https://raw.githubusercontent.com/JosefAlbers/Phi-3-Vision-MLX/main/assets/vqa.png)

### Generative Feedback Loop

```python
# Ask the agent to generate and execute code to create a plot
agent('Plot a Lissajous Curve.')

# Ask the agent to modify the generated code and create a new plot
agent('Modify the code to plot 3:4 frequency')
agent.end()
```

![Alt text](https://raw.githubusercontent.com/JosefAlbers/Phi-3-Vision-MLX/main/assets/coding_agent.png)

### External API Tool Use

```python
# Request the agent to generate an image
agent('Draw "A perfectly red apple, 32k HDR, studio lighting"')
agent.end()

# Request the agent to convert text to speech
agent('Speak "People say nothing is impossible, but I do nothing every day."')
agent.end()
```

![Alt text](https://raw.githubusercontent.com/JosefAlbers/Phi-3-Vision-MLX/main/assets/api_agent.png)

## 3. Custom Toolchains

### Example 1. In-Context Learning Agent

```python
from phi_3_vision_mlx import _load_text

# Create a custom tool named 'add_text'
def add_text(prompt):
    prompt, path = prompt.split('@')
    return f'{_load_text(path)}\n<|end|>\n<|user|>{prompt}'

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

### Example 2. Retrieval Augmented Coding Agent

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

### Example 3. Multi-Agent Interaction

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

## Documentation

API references and additional information are available at:

https://josefalbers.github.io/Phi-3-Vision-MLX/

## License

This project is licensed under the [MIT License](LICENSE).

## Citation

<a href="https://zenodo.org/doi/10.5281/zenodo.11403221"><img src="https://zenodo.org/badge/806709541.svg" alt="DOI"></a>
