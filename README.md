# Phi-3-MLX: Language and Vision Models for Apple Silicon

Phi-3-MLX is a versatile AI framework that leverages both the Phi-3-Vision multimodal model and the Phi-3-Mini-128K language model, optimized for Apple Silicon using the MLX framework. This project provides an easy-to-use interface for a wide range of AI tasks, from advanced text generation to visual question answering and code execution.

## Features

- Integration with [Phi-3.5-vision](https://huggingface.co/microsoft/Phi-3.5-vision-instruct) model
- Support for the [Phi-3.5-mini](https://huggingface.co/microsoft/Phi-3.5-mini-instruct) model
- Optimized performance on Apple Silicon using MLX
- Batched generation for processing multiple prompts
- Flexible agent system for various AI tasks
- Custom toolchains for specialized workflows
- Model quantization for improved efficiency
- LoRA fine-tuning capabilities
- API integration for extended functionality (e.g., image generation, text-to-speech)

## Minimum Requirements

Phi-3-MLX is designed to run on Apple Silicon Macs. The minimum requirements are:

- Apple Silicon Mac (M1, M2, or later)
- 8GB RAM (with quantization using `quantize_model=True` option)

For optimal performance, especially when working with larger models or datasets, we recommend using a Mac with 16GB RAM or more.

## Quick Start

Install and launch Phi-3-MLX from command line:

```bash
# Quick install (note: PyPI version may not always be up to date)
pip install phi-3-vision-mlx
phi3v

# For the latest version, you can install directly from the repository:
# git clone https://github.com/JosefAlbers/Phi-3-Vision-MLX.git
# cd Phi-3-Vision-MLX
# pip install -e .
```

To use the library in a Python script:

```python
from phi_3_vision_mlx import generate
```

## 1. Core Functionalities

### Visual Question Answering

```python
generate('What is shown in this image?', 'https://collectionapi.metmuseum.org/api/collection/v1/iiif/344291/725918/main-image')
```

### Model and Cache Quantization

```python
# Model quantization
generate("Describe the water cycle.", quantize_model=True)

# Cache quantization
generate("Explain quantum computing.", quantize_cache=True)
```

### Batch Text Generation

```python
# A list of prompts for batch generation
prompts = [
    "Write a haiku about spring.",
    "Explain the theory of relativity.",
    "Describe a futuristic city."
]
# Generate responses using Phi-3-Vision (multimodal model)
generate(prompts, max_tokens=100)

# Generate responses using Phi-3-Mini-128K (language-only model)
generate(prompts, max_tokens=100, blind_model=True)
```

### Constrained Beam Decoding

```python
from phi_3_vision_mlx import constrain

# Use constrain for structured generation (e.g., code, function calls, multiple-choice)
prompts = [
    "A 20-year-old woman presents with menorrhagia for the past several years. She says that her menses “have always been heavy”, and she has experienced easy bruising for as long as she can remember. Family history is significant for her mother, who had similar problems with bruising easily. The patient's vital signs include: heart rate 98/min, respiratory rate 14/min, temperature 36.1°C (96.9°F), and blood pressure 110/87 mm Hg. Physical examination is unremarkable. Laboratory tests show the following: platelet count 200,000/mm3, PT 12 seconds, and PTT 43 seconds. Which of the following is the most likely cause of this patient’s symptoms? A: Factor V Leiden B: Hemophilia A C: Lupus anticoagulant D: Protein C deficiency E: Von Willebrand disease",
    "A 25-year-old primigravida presents to her physician for a routine prenatal visit. She is at 34 weeks gestation, as confirmed by an ultrasound examination. She has no complaints, but notes that the new shoes she bought 2 weeks ago do not fit anymore. The course of her pregnancy has been uneventful and she has been compliant with the recommended prenatal care. Her medical history is unremarkable. She has a 15-pound weight gain since the last visit 3 weeks ago. Her vital signs are as follows: blood pressure, 148/90 mm Hg; heart rate, 88/min; respiratory rate, 16/min; and temperature, 36.6℃ (97.9℉). The blood pressure on repeat assessment 4 hours later is 151/90 mm Hg. The fetal heart rate is 151/min. The physical examination is significant for 2+ pitting edema of the lower extremity. Which of the following tests o should confirm the probable condition of this patient? A: Bilirubin assessment B: Coagulation studies C: Hematocrit assessment D: Leukocyte count with differential E: 24-hour urine protein"
]

# Define constraints for the generated text
constraints = [(0, '\nThe'), (100, ' The correct answer is'), (1, 'X.')]

# Apply constrained beam decoding
results = constrain(prompts, constraints, blind_model=True, quantize_model=True, use_beam=True)
```

### Choosing From Options

```python
from phi_3_vision_mlx import choose

# Select best option from choices for given prompts
prompts = [
    "What is the largest planet in our solar system? A: Earth B: Mars C: Jupiter D: Saturn",
    "Which element has the chemical symbol 'O'? A: Osmium B: Oxygen C: Gold D: Silver"
]

# For multiple-choice or decision-making tasks
choose(prompts, choices='ABCDE')
```

### LoRA Fine-tuning

```python
from phi_3_vision_mlx import train_lora, test_lora

# Train a LoRA adapter
train_lora(
    lora_layers=5,  # Number of layers to apply LoRA
    lora_rank=16,   # Rank of the LoRA adaptation
    epochs=10,      # Number of training epochs
    lr=1e-4,        # Learning rate
    warmup=0.5,     # Fraction of steps for learning rate warmup
    dataset_path="JosefAlbers/akemiH_MedQA_Reason"
)

# Generate text using the trained LoRA adapter
generate("Describe the potential applications of CRISPR gene editing in medicine.",
    blind_model=True,
    quantize_model=True,
    use_adapter=True)

# Test the performance of the trained LoRA adapter
test_lora()
```

![Alt text](https://raw.githubusercontent.com/JosefAlbers/Phi-3-Vision-MLX/main/assets/train_log.png)

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

# End conversation, clear memory for new interaction
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

### In-Context Learning Agent

```python
from phi_3_vision_mlx import add_text

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

### Retrieval Augmented Coding Agent

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
    return f'{context}\n<|end|>\n<|user|>\nPlot: {prompt}'

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

### Multi-Agent Interaction

```python
# Continued from Example 2 above
agent_writer = Agent(early_stop=100)
agent_writer(f'Write a stock analysis report on: {user_input}', images)
```

### External LLM Integration

```python
# Create Agent with Mistral-7B-Instruct-v0.3 instead
agent = Agent(toolchain = "responses, history = mistral_api(prompt, history)")

# Generate a neurology ICU admission note
agent('Write a neurology ICU admission note.')

# Follow-up questions (multi-turn conversation)
agent('Give me the inpatient BP goal for this patient.')
agent('DVT ppx for this patient?')
agent("Patient's prognosis?")

# End
agent.end()
```

## Benchmarks

```python
from phi_3_vision_mlx import benchmark

benchmark()
```

| Task                  | Vanilla Model | Quantized Model | Quantized Cache | LoRA Adapter |
|-----------------------|---------------|-----------------|-----------------|--------------|
| Text Generation       |  25.02 tps     |  61.01 tps      |  18.68 tps       |  24.72 tps    |
| Image Captioning      |  21.29 tps     |  44.26 tps      |  5.56 tps       |  20.48 tps    |
| Batched Generation    |  236.60 tps     |  149.23 tps      |  121.92 tps       |  232.78 tps    |

*(On M1 Max 64GB)*

## Documentation

API references and additional information are available at:

https://josefalbers.github.io/Phi-3-Vision-MLX/

Also check out our tutorial series available at:

https://medium.com/@albersj66

## License

This project is licensed under the [MIT License](LICENSE).

## Citation

<a href="https://zenodo.org/doi/10.5281/zenodo.11403221"><img src="https://zenodo.org/badge/806709541.svg" alt="DOI"></a>