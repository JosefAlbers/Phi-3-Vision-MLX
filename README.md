# Phi-3 Vision Model Inference with MLX

This repository provides code for running inference with the Microsoft Phi-3 vision model on Apple Silicon. Phi-3 is a powerful transformer model capable of understanding and generating text in response to both text prompts and images.

## Key Features

Su-scaled Rotary Position Embeddings: Utilizes Su-scaled Rotary Position Embeddings for enhanced positional encoding.

## Example Usage

```python
phi3v("<|user|>Write a short funny sci-fi.<|end|>\n<|assistant|>\n")
```

```zsh
Title: The Great Galactic Gaffe

In the year 3021, the intergalactic space station, Orion-7, was the hub of all space travel and exploration. The station was a bustling metropol
```

```python
prompt = f"<|user|>\n<|image_1|>\nWhat is shown in this image?<|end|>\n<|assistant|>\n"
images = [Image.open(requests.get("https://assets-c4akfrf5b4d3f4b7.z01.azurefd.net/assets/2024/04/BMDataViz_661fb89f3845e.png" , stream=True).raw)]
phi3v(prompt, images)
```

```zsh
The image displays a bar chart showing the percentage of respondents who agree with various statements about their preparedness for meetings. The chart has five vertical bars, each representing a different statement, with percentages ranging from 75% to 7
```