# Phi-3 Vision Model Inference with MLX

This repository provides code for running inference with the Microsoft [Phi-3-vision-128k-instruct](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct) model on Apple Silicon. The code prioritizes a simplified, barebones implementation of the VLM model within the [MLX](https://github.com/ml-explore/mlx/issues/12) framework. 

## Key Features

- Su-scaled RoPE: Utilizes Su-scaled Rotary Position Embeddings for enhanced positional encoding.

## Example Usage

```python
generate("<|user|>Write a short funny sci-fi.<|end|>\n<|assistant|>\n")
```

```zsh
Title: The Great Galactic Gaffe

In the year 3021, the intergalactic space station, Orion-7, was the hub of all space travel and exploration. The station was a bustling metropol
```

```python
prompt = f"<|user|>\n<|image_1|>\nWhat is shown in this image?<|end|>\n<|assistant|>\n"
images = [Image.open(requests.get("https://assets-c4akfrf5b4d3f4b7.z01.azurefd.net/assets/2024/04/BMDataViz_661fb89f3845e.png" , stream=True).raw)]
generate(prompt, images)
```

```zsh
The image displays a bar chart showing the percentage of respondents who agree with various statements about their preparedness for meetings. The chart has five vertical bars, each representing a different statement, with percentages ranging from 75% to 7
```