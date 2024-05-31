# Phi-3-Vision VLM Model for Apple MLX: An All-in-One Port

This project brings the powerful phi-3-vision VLM to Apple's MLX framework, offering a comprehensive solution for various text and image processing tasks. With a focus on simplicity and efficiency, this implementation offers a straightforward and minimalistic integration of the VLM model. It seamlessly incorporates essential functionalities such as generating quantized model weights, optimizing KV cache quantization during inference, facilitating LoRA/QLoRA training, and conducting model benchmarking, all encapsulated within a single file for convenient access and usage.

## Key Features

* **Su-scaled RoPE:** Implements Su-scaled Rotary Position Embeddings to manage sequences of up to 128K tokens.
* **Model Quantization:** Reduce model size for faster loading and deployment (2.3GB quantized vs 8.5GB original).
* **KV Cache Quantization:** Optimize inference for processing long contexts with minimal overhead (5.3s quantized vs 5.1s original).
* **LoRA Training:** Easily customize the model for specific tasks or datasets using LoRA.
* **Benchmarking:** Quickly assess model performance on any dataset (WIP).

## Usage

```python
prompt = "<|user|>\n<|image_1|>\nWhat is shown in this image?<|end|>\n<|assistant|>\n"
images = [Image.open(requests.get("https://assets-c4akfrf5b4d3f4b7.z01.azurefd.net/assets/2024/04/BMDataViz_661fb89f3845e.png" , stream=True).raw)]
```

**Image Captioning**

```python
model, processor = load()
generate(model, processor, prompt, images)
```

```zsh
The image displays a bar chart showing the percentage of
4.43s user 3.17s system 71% cpu 10.711 total
```

**Cache Quantization**

```python
model, processor = load(use_quantized_cache=True)
print(generate(model, processor,  "<|user|>Write an exciting sci-fi.<|end|>\n<|assistant|>\n"))
```

```zsh
Title: The Last Frontier\n\nIn the
2.49s user 4.52s system 131% cpu 5.325 total
```

**Model Quantization**

```python
quantize(from_path='phi3v', to_path='quantized_phi3v', q_group_size=64, q_bits=4)
```

```zsh
4.30s user 3.31s system 119% cpu 6.368 total
```

```python
model, processor = load(model_path='quantized_phi3v')
print(generate(model, processor, "<|user|>Write an exciting sci-fi.<|end|>\n<|assistant|>\n"))
```

```zsh
Title: The Quantum Leap\n\nIn
3.78s user 0.87s system 205% cpu 2.264 total
```

**LoRA Training**

```python
train_lora()
```

```zsh
22.50s user 27.58s system 22% cpu 3:41.58 total
```

![Alt text](assets/train_log.png)

**Benchmarking (WIP)**

```python
recall()
```

```zsh
10.65s user 10.98s system 37% cpu 57.669 total
```

## Installation

```zsh
git clone https://github.com/JosefAlbers/Phi-3-Vision-MLX.git
```

## Benchmarks

| Task                  | Vanilla Model | Quantized Model | Quantized KV Cache | LoRA Adapter |
|-----------------------|---------------|-----------------|--------------------|--------------|
| Image Captioning      | 10.71s        | 8.51s           | 12.79s             | 11.70s       |
| Text Generation       | 5.07s         | 2.24s           | 5.27s              | 5.10s        |

## License

This project is licensed under the [MIT License](LICENSE).