# Part 2: Implementing Su-scaled Rotary Position Embeddings (RoPE) for Phi-3-Vision

## Introduction

Welcome to Part 2 of our Phi-3-Vision porting series. In Part 1, we've created a basic implementation of the model in MLX. However, we also noted that it struggles with longer sequences. Today, we'll address this limitation by implementing Su-scaled Rotary Position Embeddings (RoPE), which will significantly enhance our model's ability to handle long contexts of up to 128K tokens.

The full implementation of this tutorial is available at https://github.com/JosefAlbers/Phi-3-Vision-MLX/tree/main/assets/tutorial_2.py

## 1. Understanding Rotary Position Embeddings (RoPE)

Before we delve into Su-scaled RoPE, let's first understand the basics of Rotary Position Embeddings (RoPE).

RoPE is a technique that injects positional information into the model's token representations without adding extra tokens or increasing the model's parameter count. The key idea is to apply a rotation to each token's embedding based on its position in the sequence.

1. **Frequency Calculation**: For each dimension d in the embedding space, RoPE calculates a frequency:

    ```python
    inv_freq = 1 / (theta ** (d / dim))
    ```

    <img src="https://raw.githubusercontent.com/JosefAlbers/Phi-3-Vision-MLX/main/assets/tutorial_part2_rope_inv.png">

    <details><summary>Code for the above plot</summary><pre>
    ```python
    import numpy as np
    import matplotlib.pyplot as plt

    dim = 3072 // 32  # hidden_size / num_heads
    position_ids = np.arange(15)[None]
    inv_freq_shape = np.arange(0, dim, 2) / dim
    inv_freq_vanilla = 1.0 / (10000.0**inv_freq_shape)

    plt.figure(figsize=(10, 6))
    plt.title('Inverse Frequencies in RoPE')
    plt.plot(inv_freq_vanilla, label='Vanilla RoPE')
    plt.xlabel('Dimension')
    plt.ylabel('Inverse Frequency')
    plt.legend()
    plt.show()
    ```
    </pre></details><br>


2. **Position-Frequency Interaction**: These frequencies are then multiplied by the token positions to create unique sinusoidal patterns for each position.

    ```python
    freqs = inv_freq @ position_ids.T
    ```

    <img src="https://raw.githubusercontent.com/JosefAlbers/Phi-3-Vision-MLX/main/assets/tutorial_part2_rope_int.png">

    <details><summary>Code for the above plot</summary><pre>
    ```python
    inv_freq_expanded = np.repeat(inv_freq_vanilla[None, :, None], position_ids.shape[0], axis=0)
    position_ids_expanded = position_ids[:, None, :]
    freqs = np.matmul(inv_freq_expanded, position_ids_expanded).transpose(0, 2, 1)

    plt.figure(figsize=(12, 6))
    plt.title('Frequency-Position Interaction in RoPE')
    for i, freq_i in enumerate(freqs[0]):
        plt.plot(freq_i.flatten(), label=f'Position {i}')
    plt.xlabel('Dimension')
    plt.ylabel('Frequency')
    plt.legend(title="Position IDs", loc="upper right", bbox_to_anchor=(1.15, 1), frameon=False)
    plt.tight_layout()
    plt.show()
    ```
    </pre></details><br>


3. **Rotation Application**: The resulting patterns are used to rotate the token embeddings in 2D planes.

    For a token at position `pos`, RoPE applies the following rotation:

    ```python
    x_rotated = [x * cos(pos * freq) - y * sin(pos * freq),
                 y * cos(pos * freq) + x * sin(pos * freq)]
    ```

    <img src="https://raw.githubusercontent.com/JosefAlbers/Phi-3-Vision-MLX/main/assets/tutorial_part2_rope_rot.png">

    <details><summary>Code for the above plot</summary><pre>
    ```python
    dim = 64  # Example dimension
    max_seq_len = 100

    # Calculate frequencies
    freqs = 1.0 / (10000 ** (np.arange(0, dim, 2) / dim))

    # Generate position embeddings
    pos_enc = np.zeros((max_seq_len, dim))
    positions = np.arange(max_seq_len)[:, np.newaxis]
    angles = positions * freqs[np.newaxis, :]

    pos_enc[:, 0::2] = np.cos(angles)
    pos_enc[:, 1::2] = np.sin(angles)

    # Visualize
    plt.figure(figsize=(12, 6))
    plt.imshow(pos_enc, aspect='auto', cmap='coolwarm')
    plt.title('Vanilla RoPE Embeddings')
    plt.xlabel('Dimension')
    plt.ylabel('Position')
    plt.colorbar()
    plt.show()
    ```
    </pre></details><br>


Now that we understand RoPE, let's explore how Su-scaled RoPE builds upon and enhances this concept.

## 2. Understanding SuRoPE

SuRoPE extends RoPE by introducing scaling factors for different sequence length ranges. 

```python
freq = 1 / (SU_FACTOR * theta ** (d / dim))
```

This allows the model to better generalize to sequences longer than those seen during training.

1. **Short and Long Factors**: Two sets of scaling factors are used, one for shorter sequences and one for longer sequences.

    <img src="https://raw.githubusercontent.com/JosefAlbers/Phi-3-Vision-MLX/main/assets/tutorial_part2_surope_fac.png">
    
    <details><summary>Code for the above plot</summary><pre>
    ```python
    import numpy as np
    import matplotlib.pyplot as plt

    dim = 3072 // 32  # hidden_size / num_heads
    position_ids = np.arange(15)[None]
    inv_freq_shape = np.arange(0, dim, 2) / dim

    short_factors = np.array([1.05, 1.05, 1.05, 1.10, 1.10, 1.10, 1.25, 1.25, 1.40, 1.45, 1.55, 1.85, 1.90, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.10, 2.10, 2.20, 2.35, 2.35, 2.35, 2.35, 2.40, 2.40, 2.65, 2.70, 2.90, 2.95, 3.05, 3.05, 3.05, 3.05])
    long_factors = np.array([1.03, 1.05, 1.05, 1.08, 1.23, 1.23, 1.30, 1.45, 1.60, 1.65, 1.90, 2.86, 3.69, 5.42, 5.49, 5.49, 9.09, 11.58, 15.66, 15.77, 15.79, 18.36, 22.00, 23.08, 30.01, 32.35, 32.59, 35.56, 39.95, 53.84, 56.20, 57.95, 59.29, 59.77, 59.92, 61.19, 61.96, 62.50, 63.37, 63.48, 63.48, 63.66, 63.85, 64.08, 64.76, 64.80, 64.81, 64.81])

    plt.figure(figsize=(10, 6))
    plt.title('Su-scaled RoPE Factors')
    plt.plot(short_factors, label='Short Factors')
    plt.plot(long_factors, label='Long Factors')
    plt.xlabel('Factor Index')
    plt.ylabel('Factor Value')
    plt.legend()

    inv_freq_vanilla = 1.0 / (10000.0**inv_freq_shape)
    inv_freq_short = 1.0 / (short_factors * 10000.0**inv_freq_shape)
    inv_freq_long = 1.0 / (long_factors * 10000.0**inv_freq_shape)
    plt.show()
    ```
    </pre></details><br>


2. **Adaptive Scaling**: The choice between short and long factors is made based on the sequence length.

    <img src="https://raw.githubusercontent.com/JosefAlbers/Phi-3-Vision-MLX/main/assets/tutorial_part2_surope_inv.png">

    <details><summary>Code for the above plot</summary><pre>
    ```python
    plt.figure(figsize=(10, 6))
    plt.title('Inverse Frequencies with Su-scaling')
    plt.plot(inv_freq_vanilla, label='Vanilla RoPE')
    plt.plot(inv_freq_short, label='Short-scaled RoPE')
    plt.plot(inv_freq_long, label='Long-scaled RoPE')
    plt.xlabel('Dimension')
    plt.ylabel('Inverse Frequency')
    plt.legend()
    plt.tight_layout()
    plt.show()
    ```
    </pre></details><br>


3. **Scaling Factor**: An additional scaling factor is applied to adjust for the extended maximum position embeddings.

## 3. Implementing Su-scaled RoPE

Now that we understand the theory behind Su-scaled RoPE, let's implement it in code. We'll create a `SuRoPE` class that encapsulates all the functionality we've discussed:

```python
import mlx.core as mx
import mlx.nn as nn
import math

class SuRoPE:
    def __init__(self, config):
        self.dim = config.hidden_size // config.num_attention_heads
        self.original_max_position_embeddings = config.original_max_position_embeddings
        self.rope_theta = config.rope_theta
        self.scaling_factor = math.sqrt(1 + math.log(config.max_position_embeddings / config.original_max_position_embeddings) / math.log(config.original_max_position_embeddings))
        self.long_factor = config.rope_scaling["long_factor"]
        self.short_factor = config.rope_scaling["short_factor"]

    def __call__(self, q, k, position_ids=None):
        position_ids = mx.arange(q.shape[2], dtype=mx.float32)[None] if position_ids is None else position_ids
        cos, sin = self._get_cos_sin(position_ids)
        q = (q * cos) + (self._rotate_half(q) * sin)
        k = (k * cos) + (self._rotate_half(k) * sin)
        return q, k

    def _get_cos_sin(self, position_ids):
        su_factor = self.long_factor if mx.max(position_ids) > self.original_max_position_embeddings else self.short_factor
        position_ids_expanded = position_ids[:, None, :]
        inv_freq = 1.0 / (mx.array(su_factor, dtype=mx.float32) * self.rope_theta**(mx.arange(0, self.dim, 2, dtype=mx.float32) / self.dim))
        inv_freq_expanded = mx.repeat(inv_freq[None, :, None], position_ids.shape[0], axis=0)
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(0, 2, 1)
        emb = mx.concatenate([freqs, freqs], axis=-1)
        cos = mx.expand_dims(mx.cos(emb) * self.scaling_factor, axis=1)
        sin = mx.expand_dims(mx.sin(emb) * self.scaling_factor, axis=1)
        return cos, sin

    @staticmethod
    def _rotate_half(x):
        midpoint = x.shape[-1] // 2
        x1, x2 = x[..., :midpoint], x[..., midpoint:]
        return mx.concatenate([-x2, x1], axis=-1)
```

## 4. Integrating Su-scaled RoPE into Phi-3-Vision

Integrating our Su-scaled RoPE implementation into the Phi-3-Vision model is straightforward. We only need to add two lines to our `Phi3Attention` module:

```python
class Phi3Attention(nn.Module):
    def __init__(self, config):
        # ...
        self.rope = SuRoPE(config)

    def __call__(self, x):
        # ...
        q, k = self.rope(q, k)
        # ...
```

These simple modifications allow our model to leverage Su-scaled RoPE, enabling it to handle sequences up to 128K tokens effectively.

## 5. Using the Updated Phi-3-Vision Model

Let's try an example that includes both text and an image:

```python
from PIL import Image
import requests
prompt = f"<|user|>\n<|image_1|>\nWhat is shown in this image?<|end|>\n<|assistant|>\n"
images = [Image.open(requests.get("https://assets-c4akfrf5b4d3f4b7.z01.azurefd.net/assets/2024/04/BMDataViz_661fb89f3845e.png" , stream=True).raw)]
inputs = processor(prompt, images, return_tensors="np")
input_ids, pixel_values, image_sizes = [mx.array(inputs[i]) for i in ['input_ids', 'pixel_values', 'image_sizes']]
print(input_ids.shape)
# Output: (1, 1939)
```

Note that the input is translated into 1939 tokens. Let's generate a response:

```python
logits = model(input_ids, pixel_values, image_sizes)
token = mx.argmax(logits[:, -1, :], axis=-1)
list_tokens = token.tolist()
for i in range(50):
    input_ids = mx.concatenate([input_ids, token[None]], axis=-1)
    logits = model(input_ids)
    token = mx.argmax(logits[:, -1, :], axis=-1)
    list_tokens += token.tolist()
print(processor.tokenizer.decode(list_tokens))
# Output: The image displays a chart with a series of connected dots forming a line that trends upwards, indicating a positive correlation between two variables. The chart is labeled with 'X' on the horizontal axis and 'Y' on the vertical axis,
```

This example showcases the model's ability to process a long input sequence (1939 tokens from the image plus the text prompt) and generate a coherent response, demonstrating the effectiveness of our Su-scaled RoPE implementation.

## 6. Limitations

While our Su-scaled RoPE implementation enhances the model's capacity for long sequences, two key limitations remain:

1. **Single Input Processing**: The current implementation processes only one input at a time, limiting throughput for multiple queries.

2. **Inefficient Generation**: Our token-by-token generation without caching leads to unnecessary repeated computations, slowing down the process.

These issues will be addressed in upcoming tutorials, where we'll explore efficient batching and caching mechanisms to improve the model's speed and inefficiency.

## Conclusion

In this tutorial, we implemented Su-scaled Rotary Position Embeddings (SuRoPE), enabling our model to handle sequences up to 128K tokens. 

In Part 3, we'll explore batching techniques to further optimize our Phi-3-Vision implementation in MLX.