# Part 1: Porting Phi-3-Vision to MLX

## Introduction

Welcome to Part 1 of our tutorial series on porting Phi-3-Vision from PyTorch to Apple's MLX framework. This guide will walk you through:

1. Analyzing the original PyTorch code
2. Translating core components to MLX
3. Building a basic MLX implementation
4. Loading and running the ported model

By following this guide, you'll gain an understanding of the process involved in porting AI models to MLX.

## 1. Finding and Understanding the Model Code

Our first task is to locate the code file for the original Phi-3-Vision implementation:

1. Visit the Hugging Face model hub: https://huggingface.co/models
2. Search for "phi-3-vision"
3. Click on the model to access its repository
4. Look for a file named `modeling_phi3_v.py`

Now open the file:

![Alt Text](tutorial_part1_meme_idonteven.gif)

But wait! Before you panic, here's how we'll approach this:

1. Scroll to the bottom of the file to find the top-level model class (`Phi3VForCausalLM` in our case)
2. Look for the `forward` method in this class
3. Trace the flow of data through the model by following method calls

By following these steps, we can identify five key components:

- Main model (`Phi3VModel`)
- Decoder layers (`Phi3DecoderLayer`)
- Attention mechanism (`Phi3Attention`)
- Feed-forward network (`Phi3MLP`)
- Image embedding (`Phi3ImageEmbedding`)

With these key components identified, we're ready to begin the translation process to MLX.

## 2. Differences between PyTorch and MLX

As we prepare to port the model, let's review a few differences between PyTorch and MLX that will be helpful in our translation process:

1. **Array Creation**: MLX doesn't require specifying device location (e.g., 'CPU', 'GPU').
2. **Lazy Computation**: Arrays in MLX are only materialized when `eval()` is called.
3. **Model Definition**: MLX uses `__call__` instead of `forward` for the model's forward pass.

Keep these differences in mind as we port Phi-3-Vision to MLX.

## 3. Understanding the Model Structure

Let's break down the key components of Phi-3-Vision:

### 3.1 Top-Level Model: Phi3VForCausalLM

This class serves as the main entry point of the model. It encapsulates the core `Phi3VModel` and adds a language modeling head.

```python
class Phi3VForCausalLM(nn.Module):
    # ...
    def __call__(self, input_ids, pixel_values=None, image_sizes=None):
        x = self.model(input_ids, pixel_values, image_sizes)
        return self.lm_head(x)
```

This top-level class serves two main functions:

1. It encapsulates the core model (`Phi3VModel`), which processes the input and generates contextualized representations.
2. It applies a linear transformation (the "language model head") to these representations, converting them into logits over the entire vocabulary. These logits represent the model's predictions for the next token in the sequence.

### 3.2 Core Model: Phi3VModel

The Phi3VModel implements the main transformer architecture.

```python
class Phi3VModel(nn.Module):
    # ...
    def __call__(self, input_ids, pixel_values, image_sizes):
        x = self.embed_tokens(input_ids)
        x = self.vision_embed_tokens(x, pixel_values, image_sizes)
        for l in self.layers:
            x = l(x)
        return self.norm(x)
```

This class processes inputs through four stages:

1. **Text Embedding**: Input tokens are converted to dense vector representations.
2. **Vision Embedding**: If present, image inputs are processed and integrated with the text embeddings.
3. **Transformer Layers**: The combined embeddings are then passed through a series of decoder layers.
4. **Normalization**: The output is normalized before being returned.

### 3.3 Image Embedding: Phi3ImageEmbedding

This component processes image inputs and integrates them with text embeddings. 

```python
class Phi3ImageEmbedding(nn.Module):
    # ...
    def __call__(self, txt_embeds, img_embeds, img_sizes, positions):
        # Process images with CLIP
        img_features = self.img_processor.vision_model(img_embeds)
        
        # Reshape and concatenate features
        global_features = self._process_global_features(img_features)
        local_features = self._process_local_features(img_features, img_sizes)
        
        # Apply additional projections
        x = mx.concatenate([local_features, global_features], axis=1)
        for layer in self.img_projection:
            x = layer(x)
        
        # Integrate with text embeddings
        txt_embeds = self._integrate_features(txt_embeds, x, positions)
        return txt_embeds
```

This class combines a CLIP (Contrastive Language-Image Pre-training) model with custom processing steps:

1. **CLIP Processing**: The model uses a pre-trained CLIP vision model to extract initial features from the input images.
2. **Additional Processing**: After CLIP processing, the model applies additional processing steps:
   - It reshapes and concatenates the features for both global and local (sub-image) representations.
   - It applies additional linear projections and non-linear activations (GELU) to further process these features.
3. **Integration with Text Embeddings**: Finally, the processed image features are integrated with the text embeddings at specific positions in the input sequence.

### 3.4 Decoder Layer: Phi3DecoderLayer

Each decoder layer is a fundamental building block of the transformer architecture.

```python
class Phi3DecoderLayer(nn.Module):
    # ...
    def __call__(self, x):
        r = self.self_attn(self.input_layernorm(x))
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r
```

The decoder layer performs a series of operations to its input:

1. **Self-Attention**: This mechanism allows the model to weigh the importance of different parts of the input when processing each element, enabling it to capture long-range dependencies in the sequence.
2. **Feedforward Neural Network (MLP)**: This subnet processes each position independently, introducing non-linearity and increasing the model's capacity to learn complex functions.
3. **Residual Connections**: After both the self-attention and MLP operations, the input is added to the output. This technique helps in mitigating the vanishing gradient problem and allows for easier training of deep networks.
4. **Layer Normalization**: Applied before the self-attention and MLP operations, this normalizes the inputs to each sub-layer, stabilizing the learning process and allowing for deeper networks.

The combination of these components enables each layer to refine and enrich the representations passed through the model.

### 3.5 Attention Mechanism: Phi3Attention

The attention mechanism allows the model to weigh the importance of different parts of the input when processing each element.

```python
class Phi3Attention(nn.Module):
    # ...
    def __call__(self, x):
        B, L, _ = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = mx.split(qkv, self.chop, axis=-1)
        q = q.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        mask = mx.triu(mx.full((x.shape[1], x.shape[1]), -mx.inf), k=1)
        w = (q * self.scale) @ k.transpose(0, 2, 3, 1)
        w += mask
        w = mx.softmax(w, axis=-1)
        o = w @ v
        o = o.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(o).astype(qkv.dtype)
```

Key aspects of this implementation:

1. **Projection and Splitting**: The input is first projected into query (q), key (k), and value (v) representations using a single linear projection (`qkv_proj`), which is then split.
2. **Multi-head Reshaping**: The q, k, and v tensors are reshaped to separate the heads and prepare for the attention computation.
3. **Attention Mask**: A causal mask is created to ensure that each position can only attend to previous positions.
4. **Scaled Dot-Product Attention**: The core attention computation is performed. 

    FYI, MLX provides a faster optimized version of this operation:

    ```python
    # This:
    w = (q * self.scale) @ k.transpose(0, 1, 3, 2)
    w += mask
    w = mx.softmax(w, axis=-1)
    o = w @ v

    # Is equivalent to:
    o = mx.fast.scaled_dot_product_attention(q,k,v,scale=self.scale,mask=mask)
    ```

5. **Output Projection**: The attention output is reshaped and projected back to the original dimensionality.

The attention mechanism supports both standard multi-head attention and grouped-query attention by allowing different numbers of heads for queries (`n_heads`) versus keys/values (`n_kv_heads`). 

![https://arxiv.org/abs/2305.13245v3](tutorial_part1_gqa.png)

In the current configuration, however, these are set to the same value (32), resulting in standard multi-head attention.

### 3.6 MLP Layer: Phi3MLP

The MLP layer applies non-linear transformations to the attention outputs.

```python
class Phi3MLP(nn.Module):
    # ...
    def __call__(self, x):
        x = self.gate_up_proj(x)
        gate, x = mx.split(x, 2, axis=-1)
        return self.down_proj(nn.silu(gate) * x)
```

This implements a gated feedforward network:

1. Gated Architecture:
   - The input is first projected into two separate spaces: one for the 'gate' and one for the 'values'.
   - This is achieved through a single linear projection followed by a split operation.
2. Activation Function:
   - The gate portion uses the SiLU (Sigmoid Linear Unit) activation, also known as the swish function.
   - SiLU is defined as f(x) = x * sigmoid(x), which has been shown to perform well in deep networks.
3. Gating Mechanism:
   - The activated gate is element-wise multiplied with the value portion.
   - This allows the network to dynamically control information flow, potentially helping with gradient flow and enabling more complex functions to be learned.
4. Final Projection:
   - The gated output is then projected back to the model's hidden size through a final linear layer.

This design combines the benefits of gating mechanisms (often seen in LSTMs and GRUs) with the simplicity and effectiveness of feedforward networks, potentially allowing for more expressive computations within each transformer layer.

## 4. Loading and Using the Model

Now that we've ported our model to MLX, let's load and use it for text generation.

First, we'll download the model configuration and weights from huggingface:

```python
model_path = snapshot_download('microsoft/Phi-3-vision-128k-instruct')
```

Next, we'll load the model configuration:

```python
with open(f"{model_path}/config.json", "r") as f:
    config = json.load(f)
model_config = SimpleNamespace(**config)
```

Now, let's load and "sanitize" the model weights:

```python
model_weight = [(k, v.transpose(0, 2, 3, 1) if "patch_embedding.weight" in k else v) 
                for wf in glob.glob(f"{model_path}/*.safetensors") 
                for k, v in mx.load(wf).items()]
```

The line `v.transpose(0, 2, 3, 1) if "patch_embedding.weight" in k else v` adapts the patch embedding weights to MLX's format by converting them from PyTorch's NCHW (batch, channel, height, width) to MLX's NHWC (batch, height, width, channel) format. This transposition, often called "weight sanitization", is necessary when porting the model from PyTorch to MLX.

With our configuration and weights ready, we can initialize and load our model:

```python
model = Phi3VForCausalLM(model_config)
model.load_weights(model_weight)
mx.eval(model.parameters())
model.eval()
```

Now that our model is loaded, let's use it to generate some text. First, we'll load the pretrained processor:

```python
processor = AutoProcessor.from_pretrained('microsoft/Phi-3-vision-128k-instruct', trust_remote_code=True)
```

Then, we'll process our input text and generate the first token:

```python
inputs = processor('Hello world!', return_tensors='np')
input_ids = mx.array(inputs['input_ids'])
logits = model(input_ids)
token = mx.argmax(logits[:, -1, :], axis=-1)
list_tokens = token.tolist()
```

This code processes the input text "Hello world!" and generates the first token. We use the `AutoProcessor` to tokenize the input, then pass it through the model to get logits. The token with the highest probability is selected as the next token.

To generate more tokens, we can use a simple loop:

```python
for i in range(5):
    input_ids = mx.concatenate([input_ids, token[None]], axis=-1)
    logits = model(input_ids)
    token = mx.argmax(logits[:, -1, :], axis=-1)
    list_tokens += token.tolist()
```

This loop generates five additional tokens by repeatedly feeding our model's output back as input.

```python
print(processor.tokenizer.decode(list_tokens))
# Output: How are you doing?<|end|>
```

And there you have it! We've successfully ported Phi-3-Vision to MLX, loaded the model, and generated text. While this implementation is basic, it demonstrates that our port is functional and capable of generating coherent text.

## 5. Limitations and Next Steps

While we've successfully ported the core structure of Phi-3-Vision to MLX, several key features are yet to be implemented:

1. **Position Encoding**: For encoding token positions in long contexts.
2. **Key-Value Caching**: For efficient autoregressive generation.
3. **Batch Processing**: For handling multiple inputs simultaneously.

These will be addressed in upcoming tutorials. Stay tuned as we continue to refine and enhance our MLX implementation of Phi-3-Vision!

## Conclusion

In this tutorial, we've laid the groundwork for porting Phi-3-Vision to MLX:

1. We analyzed the original model's architecture and core components.
2. We translated these components to MLX, adapting to its unique features.
3. We implemented a basic version of the model capable of text generation.
4. We loaded the model and demonstrated simple text generation.

The full implementation is available at https://github.com/JosefAlbers/Phi-3-Vision-MLX/tree/main/assets/tutorial_1.py