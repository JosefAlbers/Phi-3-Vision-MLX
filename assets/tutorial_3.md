# Part 3: Implementing Batching for Phi-3-Vision in MLX

## Introduction

In this tutorial, we will explore how to implement batching for the Phi-3-Vision model in MLX. Batching enables the model to process multiple inputs in parallel, significantly enhancing computational efficiency and accelerating text generation.

The full implementation of this tutorial is available at https://github.com/JosefAlbers/Phi-3-Vision-MLX/tree/main/assets/tutorial_3.py

## 1. Understanding Batching

Batching is a technique that allows the model to process multiple inputs simultaneously. This approach is particularly advantageous for smaller large language models (sLLMs) like Phi-3, as it can massively speed up the text generation process.

## 2. Implementing Batching Utilities

To implement batching, we need to create utility functions that can handle padding, updating inputs, and generating attention masks.

### 2.1 Padding Function

The `pad_to_batch` function takes in a dictionary of inputs and returns a padded version of the inputs, along with the corresponding position IDs and attention masks.

```python
def pad_to_batch(inputs):
    input_ids = [i.tolist() for i in inputs['input_ids']]
    max_length = max(len(sublist) for sublist in input_ids)
    return {
        'input_ids': mx.array([[0]*(max_length-len(sublist)) + sublist for sublist in input_ids]),
        'position_ids': mx.array([[1]*(max_length-len(sublist)) + list(range(len(sublist))) for sublist in input_ids]),
        'attention_mask': mx.array([[0]*(max_length-len(sublist)) + [1]*len(sublist) for sublist in input_ids]),
    }
```

This function pads the inputs to the same length, adjusts the position IDs, and creates attention masks. Note that we're padding on the left side to preserve the causal structure of the input sequence, as required by autoregressive models.

### 2.2 Input Update Function

The `update_inputs` function updates the inputs with newly generated tokens, maintaining the correct structure for position IDs and attention masks.

```python
def update_inputs(inputs, token):
    input_ids, position_ids, attention_mask = inputs['input_ids'], inputs['position_ids'], inputs['attention_mask']
    return {
        'input_ids': mx.concatenate([input_ids, token[:,None]], axis=-1),
        'position_ids': mx.concatenate([position_ids, position_ids[:, -1:] + 1], axis=1),
        'attention_mask': mx.concatenate([attention_mask, mx.ones((input_ids.shape[0], 1), dtype=attention_mask.dtype)], axis=1),
    }
```

This function updates our inputs with newly generated tokens, maintaining the correct structure for position IDs and attention masks.

## 3. Modifying the Model for Batched Inputs

To enable batching, we need to update our model to use the `position_ids` and `attention_mask`.

### 3.1 Updating the Model Interface

We modify the top-level `Phi3VForCausalLM` class to accept the batched inputs and pass them to its model.

```python
class Phi3VForCausalLM(nn.Module):
    # ...
    def __call__(self, input_ids, pixel_values=None, image_sizes=None, position_ids=None, attention_mask=None):
        x = self.model(input_ids, pixel_values, image_sizes, position_ids, attention_mask)
        return self.lm_head(x)
```

### 3.2 Updating the Phi3VModel

Next, modify the `Phi3VModel` to pass `position_ids` and `attention_mask` to each layer:

```python
class Phi3VModel(nn.Module):
    # ...
    def __call__(self, input_ids, pixel_values, image_sizes, position_ids, attention_mask):
        x = self.embed_tokens(input_ids)
        x = self.vision_embed_tokens(x, pixel_values, image_sizes)
        for l in self.layers:
            x = l(x, position_ids, attention_mask)
        return self.norm(x)
```

### 3.3 Updating the Attention Mechanism

Finally, update the Phi3Attention module to utilize `position_ids` and `attention_mask`:

```python
class Phi3Attention(nn.Module):
    # ...
    def __call__(self, x, position_ids, attention_mask):
        # ...
        q, k = self.rope(q, k, position_ids)
        mask = mx.triu(mx.full((v.shape[2], v.shape[2]), -mx.inf), k=1)
        if attention_mask is not None:
            mask += mx.where(attention_mask[:, :, None]*attention_mask[:, None, :]==1, 0, -mx.inf)
            mask = mx.expand_dims(mask, 1)
        # ...
```

## 4. Using Batched Inputs

Here's an example of batched text generation:

```python
# Prepare batched inputs
inputs = processor(['Hello World!', 'Guten Tag!'], return_tensors='np')
inputs = pad_to_batch(inputs)

# Generate tokens
logits = model(**inputs)
token = mx.argmax(logits[:, -1, :], axis=-1)
list_tokens = [token]
for i in range(5):
    inputs = update_inputs(inputs, token)
    logits = model(**inputs)
    token = mx.argmax(logits[:, -1, :], axis=-1)
    list_tokens.append(token)
list_tokens = mx.stack(list_tokens, axis=1).tolist()
print(processor.tokenizer.batch_decode(list_tokens))
# Output: ['How are you doing today?', 'Was möchten Sie w']
```

## Conclusion

By implementing custom batching for Phi-3-Vision, we've enabled our model to efficiently handle multiple inputs while ensuring correct behavior for autoregressive generation. This approach provides fine-grained control over input processing, position IDs, and attention masks, which is crucial for optimal model performance.

In the next part, we'll explore implementing efficient caching mechanisms to further accelerate text generation, especially for longer sequences.