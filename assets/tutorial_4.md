# Part 4: Implementing Caching for Phi-3-Vision in MLX

## Introduction

In this tutorial, we'll implement caching for our Phi-3-Vision model in MLX. Caching is a key optimization technique that can significantly improve the efficiency of language models, especially during text generation tasks. By storing and reusing intermediate computational results, we can reduce redundant calculations and speed up the overall inference process.

The full implementation of this tutorial is available at https://github.com/JosefAlbers/Phi-3-Vision-MLX/tree/main/assets/tutorial_4.py

## 1. The Need for Caching

Our previous implementation of the Phi-3-Vision model processes the entire input sequence from scratch for each new token. This approach becomes inefficient as the sequence grows:

```
Without Caching:

Iteration 1: [Prompt] -> Model -> Token 1
Iteration 2: [Prompt, Token 1] -> Model -> Token 2
Iteration 3: [Prompt, Token 1, Token 2] -> Model -> Token 3
```

This repetitive processing leads to unnecessary computations.

## 2. How Caching Helps

Caching solves this problem by storing and reusing intermediate computations from previous iterations:

```
With Caching:

Iteration 1: [Prompt] -> Model -> Token 1, Cache
Iteration 2: Cache + [Token 1] -> Model -> Token 2, Cache
Iteration 3: Cache + [Token 2] -> Model -> Token 3, Cache
```

Instead of processing the entire sequence each time, the model processes only the new token and uses the cached information for the rest.

## 3. Implementing Caching

To implement caching, we need to modify the attention mechanism and the model layers to handle the cache.

### 3.1 Modifying the Attention Mechanism

We modify the attention mechanism to handle both cached and non-cached scenarios. We add a cache parameter to the `__call__` method, which is used to store and retrieve the cached values:

```python
class Phi3Attention(nn.Module):
    # ...
    def __call__(self, x, position_ids, attention_mask, cache):
        # ...
        if cache is None:
            position_ids = mx.arange(q.shape[2], dtype=mx.float32)[None] if position_ids is None else position_ids
            q, k = self.rope(q, k, position_ids)
            mask = mx.triu(mx.full((v.shape[2], v.shape[2]), -mx.inf), k=1)
            if attention_mask is not None:
                mask += mx.where(attention_mask[:, :, None]*attention_mask[:, None, :]==1, 0, -mx.inf)
                mask = mx.expand_dims(mask, 1)
        else:
            past_k, past_v, past_p, past_m = cache
            position_ids = past_p[:,-1:]+1
            mask = mx.pad(past_m[:,:,-1:,:], ((0,0),(0,0),(0,0),(0,1)))
            q, k = self.rope(q, k, position_ids)
            k = mx.concatenate([past_k, k], axis=2)
            v = mx.concatenate([past_v, v], axis=2)
        
        cache = (k, v, position_ids, mask)
        # ...
        return self.o_proj(o).astype(qkv.dtype), cache
```

This modification allows the attention mechanism to either compute from scratch or use and update the cache, depending on whether a cache is provided.

### 3.2 Updating the Model Layers

Next, we update the model layers to handle the cache by adding a cache parameter to the `__call__` method and passing it through each layer.

```python
class Phi3DecoderLayer(nn.Module):
    # ...
    def __call__(self, x, position_ids, attention_mask, cache):
        r, cache = self.self_attn(self.input_layernorm(x), position_ids, attention_mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r, cache
        
class Phi3VModel(nn.Module):
    # ...
    def __call__(self, input_ids, pixel_values, image_sizes, position_ids, attention_mask, cache):
        x = self.embed_tokens(input_ids)
        x = self.vision_embed_tokens(x, pixel_values, image_sizes)
        cache = [None]*len(self.layers) if cache is None else cache
        for i, l in enumerate(self.layers):
            x, cache[i] = l(x, position_ids, attention_mask, cache[i])
        return self.norm(x), cache

class Phi3VForCausalLM(nn.Module):
    # ...
    def __call__(self, input_ids, pixel_values=None, image_sizes=None, position_ids=None, attention_mask=None, cache=None):
        x, cache = self.model(input_ids, pixel_values, image_sizes, position_ids, attention_mask, cache)
        return self.lm_head(x), cache
```

## 4. Using Caching

Here's an example use of caching in text generation:

```python
# Initial input processing
inputs = processor('Hello world!', return_tensors='np')
input_ids = mx.array(inputs['input_ids'])

# Initial forward pass
logits, cache = model(input_ids)
token = mx.argmax(logits[:, -1, :], axis=-1)
list_tokens = token.tolist()

# Generate additional tokens using cache
for i in range(5):
    logits, cache = model(token[:,None], cache=cache)
    token = mx.argmax(logits[:, -1, :], axis=-1)
    list_tokens += token.tolist()

print(processor.tokenizer.decode(list_tokens))
```

In this example, we first process the initial input and obtain the cache. Then, for each subsequent token generation, we use and update this cache, significantly reducing computation time for longer sequences.

## Conclusion

By implementing caching in our Phi-3-Vision model, we've significantly improved its efficiency for token generation, especially for longer sequences. This optimization is important for practical applications of large language models, enabling faster and more efficient text generation.

In the upcoming tutorials, we'll explore advanced decoding strategies that allow for greater control over the model's output. These techniques will enhance the versatility of Phi-3-Vision, enabling its adaptation to a wide range of specific tasks and requirements.