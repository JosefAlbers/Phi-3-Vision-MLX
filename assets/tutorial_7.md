# Part 7: Understanding LoRA Training with MLX

## Introduction

In this tutorial, we'll explore the concept of Low-Rank Adaptation (LoRA) and how it can be implemented for training language models using MLX. We'll use a simplified version of LoRA training for the Phi-3 model as an illustrative example.

The full implementation of this tutorial is available at https://github.com/JosefAlbers/Phi-3-Vision-MLX/tree/main/assets/tutorial_7.py

## Understanding LoRA

Low-Rank Adaptation (LoRA) is an efficient fine-tuning technique for large language models. It works by adding small, trainable rank decomposition matrices to existing weights in the model, allowing for task-specific adaptation with minimal additional parameters.

The key idea behind LoRA is to represent the weight update as a low-rank decomposition:

```
W = BA
```

Where:

- B is a matrix of shape (d_model, r)
- A is a matrix of shape (r, d_model)
- r is the rank of the decomposition (typically much smaller than d_model)

This approach offers several advantages:

1. **Efficient Parameter Updates**: LoRA allows for updating only a small subset of parameters, making fine-tuning more computationally efficient.
2. **Flexibility in Layer Selection**: LoRA can be applied to specific layers (often attention layers), allowing for targeted model adaptation.
3. **Rank Control**: The rank of the LoRA decomposition can be adjusted, offering a balance between model adaptability and efficiency.
4. **Low-Rank Update**: By using low-rank matrices for updates, LoRA significantly reduces the number of trainable parameters.
5. **Efficiency in Fine-Tuning**: The approach enables efficient task-specific adaptation of large language models without the need to update all parameters.
6. **Modular Adaptation**: By saving LoRA weights separately, different adapters can be easily swapped for various tasks, enhancing the model's versatility.

## Implementing LoRA in MLX

Let's break down the key components:

### 1. LoRA Linear Layer

We create a LoRALinear class that modifies the standard linear layer to incorporate LoRA matrices:

```python
class LoRALinear(nn.Module):
    # ...
    def __call__(self, x):
        y = self.linear(x)
        z = (self.dropout(x) @ self.lora_a) @ self.lora_b
        z = y + (self.scale * z)
        return z.astype(x.dtype)
```

This class adds trainable LoRA matrices (`lora_a` and `lora_b`) to an existing linear layer, enabling efficient fine-tuning.

### 2. Applying LoRA to the Model

To apply LoRA to specific layers of the model, we use a function that replaces standard linear layers with LoRA-enabled versions:

```python
def linear_to_lora_layers(model, lora_targets, lora_layers, lora_rank):
    def to_lora(layer):
        return LoRALinear.from_linear(layer, r=lora_rank, alpha=lora_rank, scale=1.0, dropout=0.0)
    for l in model.layers[-lora_layers:]:
        lora_layers = [(k, to_lora(m)) for k, m in l.named_modules() if k in lora_targets]
        l.update_modules(tree_unflatten(lora_layers))
```

This function allows us to selectively apply LoRA to specific parts of the model, typically attention layers.

### 3. Training Loop

The main training function, `train_lora`, orchestrates the LoRA fine-tuning process:

```python
def train_lora(model_path='microsoft/Phi-3-vision-128k-instruct',
               adapter_path='adapters',
               lora_targets=["self_attn.qkv_proj"],
               lora_layers=1,
               lora_rank=16,
               num_steps=3,
               learning_rate=1e-4):
```

Key steps in the training process:

#### Data Preparation

```python
def prepare_batch(index):
    tokens = mx.array(processor.tokenizer.encode(dataset[index]))[None]
    input_ids = tokens[:, :-1]
    target_ids = tokens[:, 1:]
    return input_ids, target_ids
```

This function prepares a single training example. It tokenizes the input text, converts it to an MLX array, and splits it into input and target sequences. The `[None]` adds a batch dimension, and `[:, :-1]` and `[:, 1:]` create overlapping sequences for next-token prediction.

#### Loss Computation

```python
def compute_loss(model, input_ids, target_ids):
    logits, _ = model(input_ids)
    return nn.losses.cross_entropy(logits, target_ids, reduction='mean')
```

This function computes the loss for a batch. It passes the input through the model to get logits, then calculates the cross-entropy loss between these logits and the target ids. The `reduction='mean'` argument ensures we get the average loss across all tokens.

#### Model Setup

```python
model, processor = load(model_path)
model.freeze()
linear_to_lora_layers(model, lora_targets, lora_layers, lora_rank)
model.train()
```

Here, we load the pre-trained model and its processor, freeze the base model parameters, apply LoRA to specified layers, and set the model to training mode. Freezing the base model ensures only the LoRA parameters are updated during training.

#### Optimization Setup

```python
loss_and_grad_fn = nn.value_and_grad(model, compute_loss)
optimizer = optim.AdamW(learning_rate=learning_rate)
```

We create a function that computes both the loss and its gradient with respect to the model parameters. We also initialize the AdamW optimizer with the specified learning rate.

#### Training Loop

```python
for step in range(num_steps):
    input_ids, target_ids = prepare_batch(step)
    loss, gradients = loss_and_grad_fn(model, input_ids, target_ids)
    optimizer.update(model, gradients)
    mx.eval(model_state, loss)
```

This loop prepares batches, computes loss and gradients, updates model parameters, and evaluates the model state and loss.

#### Saving LoRA Weights

```python
mx.save_safetensors(f'{adapter_path}/adapters.safetensors',
                    dict(tree_flatten(model.trainable_parameters())))
```

After training, we save only the trainable parameters (which are the LoRA weights) in the safetensors format.

## Conclusion

LoRA training offers an efficient approach to adapting large language models like Phi-3 to specific tasks with minimal additional parameters. This tutorial provides an overview of LoRA and its implementation in MLX, highlighting key components and considerations. While simplified, it serves as a starting point for integrating LoRA into MLX-based model training pipelines.

In upcoming tutorials, we'll explore practical ways to extend and apply language models like Phi-3. We'll delve into topics such as implementing agent classes and toolchain systems, which allow for creating flexible AI workflows and chaining together different operations. These extensions will showcase how to build more versatile and powerful applications on top of the core language model capabilities we've discussed so far.