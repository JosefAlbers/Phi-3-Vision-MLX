# Part 5: Implementing Choice Selection with Phi-3-Vision

## Introduction

In this tutorial, we'll explore how to implement a choice selection function for our Phi-3-Vision model. This function constrains the model to pick from a small set of predefined options, making it useful for multiple-choice scenarios.

The full implementation of this tutorial is available at https://github.com/JosefAlbers/Phi-3-Vision-MLX/tree/main/assets/tutorial_5.py

## Understanding Choice Selection

Choice selection is a straightforward concept: we give the model a prompt and a set of choices, then ask it to select the most likely one. This is particularly useful when we have predefined answers and want the model to pick the best one.

## Implementing Choice Selection

Here's our choice selection function:

```python
def choose(prompts, choices='ABCDE'):
    # 1. Prompt Processing
    inputs = batch_process(prompts)
    # 2. Option Encoding
    options = [processor.tokenizer.encode(f' {i}')[-1] for i in choices]
    # 3. Model Prediction
    logits, _ = model(**inputs)
    # 4. Option Selection
    indices = mx.argmax(logits[:, -1, options], axis=-1).tolist()
    # 5. Output Formatting
    output = [choices[i] for i in indices]
    return output
```

Let's break it down:

1. **Prompt Processing**: Process the input prompts using the `batch_process` function.
2. **Option Encoding**: Encode each possible choice as a token ID.
3. **Model Prediction**: Run the model to get logits for the next token.
4. **Option Selection**: Use `argmax` to find the index of the highest logit among our choice options.
5. **Output Formatting**: Map these indices back to our choice letters.

The function takes a list of prompts and a string of choice letters. It returns a list of selected choices.

## Using Choice Selection

Here's an example:

```python
prompts = [
    "What is the largest planet in our solar system? A: Earth B: Mars C: Jupiter D: Saturn",
    "Which element has the chemical symbol 'O'? A: Osmium B: Oxygen C: Gold D: Silver"
]

choose(prompts, choices='ABCD')
# Output: ['C', 'B']
```

In this example, the model correctly selects 'C' (Jupiter) as the largest planet and 'B' (Oxygen) as the element with the chemical symbol 'O'.

## Limitations

The choice selection method is simple and effective for multiple-choice scenarios. It's computationally efficient as it only requires a single forward pass through the model.

However, it's limited to scenarios where we have predefined choices. For more open-ended tasks, we'll need more advanced techniques like constrained beam search, which we'll cover in the next tutorial.

## Conclusion

Choice selection provides a straightforward way to guide our Phi-3-Vision model's output when we have a predefined set of options. This implementation uses the raw logits from the model to make selections, which is computationally efficient and direct.

In our next tutorial, we'll explore constrained beam search, a more advanced technique for guided generation. This will allow us to guide the model's output more flexibly, enabling us to generate new text while following specific constraints.