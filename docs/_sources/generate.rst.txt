Text Generation
===============

Visual Question Answering
-------------------------

.. code-block:: python

    generate('What is shown in this image?', 'https://collectionapi.metmuseum.org/api/collection/v1/iiif/344291/725918/main-image')

Batch Text Generation
---------------------

.. code-block:: python

    prompts = [
        "Explain the key concepts of quantum computing and provide a Rust code example demonstrating quantum superposition.",
        "Write a poem about the first snowfall of the year.",
        "Summarize the major events of the French Revolution.",
        "Describe a bustling alien marketplace on a distant planet with unique goods and creatures."
        "Implement a basic encryption algorithm in Python.",
    ]

    # Generate responses using Phi-3-Vision (multimodal model)
    generate(prompts, max_tokens=100)

    # Generate responses using Phi-3-Mini-128K (language-only model)
    generate(prompts, max_tokens=100, blind_model=True)

Structured Generation Using Constrained Decoding
------------------------------------------------

.. code-block:: python

    from phi_3_vision_mlx import constrain

    # Define the prompt that instructs the model on the task to perform.
    prompt = "Write a Python function to calculate the Fibonacci sequence up to a given number n."

    # Define constraints to guide the model in generating an appropriate response.
    # Each constraint tuple consists of (num_tokens, constraint_string).
    constraints = [(100, "\n```python\n"), (100, " return "), (200, "\n```")]

    # Apply constrained decoding using the 'constrain' function from phi_3_vision_mlx.
    constrain(prompt, constraints)

Model and Cache Quantization
----------------------------

.. code-block:: python

    # Model quantization
    generate("Describe the water cycle.", quantize_model=True)

    # Cache quantization
    generate("Explain quantum computing.", quantize_cache=True)