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

The ``constrain()`` function allows for structured generation, which can be useful for tasks like code generation, function calling, chain-of-thought prompting, or multiple-choice question answering.

.. code-block:: python

    from phi_3_vision_mlx import constrain

    # Define the prompt
    prompt = "Write a Python function to calculate the Fibonacci sequence up to a given number n."

    # Define constraints
    constraints = [
        (100, "\n```python\n"), # Start of code block
        (100, " return "),      # Ensure a return statement
        (200, "\n```")          # End of code block
    ]

    # Apply constrained decoding using the 'constrain' function from phi_3_vision_mlx
    constrain(prompt, constraints)

The ``constrain()`` function can also guide the model to provide reasoning before concluding with an answer. This approach can be especially helpful for multiple-choice questions, such as those in the Massive Multitask Language Understanding (MMLU) benchmark, where the model's thought process is as crucial as its final selection.

.. code-block:: python

    prompts = [
        "A 20-year-old woman presents with menorrhagia for the past several years. She says that her menses “have always been heavy”, and she has experienced easy bruising for as long as she can remember. Family history is significant for her mother, who had similar problems with bruising easily. The patient's vital signs include: heart rate 98/min, respiratory rate 14/min, temperature 36.1°C (96.9°F), and blood pressure 110/87 mm Hg. Physical examination is unremarkable. Laboratory tests show the following: platelet count 200,000/mm3, PT 12 seconds, and PTT 43 seconds. Which of the following is the most likely cause of this patient’s symptoms? A: Factor V Leiden B: Hemophilia A C: Lupus anticoagulant D: Protein C deficiency E: Von Willebrand disease",
        "A 25-year-old primigravida presents to her physician for a routine prenatal visit. She is at 34 weeks gestation, as confirmed by an ultrasound examination. She has no complaints, but notes that the new shoes she bought 2 weeks ago do not fit anymore. The course of her pregnancy has been uneventful and she has been compliant with the recommended prenatal care. Her medical history is unremarkable. She has a 15-pound weight gain since the last visit 3 weeks ago. Her vital signs are as follows: blood pressure, 148/90 mm Hg; heart rate, 88/min; respiratory rate, 16/min; and temperature, 36.6℃ (97.9℉). The blood pressure on repeat assessment 4 hours later is 151/90 mm Hg. The fetal heart rate is 151/min. The physical examination is significant for 2+ pitting edema of the lower extremity. Which of the following tests o should confirm the probable condition of this patient? A: Bilirubin assessment B: Coagulation studies C: Hematocrit assessment D: Leukocyte count with differential E: 24-hour urine protein"]

    constrain(prompts, constraints=[(30, ' The correct answer is'), (10, 'X.')], blind_model=True, quantize_model=True)


Model and Cache Quantization
----------------------------

.. code-block:: python

    # Model quantization
    generate("Describe the water cycle.", quantize_model=True)

    # Cache quantization
    generate("Explain quantum computing.", quantize_cache=True)