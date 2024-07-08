LoRA Fine-tuning
================

Training a LoRA Adapter
-----------------------

.. code-block:: python

    from phi_3_vision_mlx import train_lora

    train_lora(
        lora_layers=5,  # Number of layers to apply LoRA
        lora_rank=16,   # Rank of the LoRA adaptation
        epochs=10,      # Number of training epochs
        lr=1e-4,        # Learning rate
        warmup=0.5,     # Fraction of steps for learning rate warmup
        dataset_path="JosefAlbers/akemiH_MedQA_Reason"
    )

Generating Text with LoRA
-------------------------

.. code-block:: python

    generate("Describe the potential applications of CRISPR gene editing in medicine.",
        blind_model=True,
        quantize_model=True,
        use_adapter=True)

Comparing LoRA Adapters
-----------------------

.. code-block:: python

    from phi_3_vision_mlx import test_lora

    # Test model without LoRA adapter
    test_lora(adapter_path=None)
    # Output score: 0.6 (6/10)

    # Test model with the trained LoRA adapter (using default path)
    test_lora(adapter_path=True)
    # Output score: 0.8 (8/10)

    # Test model with a specific LoRA adapter path
    test_lora(adapter_path="/path/to/your/lora/adapter")