.. phi-3-vision-mlx documentation master file

Welcome to Phi-3-MLX's documentation!
============================================

Phi-3-MLX is a versatile AI framework that leverages both the Phi-3-Vision multimodal model and the recently updated Phi-3-Mini-128K language model, optimized for Apple Silicon using the MLX framework.

`View the project on GitHub <https://github.com/JosefAlbers/Phi-3-Vision-MLX>`_

Features
--------

- Support for the newly updated Phi-3-Mini-128K (language-only) model
- Integration with Phi-3-Vision (multimodal) model
- Optimized performance on Apple Silicon using MLX
- Batched generation for processing multiple prompts
- Flexible agent system for various AI tasks
- Custom toolchains for specialized workflows
- Model quantization for improved efficiency
- LoRA fine-tuning capabilities
- API integration for extended functionality (e.g., image generation, text-to-speech)

Usage
-----
.. toctree::
   :maxdepth: 2

   install
   generate
   train
   agent
   toolchain
   benchmark

API Reference
-------------

.. toctree::
   :maxdepth: 2

   module

License
-------

This project is licensed under the MIT License.

Citation
--------

.. image:: https://zenodo.org/badge/806709541.svg
   :target: https://zenodo.org/doi/10.5281/zenodo.11403221
   :alt: DOI
