.. phi-3-vision-mlx documentation master file

Welcome to phi-3-vision-mlx's documentation!
============================================

Phi-3-MLX is a versatile AI framework that leverages both the Phi-3-Vision multimodal model and the recently updated Phi-3-Mini-128K language model, optimized for Apple Silicon using the MLX framework.

`View the project on GitHub <https://github.com/JosefAlbers/Phi-3-Vision-MLX>`_

.. toctree::
   :maxdepth: 2
   :caption: Contents

   install
   generate
   train
   agent
   toolchain
   benchmark

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   module

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

Recent Updates: Phi-3 Mini Improvements
---------------------------------------

Microsoft has recently released significant updates to the Phi-3 Mini model, dramatically improving its capabilities:

- Substantially enhanced code understanding in Python, C++, Rust, and TypeScript
- Improved post-training for better-structured output
- Enhanced multi-turn instruction following
- Added support for the `<|system|>` tag
- Improved reasoning and long-context understanding

For detailed benchmark results, please refer to the tables in the README.

License
-------

This project is licensed under the MIT License.

Citation
--------

.. image:: https://zenodo.org/badge/806709541.svg
   :target: https://zenodo.org/doi/10.5281/zenodo.11403221
   :alt: DOI
