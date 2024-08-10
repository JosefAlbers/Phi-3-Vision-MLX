# Porting Phi-3-Vision to MLX: A Python Hobbyist's Journey into Advanced AI on Apple Silicon

## Introduction:

Welcome to a series on optimizing cutting-edge AI models for Apple Silicon! Over the next few weeks, we'll dive deep into the process of porting Phi-3-Vision, a powerful and compact vision-language model, from Hugging Face to MLX.

This series is designed for AI enthusiasts, developers, and researchers interested in running advanced models efficiently on Mac devices. For those eager to get started, you can find the MLX ports of both Phi-3-Mini-128K and Phi-3-Vision in my GitHub repository: https://github.com/JosefAlbers/Phi-3-Vision-MLX

## Why Phi-3-Vision?

When Microsoft Research released Phi-3, I was immediately intrigued. Despite its relatively small size of 3.8 billion parameters, it was performing on par with or even surpassing models with 7 billion parameters. This efficiency was impressive and hinted at the potential for running sophisticated AI models on consumer-grade hardware.

![Alt Text](tutorial_part0_phi3.png)

The subsequent release of Phi-3-Vision further piqued my interest. As an owner of a Mac Studio and a Python hobbyist, I saw an exciting opportunity to bring this capable vision-language model to Apple Silicon. While llama.cpp was a popular option for running large language models on Mac, its C++ codebase was way beyond my skill level, so I looked for a more accessible option. This led me to MLX, Apple's machine learning framework that not only offered a Python-friendly environment but also promised better performance than llama.cpp on Apple Silicon.

![Alt Text](tutorial_part0_phi3_v.png)

What made this journey even more exciting was that it marked my first foray into contributing to open source projects. As I worked on porting Phi-3-Vision, I found myself making my first pull requests to repositories like "mlx-examples" and "mlx-vlm". This experience was an invaluable learning experience that helped me gain a better understanding of the MLX framework and how to apply it to real-world projects. This experience also connected me with the broader AI development community.

## Useful Resources:

Before we dive into the series, I want to highlight some excellent resources that have been invaluable in my journey:

1. MLX Examples (https://github.com/ml-explore/mlx-examples): This official repository from the MLX team at Apple is a treasure trove of examples and tutorials that showcase the capabilities of the MLX framework. With a wide range of standalone examples, from basic MNIST to advanced language models and image generation, this repository is an excellent starting point for anyone looking to learn MLX. The quality and depth of the examples are truly impressive, and they demonstrate the team's commitment to making MLX accessible to developers of all levels.

    <img src="https://raw.githubusercontent.com/JosefAlbers/Phi-3-Vision-MLX/main/assets/tutorial_part0_awni.png">

    I also want to give a special shoutout to awni, the repo owner, who was incredibly kind and patient with me when I made my first-ever pull request to this repository. Despite my lack of experience with Git and GitHub, awni guided me through the process and helped me navigate the precommit hooks and other nuances of the repository. Their patience and willingness to help a newcomer like me was truly appreciated, and I'm grateful for the opportunity to have contributed to this repository. If you're new to MLX or Git, I highly recommend checking out this repository and reaching out to awni - they're a great resource and a pleasure to work with!

2. MLX-VLM (https://github.com/Blaizzy/mlx-vlm): A package specifically for running Vision Language Models on Mac using MLX. This repository was particularly helpful in understanding how to handle multimodal inputs, and I found the well-organized and well-written code to be incredibly valuable in learning not only Vision Language Models (VLMs) but also the MLX framework in general. The codebase is a great example of how to structure and implement complex AI models using MLX, making it an excellent resource for anyone looking to learn from experienced developers and improve their own MLX skills.

    <img src="https://raw.githubusercontent.com/JosefAlbers/Phi-3-Vision-MLX/main/assets/tutorial_part0_canuma.png">

    For those interested in other models, Prince Canuma has an excellent tutorial on running Google's Gemma 2 locally on Mac using MLX: https://www.youtube.com/watch?v=CKznaU1HpVQ

3. Hugging Face (https://huggingface.co/): A popular platform for natural language processing (NLP) and computer vision tasks, providing a vast range of pre-trained models, datasets, and tools. Hugging Face’s Transformers library is particularly useful for working with transformer-based models like Phi-3-Vision. Their documentation and community support are also top-notch, making it an excellent resource for anyone looking to learn more about NLP and computer vision.

These resources provide a great foundation for anyone looking to explore MLX and run advanced AI models on Apple Silicon.

## What to Expect in This Series:

### 1. MLX vs. Hugging Face: A Code Comparison

We'll start by comparing the original Hugging Face implementation with our MLX port, highlighting key differences in syntax and how MLX leverages Apple Silicon's unified memory architecture.

### 2. Implementing SuRoPE for 128K Context

We'll explore the Su-scaled Rotary Position Embedding (SuRoPE) implementation that enables Phi-3-Vision to handle impressive 128K token contexts.

### 3. Optimizing Text Generation in MLX: From Batching to Advanced Techniques

Learn how to implement efficient batch text generation, an essential feature for many real-world applications. We'll also cover custom KV-Cache implementation and other text generation optimizations.

### 4. LoRA Fine-tuning and Evaluation on MLX

Discover how to perform Low-Rank Adaptation (LoRA) training, enabling efficient fine-tuning of Phi-3-Vision on custom datasets.

### 5. Building a Versatile AI Agent

In our finale, we'll create a multi-modal AI agent showcasing Phi-3-Vision's capabilities in handling both text and visual inputs.

## Why This Series Matters:

Phi-3-Vision represents a significant advancement in compact, high-performing vision-language models. By porting it to MLX, we're making it more accessible and efficient for a wide range of applications on Apple Silicon devices. This project demonstrates the potential of running advanced AI models on consumer-grade hardware, specifically Apple Silicon Macs.

### Throughout this series, we'll highlight:
- Performance gains on Apple Silicon
- Challenges in porting and how to overcome them
- The process of contributing to open source AI projects
- Practical applications of the optimized model

### Who This Series Is For:
- AI enthusiasts and hobbyists looking to dive deeper into model optimization
- Researchers exploring efficient AI on consumer hardware
- Mac users eager to leverage their devices for AI tasks
- Anyone curious about the intersection of AI and Apple Silicon
- Beginners interested in contributing to open source AI projects

## Stay Tuned!

Our journey into optimizing Phi-3-Vision for MLX promises to be full of insights, challenges, and exciting breakthroughs. Whether you're a fellow hobbyist looking to run advanced AI models on your Mac or simply curious about the future of AI on consumer devices, this series has something for you.

Join me on this adventure in AI optimization, and let's unlock the full potential of Phi-3-Vision on Apple Silicon together!