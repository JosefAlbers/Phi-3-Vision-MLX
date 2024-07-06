# This file was authored by GitHub user @couillonnade (PR #2)

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from phi_3_vision_mlx import Agent

# Multi-turn VQA
agent = Agent()
agent('What is shown in this image?', 'https://collectionapi.metmuseum.org/api/collection/v1/iiif/344291/725918/main-image')
agent('What is the location?')
agent.end()

# Generative Feedback Loop
agent('Plot a Lissajous Curve.')
agent('Modify the code to plot 3:4 frequency')
agent.end()

# API Tool Use
agent('Draw "A perfectly red apple, 32k HDR, studio lighting"')
agent.end()
agent('Speak "People say nothing is impossible, but I do nothing every day."')
agent.end()

from phi_3_vision_mlx import train_lora, test_lora

# Train
train_lora(lora_layers=5, lora_rank=16, epochs=10, take=10, batch_size=2, lr=1e-4, warmup=.5, dataset_path="JosefAlbers/akemiH_MedQA_Reason")

# Test
test_lora()

from phi_3_vision_mlx import benchmark

# Benchmark
benchmark()
