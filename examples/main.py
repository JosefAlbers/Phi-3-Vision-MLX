''''
This is a simple example of how to use the Phi-3 Vision MLX agent.
The agent can be used for Visual Question Answering, Generative Feedback Loop, API Tool Use, and more.

Author: Josef Albers

More examples here: https://github.com/JosefAlbers/Phi-3-Vision-MLX
'''

from phi_3_vision_mlx import Agent

# Visual Question Answering (VQA)
agent = Agent()
agent('What is shown in this image?', 'https://collectionapi.metmuseum.org/api/collection/v1/iiif/344291/725918/main-image')
agent.end()

# Generative Feedback Loop
# The agent can be used to generate code, execute it, and then modify it based on feedback

agent('Plot a Lissajous Curve.')
agent('Modify the code to plot 3:4 frequency')
agent.end()

# API Tool Use
# You can use the agent to create images or generate speech using API calls

agent('Draw "A perfectly red apple, 32k HDR, studio lighting"')
agent.end()
agent('Speak "People say nothing is impossible, but I do nothing every day."')
agent.end()

