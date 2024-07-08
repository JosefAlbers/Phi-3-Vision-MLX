Agent Interactions
==================

Multi-turn Conversation
-----------------------

.. code-block:: python

    from phi_3_vision_mlx import Agent

    # Create an instance of the Agent
    agent = Agent()

    # First interaction: Analyze an image
    agent('Analyze this image and describe the architectural style:', 'https://images.metmuseum.org/CRDImages/rl/original/DP-19531-075.jpg')

    # Second interaction: Follow-up question
    agent('What historical period does this architecture likely belong to?')

    # End the conversation
    agent.end()

Generative Feedback Loop
------------------------

.. code-block:: python

    # Ask the agent to generate and execute code to create a plot
    agent('Plot a Lissajous Curve.')

    # Ask the agent to modify the generated code and create a new plot
    agent('Modify the code to plot 3:4 frequency')
    agent.end()

External API Tool Use
---------------------

.. code-block:: python

    # Request the agent to generate an image
    agent('Draw "A perfectly red apple, 32k HDR, studio lighting"')
    agent.end()

    # Request the agent to convert text to speech
    agent('Speak "People say nothing is impossible, but I do nothing every day."')
    agent.end()