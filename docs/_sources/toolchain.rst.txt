Custom Toolchains
=================

Example 1: In-Context Learning Agent
------------------------------------

.. code-block:: python

    from phi_3_vision_mlx import _load_text

    # Create a custom tool named 'add_text'
    def add_text(prompt):
        prompt, path = prompt.split('@')
        return f'{_load_text(path)}\n<|end|>\n<|user|>{prompt}'

    # Define the toolchain as a string
    toolchain = """
        prompt = add_text(prompt)
        responses = generate(prompt, images)
        """

    # Create an Agent instance with the custom toolchain
    agent = Agent(toolchain, early_stop=100)

    # Run the agent
    agent('How to inspect API endpoints? @https://raw.githubusercontent.com/gradio-app/gradio/main/guides/08_gradio-clients-and-lite/01_getting-started-with-the-python-client.md')

Example 2: Retrieval Augmented Coding Agent
-------------------------------------------

.. code-block:: python

    from phi_3_vision_mlx import VDB
    import datasets

    # Simulate user input
    user_input = 'Comparison of Sortino Ratio for Bitcoin and Ethereum.'

    # Create a custom RAG tool
    def rag(prompt, repo_id="JosefAlbers/sharegpt_python_mlx", n_topk=1):
        ds = datasets.load_dataset(repo_id, split='train')
        vdb = VDB(ds)
        context = vdb(prompt, n_topk)[0][0]
        return f'{context}\n<|end|>\n<|user|>Plot: {prompt}'

    # Define the toolchain
    toolchain_plot = """
        prompt = rag(prompt)
        responses = generate(prompt, images)
        files = execute(responses, step)
        """

    # Create an Agent instance with the RAG toolchain
    agent = Agent(toolchain_plot, False)

    # Run the agent with the user input
    _, images = agent(user_input)

Example 3: Multi-Agent Interaction
----------------------------------

.. code-block:: python

    # Continued from Example 2 above
    agent_writer = Agent(early_stop=100)
    agent_writer(f'Write a stock analysis report on: {user_input}', images)
