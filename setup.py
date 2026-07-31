from setuptools import setup#, find_packages

setup(
    name='phi-3-vision-mlx',
    url='https://github.com/JosefAlbers/Phi-3-Vision-MLX',
    py_modules=['phi_3_vision_mlx', 'gte', 'phi', 'api'],
    # packages=find_packages(),
    version='0.1.7',
    readme="README.md",
    author_email="albersj66@gmail.com",
    description="Phi-3-Vision on Apple silicon with MLX",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Josef Albers",
    license="MIT",
    python_requires=">=3.12.3",
    install_requires=[
        "mlx==0.29.3",
        "transformers==5.12.1",
        "numpy==1.26.4",
        "matplotlib==3.9.0",
        "datasets==2.19.1",
        "gradio==6.20.0",
        "requests==2.32.3",
    ],
    entry_points={
        "console_scripts": [
            "phi3v = phi_3_vision_mlx:chat_ui",
        ],
    },
    project_urls={
        "Documentation": "https://josefalbers.github.io/Phi-3-Vision-MLX/"
    },
)
