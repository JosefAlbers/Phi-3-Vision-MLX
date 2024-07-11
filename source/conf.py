project = 'phi-3-vision-mlx'
copyright = '2024, Josef Albers'
author = 'Josef Albers'
release = '0.0.8-alpha'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
]

templates_path = ['_templates']
exclude_patterns = []

autodoc_default_options = {
    'ignore-module-all': True,
    'undoc-members': False,
    'members': None,
    'special-members': None,
}

# napoleon_use_rtype = False  # Ensure this is set to False to handle Markdown-style code blocks
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_param = False
napoleon_use_ivar = False

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']

html_theme_options = {
    'github_user': 'JosefAlbers',
    'github_repo': 'Phi-3-Vision-MLX',
    'github_button': True,
    'github_type': 'star',
    'github_banner': True,
    'description': 'A versatile AI framework leveraging Phi-3-Vision and Phi-3-Mini-128K models.',
    'fixed_sidebar': True,
    'body_text_align': 'left',
    'font_size': '16px',
}
