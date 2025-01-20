# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'dkst'
copyright = '2024, Jakob Lederer'
author = 'Jakob Lederer'
from pathlib import Path
release = Path('../../VERSION').read_text().strip()  # Fetch version dynamically


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',    # Automatically extract documentation from docstrings
    'sphinx.ext.napoleon',   # Support for Google-style docstrings
    'sphinx.ext.viewcode',   # Add links to the source code
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme' # 'alabaster'
html_static_path = ['_static']


import os
import sys

sys.path.insert(0, os.path.abspath('../../dkst'))   # Correct the path(s) to the project
sys.path.insert(0, os.path.abspath('../../dkst/utils'))








