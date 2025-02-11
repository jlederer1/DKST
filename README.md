# DKST v0.1.0
The documentation for this project is available at: [DKST Documentation](https://jlederer1.github.io/DKST/)

## Project structure 
This package follows PyPI guidelines for structure, metadata, and documentation and adheres to semantic versioning.
It uses Sphinx for automatic documentation generation from docstrings, hosted via GitHub Pages.

```
project-root/
├── .pytest_cache/          # Caching for pytest
├── data/                   # Stores datasets, models, configuration and output
│   ├── config/                 # Configuration files
│   ├── datasets/               # Dataset files or scripts
│   ├── models/                 # Model-related scripts
│   └── output/                 # Output from model training or processing
│
├── dkst/                   # Main package directory
│   ├── __pycache__/            # Compiled Python files
│   ├── utils/                  # Utility functions and scripts
│   │   ├── DKST_utils.py           # Utility functions for DKST
│   │   ├── KST_utils.py            # Utility functions for KST
│   │   ├── __init__.py             # (Package initializer)
│   │   ├── relations.py            # Utilities for handling relations
│   │   └── set_operations.py       # Utilities for handling sets
│   ├── dkst_datasets.py        # Dataset initialization utilities
│   ├── models.py               # Model definitions and helpers
│   └── __init__.py         # (Package initializer)
│
├── dkst-env/               # Virtual environment folder (user-specific)
├── dkst.egg-info/          # Metadata for Python package
├── docs/                   # Documentation generated by Sphinx
│   ├── build/                  # Build artifacts for documentation
│   │   ├── doctrees/               # Intermediate Sphinx files
│   │   └── html/                   # Generated HTML documentation
│   ├── source/                 # Sphinx documentation source files
│   ├── Makefile                # Makefile for building docs (Unix-based)
│   └── make.bat                # Batch file for building docs (Windows)
│
├── examples/               # Example scripts, notebooks
├── tests/                  # Unit tests
│   ├── test_dkst_utils.py      # Tests for DKST utilities
│   ├── test_kst_utils.py       # Tests for KST utilities
│   ├── test_relations.py       # Tests for relations module
│   ├── test_set_operations.py  # Tests for set operations module
│   └── __init__.py         # (Package initializer for relative paths)
├── .gitignore              # Git ignore file
├── README.md               # Project README
├── requirements.txt        # Project dependencies
├── setup.py                # Configuration for packaging and installing the project via setuptools
└── VERSION                 # Contains the current version of the project
```

## Installation and Setup
To install the required dependencies, clone the repository and follow these steps:

1. Ensure you have Python 3 and pip installed (I recommend Python 3.9.13 for this project):
    ```sh
    python3 --version
    pip3 --version
    ```

2. Install the dependencies listed in `requirements.txt`:
    - On macOS and Linux:
        ```sh
        python3 -m venv dkst-env
        source dkst-env/bin/activate
        pip3 install -r requirements.txt
        ```
    - On Windows:
        ```sh
        python -m venv dkst-env
        dkst-env\Scripts\activate
        pip install -r requirements.txt
        ```

3. Install the appropriate version of PyTorch:
    - For macOS with Apple Silicon (M1, M2, etc.) to use Metal for GPU acceleration:
        ```sh
        pip3 install torch==2.1.1 torchvision torchaudio
        ```
    - For systems with CUDA support, check your CUDA version and install the CUDA-enabled version of PyTorch:
        ```sh
        nvcc --version
        pip install torch==2.1.1+cu118 --find-links https://download.pytorch.org/whl/torch_stable.html --no-cache-dirs
        ```
    - For macOS or systems without CUDA support, install the CPU-only version of PyTorch:
        ```sh
        pip3 install torch==2.1.1+cpu
        ```

4. Check correct functionality using pytest:
    - Ensure you have `pytest` installed:
        ```sh
        pip3 install pytest
        ```
    - Install your package in editable mode:
        ```sh
        pip3 install -e .
        ```
    - Run all tests:
        ```sh
        pytest
        ```
    - Or run a specific test (verbose mode and uncaptured output):
        ```
        pytest -k test_Dataset02 -v -s
        ```

## Notes

- For PyTorch with CUDA 11.8 support, the `requirements.txt` file includes a `--find-links` URL to ensure the correct version is installed.




------------------ For collaborators ------------------



## Installing Package in Editable Mode

To install the package in editable mode for development purposes, follow these steps:
1. Ensure you have Python 3 and pip installed:
    ```sh
    python3 --version
    pip3 --version
    ```
2. Create and activate a virtual environment (recommended):
    - On macOS and Linux:
        ```sh
        python3 -m venv myenv
        source myenv/bin/activate
        ```
    - On Windows:
        ```sh
        python -m venv myenv
        myenv\Scripts\activate
        ```
3. Install the package in editable mode:
    ```sh
    pip install -e .
    ```

## Building the Documentation

To build the documentation, follow these steps:

1. Ensure you have Sphinx installed:
    ```sh
    pip3 install sphinx
    ```
2. Navigate to the `docs` directory:
    ```sh
    cd docs
    ```
3. Build the documentation using the Makefile (on Unix-based systems):
    ```sh
    make clean
    make html
    ```
   Or using the make.bat file (on Windows):
    ```sh
    .\make.bat html
    ```

The generated documentation can be accessed via `docs/build/html/index.html`.
Add your content using `reStructuredText` syntax to the .rst files. See the
[reStructuredText documentation](https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html) for details.