from setuptools import setup, find_packages
from pathlib import Path

# Read the version from the VERSION file
version = Path("VERSION").read_text().strip()

setup(
    name="dkst",
    version=version,  # Use the dynamic version
    author="Jakob Lederer",
    author_email="jlederer@uos.de",
    description="A package for Deep Knowledge Structure Theory",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/jlederer1/dkst",
    packages=find_packages(),
    install_requires=[ # See requirements.txt | todo 
    "torch",
    "numpy",
    "tqdm",
    "matplotlib",
    "scipy",

],
    extras_require={
        'dev': [
            'sphinx',
            'ipykernel',
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',   # todo
)


