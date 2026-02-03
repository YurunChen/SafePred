"""
Setup script for SafePred package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="safepred",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Safety-TS-LMA: Tree Search for Language Model Agents with Safety Awareness",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/safepred",
    packages=find_packages(exclude=['tests', 'examples', 'docs']),
    package_data={
        'SafePred': ['config/config.yaml'],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        # Add other dependencies as needed
        # "torch>=1.9.0",  # If using PyTorch models
        # "transformers>=4.20.0",  # If using HuggingFace models
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
    },
)

