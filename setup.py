"""
Setup configuration for Coherence-Aware AI Framework (CAAF)
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements
requirements = (this_directory / "requirements.txt").read_text().splitlines()

setup(
    name="coherence-aware-ai",
    version="0.1.0",
    author="CAAF Team",
    author_email="",
    description="A drop-in coherence measurement and response optimization system for LLMs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/GreatPyreneseDad/CAAF",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.18.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "dashboard": [
            "streamlit>=1.28.0",
            "plotly>=5.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "caaf-api=src.api:main",
            "caaf-demo=examples.basic_usage:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.json", "*.yaml", "*.yml"],
    },
)