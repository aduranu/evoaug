"""Setup script for the EvoAug package.

The canonical metadata lives in pyproject.toml; this file is kept as a
thin compatibility shim so ``pip install -e .`` continues to work on
older toolchains.
"""

from setuptools import setup, find_packages
import os


def _read_long_description():
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return "Evolution-inspired data augmentations for genomic sequence models."


setup(
    name="evoaug",
    version="2.0.0",
    author="Peter K. Koo",
    author_email="koo@cshl.edu",
    description=(
        "Evolution-inspired data augmentations for genomic sequence models "
        "(PyTorch DataLoader API)."
    ),
    long_description=_read_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/p-koo/evoaug",
    project_urls={
        "Documentation": "https://evoaug.readthedocs.io/",
        "Bug Tracker": "https://github.com/p-koo/evoaug/issues",
        "Source Code": "https://github.com/p-koo/evoaug",
    },
    packages=find_packages(include=["evoaug", "evoaug.*"]),
    license="MIT",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "lightning>=2.0.0",
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "h5py>=3.1.0",
        "scikit-learn>=1.0.0",
    ],
    extras_require={
        "examples": ["matplotlib>=3.3", "seaborn>=0.11", "jupyter>=1.0"],
        "docs": [
            "sphinx>=7.0",
            "sphinx-rtd-theme>=1.0",
            "myst-parser>=1.0",
            "sphinx-autodoc-typehints>=1.0",
        ],
        "full": [
            "matplotlib>=3.3",
            "seaborn>=0.11",
            "jupyter>=1.0",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
