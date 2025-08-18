"""
Setup script for EvoAug2 package.
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "EvoAug2: Evolution-inspired sequence augmentations as a DataLoader"

# Read requirements
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return [
        'torch>=1.12.0',
        'numpy>=1.21.0',
        'lightning>=2.0.0',
    ]

setup(
    name="evoaug",
    version="2.0.0",
    author="EvoAug Team",
    author_email="koo@cshl.edu",
    description="Evolution-inspired sequence augmentations as a DataLoader",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/evoaug/evoaug",
    project_urls={
        "Bug Reports": "https://github.com/evoaug/evoaug/issues",
        "Source": "https://github.com/evoaug/evoaug",
        "Documentation": "https://evoaug.readthedocs.io/",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
            "sphinx-autodoc-typehints>=1.12",
        ],
    },
    keywords=[
        "genomics",
        "deep-learning",
        "data-augmentation",
        "pytorch",
        "bioinformatics",
        "sequence-analysis",
        "evolution",
    ],
    license="MIT",
    zip_safe=False,
    include_package_data=True,
)
