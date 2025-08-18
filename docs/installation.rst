Installation Guide
==================

This guide covers different ways to install EvoAug2 and its dependencies.

Prerequisites
------------

* **Python**: 3.8 or higher
* **PyTorch**: 1.9.0 or higher
* **CUDA**: Optional, for GPU acceleration (CUDA 10.2 or higher)

Installation Methods
-------------------

PyPI Installation (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The easiest way to install EvoAug2 is through PyPI:

.. code-block:: bash

   pip install evoaug2

This will install the core package with minimal dependencies.

Installation with Extras
~~~~~~~~~~~~~~~~~~~~~~~~

EvoAug2 provides several installation options with additional dependencies:

**Development dependencies** (for contributing to EvoAug2):

.. code-block:: bash

   pip install evoaug2[dev]

**Documentation dependencies** (for building docs):

.. code-block:: bash

   pip install evoaug2[docs]

**Full installation** (includes all optional dependencies):

.. code-block:: bash

   pip install evoaug2[full]

**Examples and visualization** (for running examples):

.. code-block:: bash

   pip install evoaug2[examples]

From Source
~~~~~~~~~~

To install from the latest development version:

.. code-block:: bash

   git clone https://github.com/evoaug2/evoaug2.git
   cd evoaug2
   pip install -e .

For development installation with all dependencies:

.. code-block:: bash

   pip install -e ".[dev,docs,examples]"

Conda Installation
~~~~~~~~~~~~~~~~~

EvoAug2 is available through conda-forge:

.. code-block:: bash

   conda install -c conda-forge evoaug2

Docker Installation
~~~~~~~~~~~~~~~~~~

A Docker image is available for easy deployment:

.. code-block:: bash

   docker pull evoaug2/evoaug2:latest
   docker run -it evoaug2/evoaug2:latest

Dependencies
-----------

Core Dependencies
~~~~~~~~~~~~~~~~

* **torch** >= 1.9.0: PyTorch deep learning framework
* **torchvision** >= 0.10.0: Computer vision utilities for PyTorch
* **pytorch-lightning** >= 1.5.0: High-level PyTorch training framework
* **numpy** >= 1.20.0: Numerical computing library
* **scipy** >= 1.7.0: Scientific computing library
* **scikit-learn** >= 1.0.0: Machine learning library
* **h5py** >= 3.1.0: HDF5 file format support

Optional Dependencies
~~~~~~~~~~~~~~~~~~~~

* **matplotlib** >= 3.3.0: Plotting library
* **seaborn** >= 0.11.0: Statistical data visualization
* **jupyter** >= 1.0.0: Jupyter notebook support
* **ipywidgets** >= 7.6.0: Interactive widgets for Jupyter

Development Dependencies
~~~~~~~~~~~~~~~~~~~~~~~

* **pytest** >= 6.0.0: Testing framework
* **pytest-cov** >= 2.0.0: Coverage testing
* **black** >= 21.0.0: Code formatter
* **isort** >= 5.0.0: Import sorter
* **flake8** >= 3.8.0: Linting tool
* **mypy** >= 0.800.0: Type checking
* **pre-commit** >= 2.15.0: Git hooks

Documentation Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~

* **sphinx** >= 4.0.0: Documentation generator
* **sphinx-rtd-theme** >= 1.0.0: Read the Docs theme
* **sphinx-autodoc-typehints** >= 1.12.0: Type hints support
* **myst-parser** >= 0.15.0: Markdown parser
* **nbsphinx** >= 0.8.0: Jupyter notebook support

Verification
-----------

After installation, verify that EvoAug2 is working correctly:

.. code-block:: python

   import evoaug
   print(evoaug.__version__)
   
   # Test basic functionality
   from evoaug.augment import RandomMutation
   from evoaug.evoaug import RobustLoader
   print("EvoAug2 installed successfully!")

GPU Support
----------

EvoAug2 automatically detects and uses GPU acceleration if available:

.. code-block:: python

   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   if torch.cuda.is_available():
       print(f"CUDA version: {torch.version.cuda}")
       print(f"GPU device: {torch.cuda.get_device_name()}")

To install PyTorch with CUDA support, visit the `PyTorch installation page <https://pytorch.org/get-started/locally/>`_.

Troubleshooting
--------------

Common Issues
~~~~~~~~~~~~

**Import Error: No module named 'torch'**

.. code-block:: bash

   pip install torch torchvision

**CUDA not found**

Install PyTorch with CUDA support:

.. code-block:: bash

   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

**Version conflicts**

Use a virtual environment:

.. code-block:: bash

   python -m venv evoaug2_env
   source evoaug2_env/bin/activate  # On Windows: evoaug2_env\Scripts\activate
   pip install evoaug2

**Permission errors**

Use user installation:

.. code-block:: bash

   pip install --user evoaug2

Getting Help
-----------

If you encounter installation issues:

1. Check the `GitHub issues <https://github.com/evoaug2/evoaug2/issues>`_
2. Review the `troubleshooting guide <https://evoaug2.readthedocs.io/en/latest/troubleshooting.html>`_
3. Ask questions on `GitHub Discussions <https://github.com/evoaug2/evoaug2/discussions>`_
4. Report bugs with detailed system information

System Information
~~~~~~~~~~~~~~~~~

When reporting issues, include:

.. code-block:: bash

   python --version
   pip list | grep -E "(torch|evoaug|numpy|scipy)"
   nvidia-smi  # If using GPU 