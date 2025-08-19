Installation
============

EvoAug2 can be installed via pip from PyPI or directly from the GitHub repository. The package requires Python 3.8 or higher and PyTorch 1.9.0 or higher.

Installation from PyPI
----------------------

The recommended way to install EvoAug2 is through PyPI:

.. code-block:: bash

   pip install evoaug2

This will install the core package with all required dependencies.

Installation with Optional Dependencies
-------------------------------------

EvoAug2 provides several optional dependency groups for different use cases:

**Full installation (recommended for most users):**

.. code-block:: bash

   pip install evoaug2[full]

This includes all dependencies for training, evaluation, and visualization.

**Development installation:**

.. code-block:: bash

   pip install evoaug2[dev]

This includes development tools like testing, linting, and code formatting.

**Documentation installation:**

.. code-block:: bash

   pip install evoaug2[docs]

This includes Sphinx and related tools for building documentation.

**Examples installation:**

.. code-block:: bash

   pip install evoaug2[examples]

This includes Jupyter and visualization libraries for running examples.

Installation from Source
-----------------------

To install the latest development version from source:

.. code-block:: bash

   git clone https://github.com/aduranu/evoaug.git
   cd evoaug
   pip install -e .

For development installation with all dependencies:

.. code-block:: bash

   pip install -e .[dev,full]

System Requirements
------------------

**Operating System:**
- Linux (Ubuntu 18.04+, CentOS 7+, etc.)
- macOS 10.14+
- Windows 10+ (with WSL2 recommended)

**Python:**
- Python 3.8 or higher
- pip 20.0 or higher

**Hardware:**
- CPU: Any modern multi-core processor
- RAM: 8GB minimum, 16GB+ recommended
- GPU: NVIDIA GPU with CUDA support (optional but recommended for training)
- Storage: 2GB+ free space

**GPU Support:**
- CUDA 11.0 or higher (for PyTorch GPU support)
- cuDNN 8.0 or higher
- Compatible NVIDIA drivers

Dependencies
------------

**Core Dependencies:**
- PyTorch >= 1.9.0
- PyTorch Lightning >= 1.5.0
- NumPy >= 1.20.0
- SciPy >= 1.7.0
- h5py >= 3.1.0
- scikit-learn >= 1.0.0

**Optional Dependencies:**
- matplotlib >= 3.3.0 (for plotting)
- seaborn >= 0.11.0 (for enhanced plotting)
- Jupyter >= 1.0.0 (for notebooks)
- ipywidgets >= 7.6.0 (for interactive widgets)

**Development Dependencies:**
- pytest >= 6.0 (for testing)
- black >= 21.0 (for code formatting)
- isort >= 5.0 (for import sorting)
- flake8 >= 3.8 (for linting)
- mypy >= 0.800 (for type checking)

Verification
------------

After installation, verify that EvoAug2 is working correctly:

.. code-block:: python

   import evoaug
   print(evoaug.__version__)  # Should print 2.0.3

   # Test basic functionality
   from evoaug.augment import RandomMutation
   augmenter = RandomMutation(mut_frac=0.1)
   print("EvoAug2 installation successful!")

Troubleshooting
---------------

**Common Installation Issues:**

1. **PyTorch Installation Problems:**
   - Visit https://pytorch.org/ for platform-specific installation instructions
   - Ensure CUDA version compatibility if using GPU

2. **Permission Errors:**
   - Use virtual environments: ``python -m venv evoaug_env && source evoaug_env/bin/activate``
   - Or use ``pip install --user evoaug2`` for user installation

3. **Missing Dependencies:**
   - Install system-level dependencies (e.g., ``apt-get install python3-dev`` on Ubuntu)
   - Ensure pip is up to date: ``pip install --upgrade pip``

4. **GPU Issues:**
   - Verify CUDA installation: ``nvidia-smi``
   - Check PyTorch CUDA support: ``python -c "import torch; print(torch.cuda.is_available())"``

**Getting Help:**

- Check the `GitHub Issues <https://github.com/aduranu/evoaug/issues>`_ page
- Review the `Troubleshooting Guide <advanced/troubleshooting.html>`_
- Contact the maintainers through GitHub

Next Steps
----------

After successful installation:

1. Read the :doc:`quickstart` guide to get started
2. Explore the :doc:`examples` section for usage examples
3. Check the :doc:`user_guide/overview` for detailed usage instructions
4. Run the example scripts to verify functionality 