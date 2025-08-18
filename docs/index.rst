EvoAug2 Documentation
======================

**Evolution-Inspired Data Augmentation for Genomic Sequences**

EvoAug2 is a PyTorch-based library that provides evolution-inspired data augmentation techniques for genomic sequence data. It implements various augmentation strategies that mimic biological mutations and evolutionary processes while maintaining sequence length consistency.

.. image:: https://img.shields.io/badge/Python-3.8+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python 3.8+

.. image:: https://img.shields.io/badge/PyTorch-1.9+-red.svg
   :target: https://pytorch.org/
   :alt: PyTorch 1.9+

.. image:: https://img.shields.io/badge/License-MIT-green.svg
   :target: https://opensource.org/licenses/MIT
   :alt: MIT License

.. image:: https://img.shields.io/badge/Status-Beta-orange.svg
   :target: https://github.com/evoaug2/evoaug2
   :alt: Beta Status

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   user_guide/index
   api_reference/index
   examples/index
   contributing
   changelog

Features
--------

* **Evolution-Inspired Augmentations**: Implements biological mutation processes like deletion, insertion, translocation, inversion, and point mutations
* **Length Preservation**: All augmentations maintain the original sequence length L for consistent batch shapes
* **PyTorch Integration**: Seamlessly integrates with PyTorch and PyTorch Lightning workflows
* **Flexible Configuration**: Configurable augmentation parameters and combination strategies
* **Real-time Processing**: Augmentations applied on-the-fly during training
* **Research Ready**: Designed for genomic deep learning research and applications

Quick Start
-----------

Install EvoAug2:

.. code-block:: bash

   pip install evoaug2

Basic usage:

.. code-block:: python

   from evoaug.augment import RandomMutation, RandomDeletion
   from evoaug.evoaug import RobustLoader
   
   # Create augmentations
   augmentations = [
       RandomMutation(mut_frac=0.05),
       RandomDeletion(delete_min=0, delete_max=20)
   ]
   
   # Use with RobustLoader
   loader = RobustLoader(
       base_dataset=your_dataset,
       augment_list=augmentations,
       max_augs_per_seq=1
   )
   
   # Train your model
   for batch in loader:
       x, y = batch
       # x has shape (N, A, L) - same as input
       y_hat = model(x)
       loss = criterion(y_hat, y)

What's New
----------

* **Version 2.0.0**: Complete rewrite with PyTorch Lightning integration
* **New Architecture**: Replaced model wrapper with lightweight DataLoader approach
* **Enhanced Augmentations**: Improved length preservation and batch consistency
* **Better Documentation**: Comprehensive API reference and examples
* **Performance Improvements**: Optimized augmentation algorithms

Installation
-----------

.. code-block:: bash

   # Install from PyPI
   pip install evoaug2
   
   # Install with development dependencies
   pip install evoaug2[dev]
   
   # Install with documentation dependencies
   pip install evoaug2[docs]
   
   # Install with all optional dependencies
   pip install evoaug2[full]

Requirements
-----------

* Python 3.8+
* PyTorch 1.9+
* PyTorch Lightning 1.5+
* NumPy 1.20+
* SciPy 1.7+
* scikit-learn 1.0+
* h5py 3.1+

Citation
--------

If you use EvoAug2 in your research, please cite:

.. code-block:: text

   @software{evoaug2,
     title={EvoAug2: Evolution-Inspired Data Augmentation for Genomic Sequences},
     author={EvoAug2 Development Team},
     year={2024},
     url={https://github.com/evoaug2/evoaug2}
   }

Support
-------

* **Documentation**: https://evoaug2.readthedocs.io/
* **GitHub**: https://github.com/evoaug2/evoaug2
* **Issues**: https://github.com/evoaug2/evoaug2/issues
* **Discussions**: https://github.com/evoaug2/evoaug2/discussions

License
-------

This project is licensed under the MIT License - see the `LICENSE <https://github.com/evoaug2/evoaug2/blob/main/LICENSE>`_ file for details.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search` 