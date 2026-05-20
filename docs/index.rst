EvoAug Documentation
======================

**Evolution-Inspired Data Augmentation for Genomic Sequences - DataLoader Version**

.. image:: https://img.shields.io/pypi/v/evoaug.svg
   :target: https://pypi.org/project/evoaug/
   :alt: PyPI version

.. image:: https://img.shields.io/pypi/pyversions/evoaug.svg
   :target: https://pypi.org/project/evoaug/
   :alt: Python versions

.. image:: https://img.shields.io/badge/License-MIT-blue.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License

.. image:: https://readthedocs.org/projects/evoaug/badge/?version=latest
   :target: https://evoaug.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

Welcome to EvoAug, a Python package that provides evolution-inspired data augmentation techniques for genomic sequence analysis using deep learning. This package implements the two-stage training approach described in the EvoAug paper, combining robust augmentation during training with fine-tuning on original data.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart
   examples

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/evoaug
   api/utils
   api/examples

.. toctree::
   :maxdepth: 2
   :caption: Examples

   examples/lightning_module
   examples/vanilla_pytorch

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. raw:: html

   <div class="admonition note">
   <p class="admonition-title">Note</p>
   <p>This documentation is for EvoAug version 2.0.3. For older versions, please refer to the <a href="https://github.com/p-koo/evoaug/releases">GitHub releases</a>.</p>
   </div>

.. raw:: html

   <div class="admonition tip">
   <p class="admonition-title">Tip</p>
   <p>If you're new to EvoAug, start with the <a href="quickstart.html">Quick Start Guide</a> and then explore the <a href="examples.html">Examples</a> section.</p>
   </div> 