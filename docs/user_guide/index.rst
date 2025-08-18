User Guide
==========

This section provides comprehensive information about using EvoAug2.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   augmentations
   dataloader
   lightning_integration
   best_practices
   troubleshooting

Overview
--------

EvoAug2 is designed to provide evolution-inspired data augmentation for genomic sequences while maintaining sequence length consistency. This guide covers all aspects of using the library effectively.

Key Concepts
-----------

* **Length Preservation**: All augmentations maintain the input sequence length L
* **Batch Consistency**: Consistent shapes across all sequences in a batch
* **Runtime Control**: Enable/disable augmentations as needed
* **PyTorch Integration**: Works seamlessly with PyTorch and PyTorch Lightning

Getting Started
--------------

If you haven't already, start with the :doc:`../quickstart` guide to get up and running quickly.

For detailed API information, see the :doc:`../api_reference/index`.

Examples
--------

See the :doc:`../examples/index` section for comprehensive examples and use cases. 