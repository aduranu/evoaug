Examples
========

This section provides comprehensive examples of using EvoAug2 in various scenarios.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   basic_usage
   deepstarr_training
   custom_augmentations
   lightning_integration
   performance_optimization

Overview
--------

The examples demonstrate different use cases and best practices for EvoAug2:

* **Basic Usage**: Simple examples to get started
* **DeepSTARR Training**: Complete training pipeline with EvoAug2
* **Custom Augmentations**: How to create your own augmentation classes
* **Lightning Integration**: PyTorch Lightning workflows
* **Performance Optimization**: Tips for efficient usage

Getting Started
--------------

If you're new to EvoAug2, start with the :doc:`basic_usage` examples.

For a complete training pipeline, see the :doc:`deepstarr_training` example.

Code Examples
------------

All examples include complete, runnable code that you can copy and modify for your own projects.

**Basic Augmentation**:

.. code-block:: python

   from evoaug.augment import RandomMutation
   from evoaug.evoaug import RobustLoader
   
   # Simple mutation augmentation
   mutation = RandomMutation(mut_frac=0.05)
   loader = RobustLoader(dataset, [mutation], max_augs_per_seq=1)

**Multiple Augmentations**:

.. code-block:: python

   augmentations = [
       RandomDeletion(delete_min=0, delete_max=20),
       RandomInsertion(insert_min=0, insert_max=20),
       RandomMutation(mut_frac=0.05)
   ]
   
   loader = RobustLoader(
       dataset, augmentations, 
       max_augs_per_seq=2, hard_aug=True
   )

**PyTorch Lightning Integration**:

.. code-block:: python

   class AugmentedDataModule(pl.LightningDataModule):
       def train_dataloader(self):
           return RobustLoader(
               self.dataset, self.augmentations, 
               max_augs_per_seq=2
           )

Running Examples
---------------

All examples can be run directly from the command line:

.. code-block:: bash

   # Run basic usage example
   python examples/basic_usage.py
   
   # Run DeepSTARR training
   python examples/deepstarr_training.py
   
   # Run with custom augmentations
   python examples/custom_augmentations.py

Requirements
-----------

Examples require the following dependencies:

* EvoAug2 (installed)
* PyTorch >= 1.9.0
* PyTorch Lightning >= 1.5.0
* NumPy >= 1.20.0
* Matplotlib >= 3.3.0 (for visualization examples)
* Seaborn >= 0.11.0 (for visualization examples)

Install with:

.. code-block:: bash

   pip install evoaug2[examples]

For development examples:

.. code-block:: bash

   pip install evoaug2[dev,examples]

Contributing Examples
--------------------

We welcome contributions of new examples! See the :doc:`../contributing` guide for details.

When contributing examples:

* Include complete, runnable code
* Add comprehensive comments
* Include expected output
* Test on multiple Python versions
* Follow the existing style guidelines 