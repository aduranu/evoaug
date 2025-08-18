API Reference
=============

This section provides complete API documentation for EvoAug2.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   augment
   evoaug
   utils

Overview
--------

The EvoAug2 API is organized into three main modules:

* **evoaug.augment**: Core augmentation classes and base classes
* **evoaug.evoaug**: Dataset and DataLoader implementations
* **utils**: Utility functions and data loading classes

Core Classes
-----------

Augmentation Classes
~~~~~~~~~~~~~~~~~~~

* :class:`evoaug.augment.AugmentBase` - Base class for all augmentations
* :class:`evoaug.augment.RandomDeletion` - Random deletion augmentation
* :class:`evoaug.augment.RandomInsertion` - Random insertion augmentation
* :class:`evoaug.augment.RandomTranslocation` - Translocation augmentation
* :class:`evoaug.augment.RandomInversion` - Inversion augmentation
* :class:`evoaug.augment.RandomMutation` - Point mutation augmentation
* :class:`evoaug.augment.RandomRC` - Reverse-complement augmentation
* :class:`evoaug.augment.RandomNoise` - Noise addition augmentation

Dataset and DataLoader Classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* :class:`evoaug.evoaug.AugmentedGenomicDataset` - Dataset wrapper with augmentations
* :class:`evoaug.evoaug.RobustLoader` - DataLoader with built-in augmentations

Utility Classes
~~~~~~~~~~~~~~

* :class:`utils.utils.H5DataModule` - PyTorch Lightning DataModule for H5 files
* :class:`utils.utils.H5Dataset` - Enhanced Dataset class for H5 files
* :class:`utils.model_zoo.DeepSTARR` - DeepSTARR model architecture
* :class:`utils.model_zoo.DeepSTARRModel` - PyTorch Lightning wrapper for DeepSTARR

Quick Reference
--------------

.. code-block:: python

   # Import augmentations
   from evoaug.augment import (
       RandomDeletion, RandomInsertion, RandomTranslocation,
       RandomInversion, RandomMutation, RandomRC, RandomNoise
   )
   
   # Import dataset and dataloader
   from evoaug.evoaug import AugmentedGenomicDataset, RobustLoader
   
   # Import utilities
   from utils.utils import H5Dataset, H5DataModule
   from utils.model_zoo import DeepSTARR, DeepSTARRModel

Common Patterns
--------------

**Basic Augmentation Setup**:

.. code-block:: python

   augmentations = [
       RandomMutation(mut_frac=0.05),
       RandomDeletion(delete_min=0, delete_max=20)
   ]
   
   loader = RobustLoader(
       base_dataset=dataset,
       augment_list=augmentations,
       max_augs_per_seq=1
   )

**PyTorch Lightning Integration**:

.. code-block:: python

   class AugmentedDataModule(pl.LightningDataModule):
       def train_dataloader(self):
           return RobustLoader(
               base_dataset=self.dataset,
               augment_list=self.augmentations,
               max_augs_per_seq=2
           )

**Runtime Control**:

.. code-block:: python

   # Disable for validation
   loader.disable_augmentations()
   
   # Re-enable for training
   loader.enable_augmentations()

For detailed information about each class and method, see the individual module documentation pages. 