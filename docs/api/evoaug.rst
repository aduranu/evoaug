EvoAug2 Core API
================

This page documents the core EvoAug2 modules for augmentations and training.

Augmentation Module
------------------

The `evoaug.augment` module provides evolution-inspired data augmentation techniques:

.. automodule:: evoaug.augment
   :members:
   :undoc-members:
   :show-inheritance:

Training Module
--------------

The `evoaug.evoaug` module provides training utilities and the RobustLoader:

.. automodule:: evoaug.evoaug
   :members:
   :undoc-members:
   :show-inheritance:

Usage Examples
--------------

**Basic Augmentation:**

.. code-block:: python

   from evoaug.augment import RandomMutation, RandomDeletion
   from evoaug.evoaug import RobustLoader
   
   # Create augmentations
   mutation = RandomMutation(mut_frac=0.05)
   deletion = RandomDeletion(delete_min=0, delete_max=20)
   
   # Create RobustLoader
   loader = RobustLoader(
       base_dataset=dataset,
       augment_list=[mutation, deletion],
       max_augs_per_seq=2,
       hard_aug=True,
       batch_size=32
   )

**Training Loop:**

.. code-block:: python

   for batch_seqs, batch_labels in loader:
       # Augmentations applied automatically
       outputs = model(batch_seqs)
       loss = criterion(outputs, batch_labels)
       loss.backward()
       optimizer.step() 