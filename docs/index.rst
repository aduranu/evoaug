EvoAug2 Documentation
=====================

Evolution-inspired sequence augmentations as a DataLoader.

EvoAug2 provides evolution-inspired data augmentations for genomic sequences and a simple
way to use them with any PyTorch model via a drop-in DataLoader. It replaces the old
model-wrapper approach with a lightweight dataset/loader that applies augmentations on-the-fly.

The core design goal is to keep the output sequence length exactly equal to the input
length L for every augmentation, ensuring consistent batch shapes during training.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   augmentations
   dataloader
   examples
   api

Features
--------

* **Length-preserving augmentations**: All transforms maintain input sequence length L
* **Drop-in DataLoader**: Use with any PyTorch model or PyTorch Lightning
* **Runtime control**: Enable/disable augmentations for training vs validation
* **Evolution-inspired**: Based on biological sequence evolution patterns
* **Efficient**: On-the-fly augmentation without preprocessing

Quick Start
-----------

Install the package:

.. code-block:: bash

   pip install evoaug

Basic usage with PyTorch Lightning:

.. code-block:: python

   from evoaug.augment import RandomDeletion, RandomInsertion
   from evoaug.evoaug import RobustLoader
   
   augment_list = [
       RandomDeletion(delete_min=0, delete_max=20),
       RandomInsertion(insert_min=0, insert_max=20)
   ]
   
   train_loader = RobustLoader(
       base_dataset=your_dataset,
       augment_list=augment_list,
       max_augs_per_seq=2,
       hard_aug=True,
       batch_size=32
   )

Basic usage with vanilla PyTorch:

.. code-block:: python

   for epoch in range(num_epochs):
       for x, y in train_loader:
           # x has shape (N, A, L) - same as input
           y_hat = model(x)
           loss = criterion(y_hat, y)
           loss.backward()
           optimizer.step()

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search` 