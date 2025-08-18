Quick Start Guide
=================

This guide will get you up and running with EvoAug2 in minutes.

Installation
-----------

First, install EvoAug2:

.. code-block:: bash

   pip install evoaug2

Basic Usage
-----------

Import the necessary modules:

.. code-block:: python

   import torch
   from evoaug.augment import RandomMutation, RandomDeletion
   from evoaug.evoaug import RobustLoader

Create a simple dataset:

.. code-block:: python

   # Create sample data (batch_size=4, alphabet_size=4, sequence_length=100)
   x = torch.randn(4, 4, 100)  # 4 sequences, 4 nucleotides, length 100
   y = torch.randn(4, 2)       # 4 targets, 2 classes
   
   # Create a simple dataset
   dataset = torch.utils.data.TensorDataset(x, y)

Create augmentations:

.. code-block:: python

   # Define augmentation strategies
   augmentations = [
       RandomMutation(mut_frac=0.05),      # 5% mutation rate
       RandomDeletion(delete_min=0, delete_max=20)  # Delete 0-20 nucleotides
   ]

Use with RobustLoader:

.. code-block:: python

   # Create the augmented dataloader
   loader = RobustLoader(
       base_dataset=dataset,
       augment_list=augmentations,
       max_augs_per_seq=1,  # Apply 1 augmentation per sequence
       hard_aug=True,        # Always apply exactly 1 augmentation
       batch_size=2,
       shuffle=True
   )

Train your model:

.. code-block:: python

   # Simple training loop
   model = torch.nn.Linear(400, 2)  # Simple linear model
   optimizer = torch.optim.Adam(model.parameters())
   criterion = torch.nn.MSELoss()
   
   for epoch in range(5):
       for batch_idx, (x_aug, y) in enumerate(loader):
           # x_aug has shape (batch_size, 4, 100) - same as input!
           x_flat = x_aug.view(x_aug.size(0), -1)  # Flatten for linear layer
           
           optimizer.zero_grad()
           y_hat = model(x_flat)
           loss = criterion(y_hat, y)
           loss.backward()
           optimizer.step()
           
           if batch_idx % 10 == 0:
               print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')

PyTorch Lightning Integration
----------------------------

EvoAug2 works seamlessly with PyTorch Lightning:

.. code-block:: python

   import pytorch_lightning as pl
   from evoaug.evoaug import AugmentedGenomicDataset
   
   class AugmentedDataModule(pl.LightningDataModule):
       def __init__(self, base_dataset, augmentations):
           super().__init__()
           self.base_dataset = base_dataset
           self.augmentations = augmentations
       
       def train_dataloader(self):
           # Apply augmentations during training
           train_dataset = AugmentedGenomicDataset(
               self.base_dataset, 
               self.augmentations, 
               max_augs_per_seq=2
           )
           return torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
       
       def val_dataloader(self):
           # No augmentations during validation
           val_dataset = AugmentedGenomicDataset(
               self.base_dataset, 
               self.augmentations, 
               max_augs_per_seq=0
           )
           return torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

   # Use with Lightning trainer
   trainer = pl.Trainer(max_epochs=10)
   trainer.fit(model, datamodule=AugmentedDataModule(dataset, augmentations))

Available Augmentations
----------------------

EvoAug2 provides several evolution-inspired augmentations:

**RandomMutation**: Point mutations at random positions

.. code-block:: python

   mutation = RandomMutation(mut_frac=0.05)  # 5% mutation rate

**RandomDeletion**: Delete contiguous stretches

.. code-block:: python

   deletion = RandomDeletion(delete_min=0, delete_max=20)

**RandomInsertion**: Insert random DNA sequences

.. code-block:: python

   insertion = RandomInsertion(insert_min=0, insert_max=20)

**RandomTranslocation**: Circular sequence shifts

.. code-block:: python

   translocation = RandomTranslocation(shift_min=0, shift_max=20)

**RandomInversion**: Reverse-complement regions

.. code-block:: python

   inversion = RandomInversion(invert_min=0, invert_max=20)

**RandomRC**: Reverse-complement entire sequences

.. code-block:: python

   rc = RandomRC(rc_prob=0.5)  # 50% chance of RC

**RandomNoise**: Add Gaussian noise

.. code-block:: python

   noise = RandomNoise(noise_mean=0.0, noise_std=0.2)

Advanced Configuration
---------------------

**Multiple augmentations per sequence**:

.. code-block:: python

   # Apply 2-3 augmentations per sequence
   loader = RobustLoader(
       base_dataset=dataset,
       augment_list=augmentations,
       max_augs_per_seq=3,
       hard_aug=False,  # Random number between 1-3
       batch_size=32
   )

**Custom augmentation combinations**:

.. code-block:: python

   # Create specific augmentation combinations
   augmentations = [
       RandomMutation(mut_frac=0.05),
       RandomDeletion(delete_min=0, delete_max=30),
       RandomInsertion(insert_min=0, insert_max=20),
       RandomTranslocation(shift_min=0, shift_max=25),
       RandomNoise(noise_std=0.3)
   ]

**Runtime control**:

.. code-block:: python

   # Disable augmentations for validation/finetuning
   loader.disable_augmentations()
   
   # Re-enable for training
   loader.enable_augmentations()

Real-World Example
-----------------

Here's a complete example using the DeepSTARR model:

.. code-block:: python

   from evoaug.augment import (
       RandomDeletion, RandomInsertion, RandomTranslocation,
       RandomMutation, RandomNoise
   )
   from evoaug.evoaug import RobustLoader
   from utils.model_zoo import DeepSTARR, DeepSTARRModel
   
   # Load your genomic data
   # dataset = load_genomic_data()
   
   # Define augmentations (DeepSTARR optimal parameters)
   augmentations = [
       RandomDeletion(delete_min=0, delete_max=30),
       RandomTranslocation(shift_min=0, shift_max=20),
       RandomInsertion(insert_min=0, insert_max=20),
       RandomMutation(mut_frac=0.05),
       RandomNoise(noise_mean=0, noise_std=0.3)
   ]
   
   # Create augmented dataloader
   train_loader = RobustLoader(
       base_dataset=dataset,
       augment_list=augmentations,
       max_augs_per_seq=2,
       hard_aug=True,
       batch_size=128
   )
   
   # Create model
   model = DeepSTARRModel(DeepSTARR(2), learning_rate=0.001)
   
   # Train with PyTorch Lightning
   trainer = pl.Trainer(max_epochs=100)
   trainer.fit(model, train_dataloaders=train_loader)

Key Features
-----------

* **Length Preservation**: All augmentations maintain input sequence length
* **Batch Consistency**: Consistent shapes across all sequences in a batch
* **Runtime Control**: Enable/disable augmentations as needed
* **PyTorch Native**: Works with any PyTorch model
* **Lightning Ready**: Seamless PyTorch Lightning integration
* **Efficient**: On-the-fly augmentation without preprocessing

Next Steps
----------

* Read the :doc:`user_guide/index` for detailed usage information
* Explore the :doc:`api_reference/index` for complete API documentation
* Check out :doc:`examples/index` for more examples
* Learn about :doc:`contributing` if you want to contribute

Need Help?
----------

* Check the `GitHub repository <https://github.com/evoaug2/evoaug2>`_
* Open an `issue <https://github.com/evoaug2/evoaug2/issues>`_ for bugs
* Start a `discussion <https://github.com/evoaug2/evoaug2/discussions>`_ for questions 