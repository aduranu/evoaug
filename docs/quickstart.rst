Quick Start Guide
================

This guide will get you up and running with EvoAug2 in just a few minutes. You'll learn how to apply evolution-inspired augmentations to genomic sequences and integrate them into your PyTorch training pipeline.

Basic Usage
-----------

**Import the required modules:**

.. code-block:: python

   import torch
   from evoaug.augment import RandomMutation, RandomDeletion, RandomTranslocation
   from evoaug.evoaug import RobustLoader

**Create augmentation objects:**

.. code-block:: python

   # Single nucleotide mutations (5% mutation rate)
   mutation = RandomMutation(mut_frac=0.05)
   
   # Random deletions (0-30 nucleotides)
   deletion = RandomDeletion(delete_min=0, delete_max=30)
   
   # Random translocations (0-20 nucleotide shifts)
   translocation = RandomTranslocation(shift_min=0, shift_max=20)

**Apply augmentations to a sequence:**

.. code-block:: python

   # Example DNA sequence (one-hot encoded)
   sequence = torch.randn(1, 200, 4)  # [batch, length, channels]
   
   # Apply single augmentation
   augmented_seq = mutation(sequence)
   
   # Apply multiple augmentations
   augmentations = [mutation, deletion, translocation]
   for aug in augmentations:
       sequence = aug(sequence)

Integration with PyTorch DataLoader
----------------------------------

**Create a custom dataset with augmentations:**

.. code-block:: python

   from torch.utils.data import Dataset, DataLoader
   
   class GenomicDataset(Dataset):
       def __init__(self, sequences, labels, augmentations=None):
           self.sequences = sequences
           self.labels = labels
           self.augmentations = augmentations or []
       
       def __len__(self):
           return len(self.sequences)
       
       def __getitem__(self, idx):
           seq = self.sequences[idx]
           
           # Apply augmentations during training
           if self.augmentations:
               for aug in self.augmentations:
                   seq = aug(seq)
           
           return seq, self.labels[idx]
   
   # Create dataset with augmentations
   dataset = GenomicDataset(
       sequences=your_sequences,
       labels=your_labels,
       augmentations=[mutation, deletion, translocation]
   )
   
   # Create DataLoader
   dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

Using RobustLoader (Recommended)
-------------------------------

**RobustLoader provides advanced augmentation management:**

.. code-block:: python

   from evoaug.evoaug import RobustLoader
   
   # Create base dataset
   base_dataset = YourBaseDataset(sequences, labels)
   
   # Create RobustLoader with augmentations
   dataloader = RobustLoader(
       base_dataset=base_dataset,
       augment_list=[mutation, deletion, translocation],
       max_augs_per_seq=2,  # Apply up to 2 augmentations per sequence
       hard_aug=True,        # Always apply exactly 2 augmentations
       batch_size=32,
       shuffle=True
   )

**Disable augmentations for validation/testing:**

.. code-block:: python

   # For validation/testing, disable augmentations
   dataloader.disable_augmentations()

Two-Stage Training Approach
--------------------------

**Stage 1: Training with augmentations**

.. code-block:: python

   # Train model with augmentations
   model = YourModel()
   optimizer = torch.optim.Adam(model.parameters())
   
   for epoch in range(num_epochs):
       for batch_seqs, batch_labels in dataloader:
           # Forward pass
           outputs = model(batch_seqs)
           loss = criterion(outputs, batch_labels)
           
           # Backward pass
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()

**Stage 2: Fine-tuning on original data**

.. code-block:: python

   # Create dataloader without augmentations for fine-tuning
   finetune_dataloader = DataLoader(base_dataset, batch_size=32, shuffle=True)
   
   # Fine-tune with lower learning rate
   optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
   
   for epoch in range(5):  # Fewer epochs for fine-tuning
       for batch_seqs, batch_labels in finetune_dataloader:
           outputs = model(batch_seqs)
           loss = criterion(outputs, batch_labels)
           
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()

Complete Example
---------------

Here's a complete working example:

.. code-block:: python

   import torch
   import torch.nn as nn
   from torch.utils.data import Dataset, DataLoader
   from evoaug.augment import RandomMutation, RandomDeletion
   from evoaug.evoaug import RobustLoader
   
   # Simple model
   class SimpleModel(nn.Module):
       def __init__(self, input_size=200, num_classes=2):
           super().__init__()
           self.conv1 = nn.Conv1d(4, 32, kernel_size=3, padding=1)
           self.pool = nn.AdaptiveAvgPool1d(1)
           self.fc = nn.Linear(32, num_classes)
       
       def forward(self, x):
           x = x.transpose(1, 2)  # [batch, channels, length]
           x = torch.relu(self.conv1(x))
           x = self.pool(x).squeeze(-1)
           x = self.fc(x)
           return x
   
   # Dataset
   class GenomicDataset(Dataset):
       def __init__(self, sequences, labels):
           self.sequences = sequences
           self.labels = labels
       
       def __len__(self):
           return len(self.sequences)
       
       def __getitem__(self, idx):
           return self.sequences[idx], self.labels[idx]
   
   # Generate dummy data
   num_samples = 1000
   seq_length = 200
   num_classes = 2
   
   sequences = torch.randn(num_samples, seq_length, 4)
   labels = torch.randint(0, num_classes, (num_samples,))
   
   # Create dataset
   dataset = GenomicDataset(sequences, labels)
   
   # Create augmentations
   augmentations = [
       RandomMutation(mut_frac=0.05),
       RandomDeletion(delete_min=0, delete_max=20)
   ]
   
   # Create RobustLoader
   train_loader = RobustLoader(
       base_dataset=dataset,
       augment_list=augmentations,
       max_augs_per_seq=1,
       hard_aug=False,
       batch_size=32,
       shuffle=True
   )
   
   # Create model and training components
   model = SimpleModel()
   criterion = nn.CrossEntropyLoss()
   optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
   
   # Training loop
   model.train()
   for epoch in range(10):
       total_loss = 0
       for batch_seqs, batch_labels in train_loader:
           optimizer.zero_grad()
           outputs = model(batch_seqs)
           loss = criterion(outputs, batch_labels)
           loss.backward()
           optimizer.step()
           total_loss += loss.item()
       
       print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")
   
   print("Training completed!")

Next Steps
----------

Now that you have the basics:

1. **Explore Augmentations:** Check out the :doc:`user_guide/augmentations` for all available augmentation types
2. **Advanced Training:** Learn about the two-stage approach in :doc:`user_guide/training`
3. **Examples:** Run the complete examples in :doc:`examples/lightning_module` and :doc:`examples/vanilla_pytorch`
4. **API Reference:** Consult the :doc:`api/evoaug` for detailed class and method documentation

**Need Help?**

- Check the :doc:`user_guide/troubleshooting` for common issues
- Review the :doc:`examples` section for more complex use cases
- Visit the `GitHub repository <https://github.com/aduranu/evoaug>`_ for issues and discussions 