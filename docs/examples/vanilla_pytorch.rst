Vanilla PyTorch Integration Example
==================================

This example demonstrates basic EvoAug2 functionality using vanilla PyTorch, providing a simple and direct approach to integrating evolution-inspired augmentations into your training pipeline.

Overview
--------

The vanilla PyTorch example (`example_vanilla_pytorch.py`) showcases:

- **Direct PyTorch integration** without Lightning abstractions
- **Basic augmentation application** to genomic sequences
- **Simple training loop** implementation
- **Core EvoAug2 functionality** demonstration
- **Minimal dependencies** for quick prototyping

Key Features
------------

1. **Simple Implementation**: Straightforward PyTorch code without external abstractions
2. **Core Augmentations**: Demonstrates basic mutation and deletion augmentations
3. **Easy Customization**: Simple to modify for different use cases
4. **Minimal Setup**: Requires only basic PyTorch knowledge
5. **Quick Prototyping**: Ideal for research and experimentation

File Structure
--------------

.. code-block:: text

   example_vanilla_pytorch.py
   ├── Imports and setup
   ├── Augmentation definition
   ├── Simple model definition
   ├── Training loop
   └── Basic evaluation

Usage
-----

**Basic Execution:**

.. code-block:: bash

   python example_vanilla_pytorch.py

**Prerequisites:**

.. code-block:: bash

   # Install core dependencies
   pip install evoaug2
   
   # Or install from source
   git clone https://github.com/aduranu/evoaug.git
   cd evoaug
   pip install -e .

**Dependencies:**

- PyTorch >= 1.9.0
- NumPy >= 1.20.0
- EvoAug2 core package

Code Walkthrough
----------------

**1. Imports and Setup:**

.. code-block:: python

   import torch
   import torch.nn as nn
   from evoaug.augment import RandomMutation, RandomDeletion
   from evoaug.evoaug import RobustLoader

**2. Augmentation Definition:**

.. code-block:: python

   # Define augmentations
   augmentations = [
       RandomMutation(mut_frac=0.1),      # 10% mutation rate
       RandomDeletion(delete_min=0, delete_max=20)  # 0-20 deletions
   ]

**3. Simple Model:**

.. code-block:: python

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

**4. Training Loop:**

.. code-block:: python

   # Training setup
   model = SimpleModel()
   criterion = nn.CrossEntropyLoss()
   optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
   
   # Training loop
   model.train()
   for epoch in range(num_epochs):
       for batch_seqs, batch_labels in dataloader:
           # Apply augmentations
           for aug in augmentations:
               batch_seqs = aug(batch_seqs)
           
           # Forward pass
           outputs = model(batch_seqs)
           loss = criterion(outputs, batch_labels)
           
           # Backward pass
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()

Augmentation Application
-----------------------

**Direct Application:**

.. code-block:: python

   # Apply single augmentation
   augmented_seq = mutation(sequence)
   
   # Apply multiple augmentations sequentially
   for aug in augmentations:
       sequence = aug(sequence)

**Batch Processing:**

.. code-block:: python

   # Apply to entire batch
   batch_size, seq_length, channels = batch_seqs.shape
   
   # Apply augmentations to each sequence in batch
   for i in range(batch_size):
       for aug in augmentations:
           batch_seqs[i] = aug(batch_seqs[i:i+1]).squeeze(0)

**RobustLoader Integration:**

.. code-block:: python

   # Use RobustLoader for efficient batch processing
   dataloader = RobustLoader(
       base_dataset=dataset,
       augment_list=augmentations,
       max_augs_per_seq=1,
       hard_aug=False,
       batch_size=32,
       shuffle=True
   )

Customization Examples
----------------------

**Modify Augmentation Parameters:**

.. code-block:: python

   # Adjust mutation rate
   mutation = RandomMutation(mut_frac=0.05)  # 5% mutations
   
   # Change deletion range
   deletion = RandomDeletion(delete_min=5, delete_max=50)  # 5-50 deletions
   
   # Add new augmentation types
   from evoaug.augment import RandomTranslocation, RandomNoise
   
   augmentations = [
       RandomMutation(mut_frac=0.1),
       RandomDeletion(delete_min=0, delete_max=20),
       RandomTranslocation(shift_min=0, shift_max=15),
       RandomNoise(noise_mean=0, noise_std=0.2)
   ]

**Custom Training Loop:**

.. code-block:: python

   # Add learning rate scheduling
   scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
   
   # Add validation loop
   model.eval()
   with torch.no_grad():
       val_loss = 0
       for val_seqs, val_labels in val_dataloader:
           outputs = model(val_seqs)
           val_loss += criterion(outputs, val_labels).item()
       print(f"Validation Loss: {val_loss/len(val_dataloader):.4f}")
   
   scheduler.step()

**Custom Model Architecture:**

.. code-block:: python

   class CustomModel(nn.Module):
       def __init__(self, input_size=200, num_classes=2):
           super().__init__()
           self.conv1 = nn.Conv1d(4, 64, kernel_size=5, padding=2)
           self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
           self.dropout = nn.Dropout(0.3)
           self.pool = nn.AdaptiveAvgPool1d(1)
           self.fc1 = nn.Linear(128, 64)
           self.fc2 = nn.Linear(64, num_classes)
       
       def forward(self, x):
           x = x.transpose(1, 2)
           x = torch.relu(self.conv1(x))
           x = torch.relu(self.conv2(x))
           x = self.dropout(x)
           x = self.pool(x).squeeze(-1)
           x = torch.relu(self.fc1(x))
           x = self.fc2(x)
           return x

Data Handling
-------------

**Basic Dataset:**

.. code-block:: python

   class SimpleDataset(torch.utils.data.Dataset):
       def __init__(self, sequences, labels):
           self.sequences = sequences
           self.labels = labels
       
       def __len__(self):
           return len(self.sequences)
       
       def __getitem__(self, idx):
           return self.sequences[idx], self.labels[idx]

**Data Loading:**

.. code-block:: python

   # Create dataset
   dataset = SimpleDataset(sequences, labels)
   
   # Create DataLoader
   dataloader = torch.utils.data.DataLoader(
       dataset, 
       batch_size=32, 
       shuffle=True
   )

**Data Preprocessing:**

.. code-block:: python

   # Normalize sequences
   sequences = (sequences - sequences.mean()) / sequences.std()
   
   # Convert to float32
   sequences = sequences.float()
   labels = labels.long()

Training Configuration
---------------------

**Basic Training Parameters:**

.. code-block:: python

   # Training configuration
   num_epochs = 50
   batch_size = 32
   learning_rate = 0.001
   weight_decay = 1e-6
   
   # Model parameters
   input_size = 200
   num_classes = 2
   hidden_size = 64

**Advanced Training Options:**

.. code-block:: python

   # Mixed precision training
   scaler = torch.cuda.amp.GradScaler()
   
   # Gradient clipping
   max_grad_norm = 1.0
   
   # Early stopping
   patience = 10
   best_loss = float('inf')
   patience_counter = 0

**Device Configuration:**

.. code-block:: python

   # Check for GPU availability
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   print(f"Using device: {device}")
   
   # Move model and data to device
   model = model.to(device)
   batch_seqs = batch_seqs.to(device)
   batch_labels = batch_labels.to(device)

Evaluation and Metrics
---------------------

**Basic Evaluation:**

.. code-block:: python

   model.eval()
   correct = 0
   total = 0
   
   with torch.no_grad():
       for batch_seqs, batch_labels in dataloader:
           outputs = model(batch_seqs)
           _, predicted = torch.max(outputs.data, 1)
           total += batch_labels.size(0)
           correct += (predicted == batch_labels).sum().item()
   
   accuracy = 100 * correct / total
   print(f"Accuracy: {accuracy:.2f}%")

**Loss Tracking:**

.. code-block:: python

   # Track training loss
   train_losses = []
   
   for epoch in range(num_epochs):
       epoch_loss = 0
       for batch_seqs, batch_labels in dataloader:
           # ... training code ...
           epoch_loss += loss.item()
       
       avg_loss = epoch_loss / len(dataloader)
       train_losses.append(avg_loss)
       print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

**Model Saving:**

.. code-block:: python

   # Save best model
   if avg_loss < best_loss:
       best_loss = avg_loss
       torch.save(model.state_dict(), 'best_model.pth')
       print("Saved best model!")

Comparison with Lightning Example
--------------------------------

**Advantages of Vanilla PyTorch:**

- **Direct Control**: Full control over training loop and augmentation application
- **Simple Debugging**: Easier to debug and understand
- **Minimal Dependencies**: Fewer external dependencies
- **Customization**: Easy to modify for specific research needs
- **Learning**: Better for understanding PyTorch fundamentals

**Advantages of Lightning Example:**

- **Production Ready**: Professional training workflows
- **Built-in Features**: Logging, checkpointing, distributed training
- **Less Code**: More concise implementation
- **Best Practices**: Follows PyTorch Lightning conventions
- **Scalability**: Better for large-scale experiments

**When to Use Each:**

- **Use Vanilla PyTorch**: For research, prototyping, learning, simple workflows
- **Use Lightning**: For production, complex experiments, team collaboration

Troubleshooting
---------------

**Common Issues:**

1. **Memory Errors:**
   - Reduce batch size
   - Use gradient accumulation
   - Clear cache: `torch.cuda.empty_cache()`

2. **Training Instability:**
   - Reduce learning rate
   - Add gradient clipping
   - Check data normalization

3. **Augmentation Problems:**
   - Verify input tensor shapes
   - Check augmentation parameters
   - Ensure data types are correct

**Debugging Tips:**

.. code-block:: python

   # Print tensor shapes
   print(f"Input shape: {batch_seqs.shape}")
   print(f"Label shape: {batch_labels.shape}")
   
   # Check data types
   print(f"Input dtype: {batch_seqs.dtype}")
   print(f"Label dtype: {batch_labels.dtype}")
   
   # Verify augmentation output
   print(f"Original shape: {sequence.shape}")
   augmented = mutation(sequence)
   print(f"Augmented shape: {augmented.shape}")

Next Steps
----------

After running this example:

1. **Experiment**: Try different augmentation combinations
2. **Customize**: Modify the model architecture
3. **Scale Up**: Apply to larger datasets
4. **Compare**: Run the Lightning example for comparison

**Further Learning:**

- Read the :doc:`user_guide/augmentations` for all augmentation types
- Explore the :doc:`api/evoaug` for detailed API reference
- Check the :doc:`examples/lightning_module` for advanced workflows
- Review the :doc:`user_guide/training` for training strategies

This example provides a solid foundation for understanding EvoAug2's core functionality and can be easily extended for your specific research needs. 