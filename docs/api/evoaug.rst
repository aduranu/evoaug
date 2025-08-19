EvoAug2 API Reference
=====================

This section provides comprehensive API documentation for the EvoAug2 package, including all classes, methods, and functions.

Package Overview
---------------

EvoAug2 is organized into several key modules:

- **`evoaug.augment`**: Core augmentation classes for genomic sequences
- **`evoaug.evoaug`**: Main training utilities and RobustLoader
- **`utils.model_zoo`**: Pre-built model architectures
- **`utils.utils`**: Utility functions for data handling and evaluation

Core Modules
------------

.. toctree::
   :maxdepth: 2

   evoaug/augment
   evoaug/evoaug
   utils/model_zoo
   utils/utils

Augmentation Classes
-------------------

The `evoaug.augment` module provides evolution-inspired data augmentation techniques:

**Base Augmentation Class:**

.. autoclass:: evoaug.augment.Augmentation
   :members:
   :undoc-members:
   :show-inheritance:

**Mutation-Based Augmentations:**

.. autoclass:: evoaug.augment.RandomMutation
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: evoaug.augment.RandomNoise
   :members:
   :undoc-members:
   :show-inheritance:

**Structural Augmentations:**

.. autoclass:: evoaug.augment.RandomDeletion
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: evoaug.augment.RandomInsertion
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: evoaug.augment.RandomTranslocation
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: evoaug.augment.RandomRC
   :members:
   :undoc-members:
   :show-inheritance:

Training Utilities
-----------------

The `evoaug.evoaug` module provides training infrastructure:

**RobustLoader:**

.. autoclass:: evoaug.evoaug.RobustLoader
   :members:
   :undoc-members:
   :show-inheritance:

**Training Functions:**

.. autofunction:: evoaug.evoaug.train_with_augmentations

.. autofunction:: evoaug.evoaug.finetune_model

Model Zoo
---------

The `utils.model_zoo` module provides pre-built architectures:

**DeepSTARR Models:**

.. autoclass:: utils.model_zoo.DeepSTARR
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: utils.model_zoo.DeepSTARRModel
   :members:
   :undoc-members:
   :show-inheritance:

**Other Models:**

.. autoclass:: utils.model_zoo.Basset
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: utils.model_zoo.BassetModel
   :members:
   :undoc-members:
   :show-inheritance:

Utility Functions
----------------

The `utils.utils` module provides helper functions:

**Data Handling:**

.. autofunction:: utils.utils.H5Dataset
   :noindex:

.. autofunction:: utils.utils.load_data

.. autofunction:: utils.utils.preprocess_sequences

**Evaluation:**

.. autofunction:: utils.utils.evaluate_model

.. autofunction:: utils.utils.get_predictions

.. autofunction:: utils.utils.calculate_metrics

**Visualization:**

.. autofunction:: utils.utils.plot_training_curves

.. autofunction:: utils.utils.plot_augmentation_effects

Usage Examples
--------------

**Basic Augmentation:**

.. code-block:: python

   from evoaug.augment import RandomMutation, RandomDeletion
   
   # Create augmentations
   mutation = RandomMutation(mut_frac=0.05)
   deletion = RandomDeletion(delete_min=0, delete_max=20)
   
   # Apply to sequence
   sequence = torch.randn(1, 200, 4)
   augmented = mutation(sequence)
   augmented = deletion(augmented)

**Training with RobustLoader:**

.. code-block:: python

   from evoaug.evoaug import RobustLoader
   
   # Create loader
   loader = RobustLoader(
       base_dataset=dataset,
       augment_list=[mutation, deletion],
       max_augs_per_seq=2,
       hard_aug=True,
       batch_size=32
   )
   
   # Training loop
   for batch_seqs, batch_labels in loader:
       # Augmentations applied automatically
       outputs = model(batch_seqs)
       loss = criterion(outputs, batch_labels)
       # ... rest of training

**Model Training:**

.. code-block:: python

   from utils.model_zoo import DeepSTARR, DeepSTARRModel
   
   # Create model
   model = DeepSTARRModel(DeepSTARR(2))
   
   # Train with augmentations
   trainer.fit(model, datamodule=data_module)

**Evaluation:**

.. code-block:: python

   from utils import utils
   
   # Get predictions
   predictions = utils.get_predictions(model, test_data)
   
   # Evaluate
   results = utils.evaluate_model(true_labels, predictions, task='regression')

Configuration
------------

**Augmentation Parameters:**

Each augmentation class accepts specific parameters that control the augmentation behavior:

- **RandomMutation**: `mut_frac` - fraction of positions to mutate
- **RandomDeletion**: `delete_min`, `delete_max` - deletion range
- **RandomInsertion**: `insert_min`, `insert_max` - insertion range
- **RandomTranslocation**: `shift_min`, `shift_max` - shift range
- **RandomRC**: `rc_prob` - reverse-complement probability
- **RandomNoise**: `noise_mean`, `noise_std` - noise parameters

**Training Parameters:**

- **max_augs_per_seq**: Maximum augmentations per sequence
- **hard_aug**: Whether to always apply exactly N augmentations
- **batch_size**: Training batch size
- **shuffle**: Whether to shuffle data

**Model Parameters:**

- **input_size**: Sequence length
- **num_classes**: Number of output classes
- **learning_rate**: Training learning rate
- **weight_decay**: L2 regularization

Error Handling
--------------

**Common Errors and Solutions:**

1. **Shape Mismatch Errors:**
   - Ensure input tensors have shape `[batch, length, channels]`
   - Check that augmentation parameters are within valid ranges

2. **Memory Errors:**
   - Reduce batch size
   - Use gradient accumulation
   - Enable mixed precision training

3. **Data Type Errors:**
   - Ensure input tensors are `torch.float32`
   - Check label tensors are appropriate type (long for classification)

**Debugging Tips:**

.. code-block:: python

   # Check tensor shapes and types
   print(f"Input shape: {sequence.shape}")
   print(f"Input dtype: {sequence.dtype}")
   
   # Verify augmentation parameters
   print(f"Mutation fraction: {mutation.mut_frac}")
   print(f"Deletion range: {deletion.delete_min}-{deletion.delete_max}")

Performance Considerations
------------------------

**Optimization Tips:**

1. **Use GPU Acceleration:**
   - Move tensors to GPU: `sequence = sequence.cuda()`
   - Use mixed precision training when available

2. **Batch Processing:**
   - Use appropriate batch sizes for your hardware
   - Consider gradient accumulation for large effective batch sizes

3. **Augmentation Efficiency:**
   - Limit number of augmentations per sequence
   - Use `hard_aug=False` for stochastic application

**Memory Management:**

- Monitor GPU memory usage
- Clear cache: `torch.cuda.empty_cache()`
- Use gradient checkpointing for large models

Compatibility
-------------

**PyTorch Versions:**
- PyTorch >= 1.9.0
- Python >= 3.8

**Hardware Support:**
- CPU and GPU (CUDA) support
- Mixed precision training support
- Distributed training support

**Framework Integration:**
- Native PyTorch support
- PyTorch Lightning integration
- Custom training loop support

For detailed information about specific classes and methods, see the individual module documentation pages. 