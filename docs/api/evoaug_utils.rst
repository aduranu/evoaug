EvoAug2 API Reference
=====================

This page provides comprehensive API documentation for the EvoAug2 package.

Package Overview
---------------

The EvoAug2 package consists of several core modules:

- **`evoaug.augment`**: Core augmentation classes for genomic sequences
- **`evoaug.evoaug`**: Main training utilities and RobustLoader
- **`evoaug_utils.model_zoo`**: Pre-built model architectures
- **`evoaug_utils.utils`**: Utility functions for data handling and evaluation

Core Modules
-----------

.. toctree::
   :maxdepth: 2

   evoaug/augment
   evoaug/evoaug
   evoaug_utils/model_zoo
   evoaug_utils/utils

Augmentation Classes
-------------------

The `evoaug.augment` module provides the core augmentation classes:

**Base Augmentation:**

.. autoclass:: evoaug.augment.Augmentation
   :members:
   :undoc-members:
   :show-inheritance:

**Sequence Mutations:**

.. autoclass:: evoaug.augment.RandomMutation
   :members:
   :undoc-members:
   :show-inheritance:

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

**Sequence Transformations:**

.. autoclass:: evoaug.augment.RandomRC
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: evoaug.augment.RandomNoise
   :members:
   :undoc-members:
   :show-inheritance:

Training Utilities
-----------------

The `evoaug.evoaug` module provides training utilities:

**RobustLoader:**

.. autoclass:: evoaug.evoaug.RobustLoader
   :members:
   :undoc-members:
   :show-inheritance:

**Training Functions:**

.. autofunction:: evoaug.evoaug.train_with_augmentations

.. autofunction:: evoaug.evoaug.finetune_model

Model Architectures
------------------

The `evoaug_utils.model_zoo` module provides pre-built architectures:

**DeepSTARR Models:**

.. autoclass:: evoaug_utils.model_zoo.DeepSTARR
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: evoaug_utils.model_zoo.DeepSTARRModel
   :members:
   :undoc-members:
   :show-inheritance:

**Basset Models:**

.. autoclass:: evoaug_utils.model_zoo.Basset
   :members:
   :undoc-members:
   :show-inheritance:

**Other Models:**

.. autoclass:: evoaug_utils.model_zoo.CNN
   :members:
   :undoc-members:
   :show-inheritance:

Utility Functions
----------------

The `evoaug_utils.utils` module provides helper functions:

**Data Handling:**

.. autofunction:: evoaug_utils.utils.H5Dataset
   :noindex:

.. autoclass:: evoaug_utils.utils.H5DataModule
   :members:
   :undoc-members:
   :show-inheritance:

**Evaluation:**

.. autofunction:: evoaug_utils.utils.evaluate_model

.. autofunction:: evoaug_utils.utils.get_predictions

.. autofunction:: evoaug_utils.utils.calculate_auroc

.. autofunction:: evoaug_utils.utils.calculate_aupr

.. autofunction:: evoaug_utils.utils.calculate_mse

.. autofunction:: evoaug_utils.utils.calculate_pearsonr

.. autofunction:: evoaug_utils.utils.calculate_spearmanr

**Training Support:**

.. autofunction:: evoaug_utils.utils.configure_optimizer

.. autofunction:: evoaug_utils.utils.get_fmaps

.. autofunction:: evoaug_utils.utils.make_directory

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

   from evoaug_utils.model_zoo import DeepSTARR, DeepSTARRModel
   
   # Create model
   model = DeepSTARRModel(DeepSTARR(2))
   
   # Train with augmentations
   trainer.fit(model, datamodule=data_module)

**Evaluation:**

.. code-block:: python

   from evoaug_utils import utils
   
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