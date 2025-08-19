PyTorch Lightning Integration Example
====================================

This example demonstrates a complete EvoAug2 training workflow using PyTorch Lightning, implementing the two-stage training approach with comprehensive checkpoint management and performance evaluation.

Overview
--------

The Lightning module example (`example_lightning_module.py`) showcases:

- **Two-stage training approach** (augmentation + fine-tuning)
- **PyTorch Lightning integration** with custom DataModules
- **Checkpoint management** and resumption capabilities
- **Performance comparison** between different training strategies
- **Comprehensive visualization** of results
- **DeepSTARR model** training on genomic regulatory data

Key Features
------------

1. **RobustLoader Integration**: Uses EvoAug2's RobustLoader for efficient augmentation management
2. **Checkpoint Resumption**: Automatically detects and resumes from existing checkpoints
3. **Performance Metrics**: Comprehensive evaluation with Pearson and Spearman correlations
4. **Visualization**: Automatic generation of comparison plots and performance summaries
5. **Flexible Configuration**: Easy parameter modification for different experiments

File Structure
--------------

.. code-block:: text

   example_lightning_module.py
   ├── Configuration and setup
   ├── Checkpoint management functions
   ├── Plotting and visualization functions
   ├── Main training function
   │   ├── Stage 1: Training with augmentations
   │   ├── Stage 2: Fine-tuning on original data
   │   └── Control: Standard training comparison
   └── Results analysis and visualization

Usage
-----

**Basic Execution:**

.. code-block:: bash

   python example_lightning_module.py

**Prerequisites:**

.. code-block:: bash

   # Install dependencies
   pip install evoaug2[full]
   
   # Download DeepSTARR data
   wget https://zenodo.org/record/7265991/files/DeepSTARR_data.h5

**Configuration:**

The script can be customized by modifying these parameters:

.. code-block:: python

   # Experiment configuration
   expt_name = 'DeepSTARR'
   data_path = '.'
   filepath = '/path/to/deepstarr-data.h5'
   output_dir = '/path/to/output/'
   batch_size = 128
   
   # Augmentation parameters (DeepSTARR optimal settings)
   augment_list = [
       RandomTranslocation(shift_min=0, shift_max=20),
       RandomRC(rc_prob=0.0),
       RandomMutation(mut_frac=0.05),
       RandomNoise(noise_mean=0, noise_std=0.3),
   ]
   
   # Training parameters
   max_augs_per_seq = 2  # Maximum augmentations per sequence
   hard_aug = True        # Always apply exactly 2 augmentations
   max_epochs = 100       # Stage 1 training epochs
   finetune_epochs = 5    # Stage 2 fine-tuning epochs

Training Stages
---------------

**Stage 1: Augmentation Training**

.. code-block:: python

   # Create augmented data module
   data_module = AugmentedDataModule(
       base_dataset, 
       augment_list, 
       max_augs_per_seq=2,
       hard_aug=True
   )
   
   # Train with augmentations
   trainer.fit(model, datamodule=data_module)

**Stage 2: Fine-tuning**

.. code-block:: python

   # Load best augmented model
   model_finetune = DeepSTARRModel.load_from_checkpoint(
       best_model_path, 
       model=deepstarr
   )
   
   # Fine-tune on original data
   model_finetune.learning_rate = 0.0001
   trainer_finetune.fit(model_finetune, datamodule=data_module_finetune)

**Control Training**

.. code-block:: python

   # Train control model without augmentations
   model_control = DeepSTARRModel(deepstarr_control)
   trainer_control.fit(model_control, datamodule=data_module_control)

DataModule Implementation
------------------------

**AugmentedDataModule:**

.. code-block:: python

   class AugmentedDataModule(pl.LightningDataModule):
       def __init__(self, base_dataset, augment_list, max_augs_per_seq, hard_aug):
           super().__init__()
           self.base_dataset = base_dataset
           self.augment_list = augment_list
           self.max_augs_per_seq = max_augs_per_seq
           self.hard_aug = hard_aug
           
       def train_dataloader(self):
           # Use RobustLoader with training dataset
           train_dataset = self.base_dataset.get_train_dataset()
           return RobustLoader(
               base_dataset=train_dataset,
               augment_list=self.augment_list,
               max_augs_per_seq=self.max_augs_per_seq,
               hard_aug=self.hard_aug,
               batch_size=self.base_dataset.batch_size,
               shuffle=True
           )
       
       def val_dataloader(self):
           # Validation with augmentations disabled
           val_dataset = self.base_dataset.get_val_dataset()
           loader = RobustLoader(...)
           loader.disable_augmentations()
           return loader

**FineTuneDataModule:**

.. code-block:: python

   class FineTuneDataModule(pl.LightningDataModule):
       def __init__(self, base_dataset):
           super().__init__()
           self.base_dataset = base_dataset
           
       def train_dataloader(self):
           return self.base_dataset.train_dataloader()

Checkpoint Management
--------------------

**Automatic Detection:**

.. code-block:: python

   def check_existing_checkpoints(output_dir, expt_name):
       """Check for existing checkpoints and return their status."""
       checkpoint_status = {
           'augmented': {'exists': False, 'path': None, 'epochs': None},
           'finetuned': {'exists': False, 'path': None, 'epochs': None},
           'control': {'exists': False, 'path': None, 'epochs': None}
       }
       
       # Check each checkpoint type
       aug_path = os.path.join(output_dir, f"{expt_name}_aug.ckpt")
       if os.path.exists(aug_path):
           checkpoint_status['augmented']['exists'] = True
           # ... extract metadata

**Resumption Logic:**

.. code-block:: python

   if checkpoint_status['augmented']['exists']:
       print(f"✓ Found existing augmented model checkpoint")
       print("Skipping Stage 1 training - using existing model.")
       best_model_path = checkpoint_status['augmented']['path']
   else:
       # Train new model
       trainer.fit(model, datamodule=data_module)

Performance Evaluation
---------------------

**Metrics Calculation:**

.. code-block:: python

   # Get predictions
   pred = utils.get_predictions(model, base_dataset.x_test, batch_size=batch_size)
   
   # Calculate correlations
   pearson_r = []
   for class_index in range(y_true.shape[-1]):
       r = stats.pearsonr(y_true[:,class_index], y_score[:,class_index])[0]
       pearson_r.append(r)
   
   spearman_r = []
   for class_index in range(y_true.shape[-1]):
       r = stats.spearmanr(y_true[:,class_index], y_score[:,class_index])[0]
       spearman_r.append(r)

**Results Storage:**

.. code-block:: python

   metrics_data['augmented'] = {
       'pearson_r': pearson_aug,
       'spearman_r': spearman_r
   }

Visualization
-------------

**Comparison Plots:**

.. code-block:: python

   def plot_metrics_comparison(metrics_data, plots_dir, expt_name):
       """Create comprehensive plots comparing metrics across model types."""
       
       # 1. Correlation Metrics Comparison (Bar Plot)
       # 2. Detailed Metrics by Class (Heatmap)
       # 3. Performance Improvement Analysis
       # 4. Individual Model Performance
       # 5. Summary Statistics Table

**Generated Files:**

- `{expt_name}_metrics_comparison.png` - Overall performance comparison
- `{expt_name}_individual_performance.png` - Individual model analysis
- `{expt_name}_performance_summary.png` - Statistical summary table

Output Structure
----------------

**Model Checkpoints:**

.. code-block:: text

   output_dir/
   ├── DeepSTARR_aug.ckpt          # Stage 1: Augmented model
   ├── DeepSTARR_finetune.ckpt     # Stage 2: Fine-tuned model
   └── DeepSTARR_standard.ckpt     # Control: Standard model

**Plots and Results:**

.. code-block:: text

   plots/
   ├── DeepSTARR_metrics_comparison.png
   ├── DeepSTARR_individual_performance.png
   └── DeepSTARR_performance_summary.png

**Training Logs:**

- PyTorch Lightning logs in `lightning_logs/`
- Console output with training progress
- Checkpoint status information

Customization
-------------

**Modify Augmentation Strategy:**

.. code-block:: python

   # Change augmentation types
   augment_list = [
       RandomDeletion(delete_min=0, delete_max=30),
       RandomInsertion(insert_min=0, insert_max=20),
       RandomMutation(mut_frac=0.1),
   ]
   
   # Adjust augmentation frequency
   max_augs_per_seq = 3  # Apply up to 3 augmentations
   hard_aug = False      # Stochastic application

**Modify Training Parameters:**

.. code-block:: python

   # Change learning rates
   learning_rate = 0.0005      # Stage 1
   finetune_lr = 0.00005      # Stage 2
   
   # Adjust epochs
   max_epochs = 50             # Stage 1
   finetune_epochs = 10        # Stage 2
   
   # Modify batch size
   batch_size = 64             # Smaller for memory constraints

**Custom Models:**

.. code-block:: python

   # Use different model architecture
   from utils.model_zoo import YourCustomModel
   model = YourCustomModel(num_classes=2)

Troubleshooting
---------------

**Common Issues:**

1. **Memory Errors:**
   - Reduce `batch_size`
   - Use gradient accumulation
   - Enable mixed precision training

2. **Checkpoint Issues:**
   - Verify output directory permissions
   - Check available disk space
   - Ensure consistent model architecture

3. **Data Loading Problems:**
   - Verify data file path
   - Check data format compatibility
   - Ensure sufficient memory for dataset

**Performance Tips:**

- Use GPU acceleration when available
- Enable mixed precision training
- Monitor memory usage during training
- Use appropriate batch sizes for your hardware

Next Steps
----------

After running this example:

1. **Analyze Results:** Review generated plots and metrics
2. **Experiment:** Modify parameters and compare results
3. **Scale Up:** Apply to larger datasets or different models
4. **Customize:** Adapt for your specific use case

**Further Learning:**

- Read the :doc:`user_guide/training` for detailed training explanations
- Explore the :doc:`api/evoaug` for API reference
- Check the :doc:`examples/vanilla_pytorch` for PyTorch-only approach
- Review the :doc:`user_guide/augmentations` for augmentation details

This example provides a production-ready template for implementing EvoAug2 in PyTorch Lightning workflows and can serve as a foundation for your own genomic sequence analysis projects. 