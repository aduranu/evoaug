Examples
========

EvoAug2 provides comprehensive examples demonstrating different integration approaches and use cases. These examples range from basic PyTorch integration to advanced PyTorch Lightning workflows.

Available Examples
-----------------

**PyTorch Lightning Integration** (:doc:`examples/lightning_module`)
   Complete training script with Lightning integration and two-stage approach.
   Shows DataModule creation, checkpoint management, and performance comparison.

**Vanilla PyTorch Integration** (:doc:`examples/vanilla_pytorch`)
   Basic PyTorch implementation demonstrating core augmentation functionality.
   Ideal for users who prefer direct PyTorch control without Lightning abstractions.

Example Categories
-----------------

**Training Approaches:**

1. **Two-Stage Training (Recommended)**
   - Stage 1: Train with augmentations for robust feature learning
   - Stage 2: Fine-tune on original data to remove augmentation bias
   - Best performance and generalization

2. **Single-Stage Training**
   - Train with augmentations throughout
   - Simpler but may have augmentation bias
   - Good for quick prototyping

**Integration Methods:**

1. **PyTorch Lightning**
   - Professional training workflows
   - Built-in logging and checkpointing
   - Easy experiment management
   - Recommended for production use

2. **Vanilla PyTorch**
   - Direct control over training loop
   - Customizable augmentation strategies
   - Good for research and experimentation

**Augmentation Strategies:**

1. **Stochastic Augmentation**
   - Randomly apply augmentations during training
   - Good for general robustness

2. **Hard Augmentation**
   - Always apply exactly N augmentations per sequence
   - Consistent training signal
   - Used in EvoAug2 paper

Running the Examples
--------------------

**Prerequisites:**

.. code-block:: bash

   # Install with examples dependencies
   pip install evoaug2[examples]
   
   # Or install from source
   git clone https://github.com/aduranu/evoaug.git
   cd evoaug
   pip install -e .[examples]

**Download Data (for DeepSTARR examples):**

.. code-block:: bash

   # Download DeepSTARR dataset
   wget https://zenodo.org/record/7265991/files/DeepSTARR_data.h5
   
   # Or use the provided script
   python -c "from utils import utils; utils.download_deepstarr_data()"

**Run Lightning Example:**

.. code-block:: bash

   python example_lightning_module.py

**Run Vanilla PyTorch Example:**

.. code-block:: bash

   python example_vanilla_pytorch.py

Example Outputs
---------------

**Training Progress:**
- Loss curves for each stage
- Validation metrics
- Augmentation statistics

**Model Checkpoints:**
- Stage 1: Augmented model
- Stage 2: Fine-tuned model
- Control: Standard training model

**Performance Comparison:**
- Correlation metrics (Pearson, Spearman)
- Visualization plots
- Statistical analysis

**Generated Files:**
- Trained models (.ckpt files)
- Performance plots (.png files)
- Training logs
- Evaluation results

Customizing Examples
--------------------

**Modify Augmentation Parameters:**

.. code-block:: python

   # Adjust mutation rate
   RandomMutation(mut_frac=0.1)  # 10% mutation rate
   
   # Change deletion range
   RandomDeletion(delete_min=5, delete_max=50)  # 5-50 nucleotides
   
   # Modify translocation range
   RandomTranslocation(shift_min=10, shift_max=30)  # 10-30 shifts

**Change Training Parameters:**

.. code-block:: python

   # Adjust learning rates
   learning_rate = 0.0005  # Lower for fine-tuning
   
   # Modify epochs
   max_epochs = 50         # Fewer epochs for quick testing
   finetune_epochs = 3     # Shorter fine-tuning
   
   # Change batch size
   batch_size = 64         # Smaller for memory constraints

**Custom Datasets:**

.. code-block:: python

   # Load your own data
   from utils import utils
   
   # Custom H5Dataset
   dataset = utils.H5Dataset(
       filepath='your_data.h5',
       batch_size=32,
       lower_case=False,
       transpose=False
   )
   
   # Or use numpy arrays
   sequences = np.load('sequences.npy')
   labels = np.load('labels.npy')

Troubleshooting Examples
------------------------

**Common Issues:**

1. **Memory Errors:**
   - Reduce batch size
   - Use gradient accumulation
   - Enable mixed precision training

2. **Data Loading Issues:**
   - Check file paths
   - Verify data format
   - Ensure sufficient disk space

3. **Training Instability:**
   - Adjust learning rate
   - Check augmentation parameters
   - Verify data preprocessing

**Getting Help:**

- Check the :doc:`user_guide/troubleshooting` section
- Review example logs and error messages
- Consult the `GitHub Issues <https://github.com/aduranu/evoaug/issues>`_ page

Next Steps
----------

After running the examples:

1. **Modify Parameters:** Experiment with different augmentation settings
2. **Custom Datasets:** Apply to your own genomic data
3. **Advanced Usage:** Explore the :doc:`user_guide` for detailed explanations
4. **API Reference:** Check the :doc:`api/evoaug` for all available options

**Example Modifications:**

- Change augmentation types and parameters
- Modify model architectures
- Adjust training schedules
- Add custom evaluation metrics
- Integrate with other frameworks

The examples provide a solid foundation for understanding EvoAug2's capabilities and can be easily adapted for your specific use cases. 