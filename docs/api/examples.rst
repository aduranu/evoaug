EvoAug2 Examples API
====================

This page documents the example scripts and their usage.

Lightning Module Example
------------------------

The PyTorch Lightning integration example demonstrates:

.. automodule:: example_lightning_module
   :members:
   :undoc-members:
   :show-inheritance:

Vanilla PyTorch Example
-----------------------

The vanilla PyTorch example shows basic integration:

.. automodule:: example_vanilla_pytorch
   :members:
   :undoc-members:
   :show-inheritance:

Running Examples
----------------

**Prerequisites:**

.. code-block:: bash

   # Install dependencies
   pip install evoaug2[examples]
   
   # Download DeepSTARR data
   python -c "from evoaug_utils import utils; utils.download_deepstarr_data()"

**Lightning Example:**

.. code-block:: bash

   python example_lightning_module.py

**Vanilla PyTorch Example:**

.. code-block:: bash

   python example_vanilla_pytorch.py

**Expected Outputs:**

- Trained model checkpoints (.ckpt files)
- Performance comparison plots (.png files)
- Training logs and evaluation metrics 