## EvoAug2: Evolution-inspired sequence augmentations as a DataLoader

EvoAug2 provides evolution-inspired data augmentations for genomic sequences and a simple way to use them with any PyTorch model via a drop-in DataLoader. It replaces the old model-wrapper approach with a lightweight dataset/loader that applies augmentations on-the-fly. The core design goal is to keep the output sequence length exactly equal to the input length L for every augmentation.

- **Where to look**:
  - `evoaug/augment.py`: augmentation implementations (length-preserving)
  - `evoaug/evoaug.py`: `AugmentedGenomicDataset` and `RobustLoader`
  - `example_training.py`: complete training script with Lightning integration and two-stage approach

## What's new
- **Loader-first design**: `RobustLoader` wraps any base dataset and applies augmentations stochastically per mini-batch, no model wrapper needed.
- **Length-preserving augmentations**:
  - **RandomDeletion**: deletes a contiguous segment and pads equally with random DNA to keep length L.
  - **RandomInsertion**: inserts random DNA at a chosen index, then trims equally from both sequence ends so final length is exactly L.
  - **RandomTranslocation**, **RandomInversion**, **RandomRC**, **RandomMutation**, **RandomNoise** all preserve input shape.
- **Safer indexing**: all per-sequence lengths/indices are handled robustly for consistent slicing across PyTorch versions.
- **Enhanced H5Dataset**: Integrated DataModule-like functionality for seamless integration with PyTorch Lightning.
- **Two-stage training workflow**: Pretrain with augmentations, then fine-tune on original data for optimal performance.
- **Checkpoint management**: Automatic detection and loading of existing checkpoints to avoid redundant training.

## Install
```bash
pip install evoaug
```

## Dependencies
- torch >= 1.12
- lightning >= 2.0.0
- numpy >= 1.21

Note: This package uses the newer `lightning` package (`lightning.pytorch`). If you use older `pytorch_lightning`, adjust the `Trainer` call accordingly.

## Augmentations
```python
from evoaug import augment

augment_list = [
    augment.RandomDeletion(delete_min=0, delete_max=30),
    augment.RandomTranslocation(shift_min=0, shift_max=20),
    augment.RandomInsertion(insert_min=0, insert_max=20),
    augment.RandomRC(rc_prob=0.0),
    augment.RandomMutation(mut_frac=0.05),
    augment.RandomNoise(noise_mean=0.0, noise_std=0.3),
]
```
All transforms return tensors with the same shape as input `(N, A, L)`.

## Using RobustLoader with PyTorch Lightning (DataModule)

### Two-Stage Training Approach (Recommended)

The recommended workflow follows the EvoAug methodology:

1. **Stage 1**: Train with EvoAug2 augmentations using `RobustLoader`
2. **Stage 2**: Fine-tune on the original data (augmentations disabled)
3. **Control**: Train a separate model on original data for baseline comparison

```python
import lightning.pytorch as pl
from evoaug.evoaug import RobustLoader
from evoaug import augment
from utils import utils

# DeepSTARR optimal augmentation hyperparameters
augment_list = [
    augment.RandomDeletion(delete_min=0, delete_max=30),
    augment.RandomTranslocation(shift_min=0, shift_max=20),
    # RandomInsertion(insert_min=0, insert_max=20),  # Commented out if causing issues
    augment.RandomRC(rc_prob=0.0),
    augment.RandomMutation(mut_frac=0.05),
    augment.RandomNoise(noise_mean=0.0, noise_std=0.3),
]

# Create enhanced H5Dataset (includes DataModule-like functionality)
base_dataset = utils.H5Dataset(filepath, batch_size=batch_size, lower_case=False, transpose=False)

# Create augmented data module for Stage 1 training
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
        # Use RobustLoader with validation dataset and disable augmentations
        val_dataset = self.base_dataset.get_val_dataset()
        loader = RobustLoader(
            base_dataset=val_dataset,
            augment_list=self.augment_list,
            max_augs_per_seq=self.max_augs_per_seq,
            hard_aug=self.hard_aug,
            batch_size=self.base_dataset.batch_size,
            shuffle=False
        )
        loader.disable_augmentations()
        return loader
    
    def test_dataloader(self):
        # Use RobustLoader with test dataset and disable augmentations
        test_dataset = self.base_dataset.get_test_dataset()
        loader = RobustLoader(
            base_dataset=test_dataset,
            augment_list=self.augment_list,
            max_augs_per_seq=self.max_augs_per_seq,
            hard_aug=self.hard_aug,
            batch_size=self.base_dataset.batch_size,
            shuffle=False
        )
        loader.disable_augmentations()
        return loader

# Create augmented data module for Stage 1
data_module = AugmentedDataModule(
    base_dataset, 
    augment_list, 
    max_augs_per_seq=2,  # DeepSTARR optimal: maximum 2 augmentations per sequence
    hard_aug=True         # DeepSTARR uses hard setting: always apply exactly 2 augmentations
)

# STAGE 1: Training with EvoAug2 Augmentations
trainer = pl.Trainer(
    max_epochs=100,
    accelerator="auto",
    devices="auto"
)

trainer.fit(model, datamodule=data_module)

# STAGE 2: Fine-tuning on Original Data
# Create fine-tune data module (no augmentations)
class FineTuneDataModule(pl.LightningDataModule):
    def __init__(self, base_dataset):
        super().__init__()
        self.base_dataset = base_dataset
        
    def train_dataloader(self):
        return self.base_dataset.train_dataloader()
        
    def val_dataloader(self):
        return self.base_dataset.val_dataloader()
        
    def test_dataloader(self):
        return self.base_dataset.test_dataloader()

data_module_finetune = FineTuneDataModule(base_dataset)

# Fine-tune for 5 epochs with lower learning rate
trainer_finetune = pl.Trainer(
    max_epochs=5,
    accelerator="auto",
    devices="auto"
)

trainer_finetune.fit(model_finetune, datamodule=data_module_finetune)
```

### Checkpoint Management

The training script automatically detects existing checkpoints and skips redundant training:

```python
def check_existing_checkpoints(output_dir, expt_name):
    """Check for existing checkpoints and return their status."""
    checkpoint_status = {
        'augmented': {'exists': False, 'path': None, 'epochs': None},
        'finetuned': {'exists': False, 'path': None, 'epochs': None},
        'control': {'exists': False, 'path': None, 'epochs': None}
    }
    
    # Check for augmented model checkpoint
    aug_path = os.path.join(output_dir, f"{expt_name}_aug.ckpt")
    if os.path.exists(aug_path):
        checkpoint_status['augmented']['exists'] = True
        checkpoint_status['augmented']['path'] = aug_path
    
    # Check for fine-tuned model checkpoint
    finetune_path = os.path.join(output_dir, f"{expt_name}_finetune.ckpt")
    if os.path.exists(finetune_path):
        checkpoint_status['finetuned']['exists'] = True
        checkpoint_status['finetuned']['path'] = finetune_path
    
    # Check for control model checkpoint
    control_path = os.path.join(output_dir, f"{expt_name}_standard.ckpt")
    if os.path.exists(control_path):
        checkpoint_status['control']['exists'] = True
        checkpoint_status['control']['path'] = control_path
    
    return checkpoint_status
```

## Using RobustLoader with a vanilla PyTorch training loop
```python
from evoaug.evoaug import RobustLoader
from evoaug import augment

# Your base dataset must return (sequence, target) where sequence has shape (A, L)
base_dataset = YourDataset(...)
augment_list = [
    augment.RandomDeletion(delete_min=0, delete_max=30),
    augment.RandomTranslocation(shift_min=0, shift_max=20),
    augment.RandomInsertion(insert_min=0, insert_max=20),
]

train_loader = RobustLoader(
    base_dataset=base_dataset,
    augment_list=augment_list,
    max_augs_per_seq=2,
    hard_aug=True,
    batch_size=128,
    shuffle=True,
    num_workers=4
)

# Standard PyTorch loop (model expects input (N, A, L))
for epoch in range(num_epochs):
    model.train()
    for x, y in train_loader:
        x = x.to(device)      # (N, A, L)
        y = y.to(device)
        optimizer.zero_grad()
        y_hat = model(x)
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()

# Validation/test: use your original non-augmented loader or disable augmentations
# train_loader.disable_augmentations()
```

## Enhanced H5Dataset Integration

The `H5Dataset` now includes DataModule-like functionality for seamless integration:

```python
from utils import utils

# Create enhanced H5Dataset with DataModule-like methods
base_dataset = utils.H5Dataset(
    filepath, 
    batch_size=batch_size, 
    lower_case=False, 
    transpose=False
)

# Access train/val/test datasets
train_dataset = base_dataset.get_train_dataset()
val_dataset = base_dataset.get_val_dataset()
test_dataset = base_dataset.get_test_dataset()

# Use with standard DataLoader or RobustLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
```

## Performance Visualization

The training script automatically generates comprehensive performance comparison plots:

```python
def plot_metrics_comparison(metrics_data, plots_dir, expt_name):
    """Create comprehensive plots comparing metrics across the three model types."""
    
    # 1. Correlation Metrics Comparison (Bar Plot)
    # 2. Detailed Metrics by Class (Heatmap)
    # 3. Performance Improvement Analysis
    # 4. Model Comparison Summary
    
    # Plots are saved to the local 'plots/' directory
    # Includes Pearson correlation, Spearman correlation, and class-wise analysis
```

## API overview
- `AugmentedGenomicDataset(base_dataset, augment_list, max_augs_per_seq=0, hard_aug=True, apply_augmentations=True)`
  - Wraps an existing dataset and applies augmentations on-the-fly.
  - `max_augs_per_seq`: maximum number of augmentations per sequence.
  - `hard_aug=True`: always apply exactly `max_augs_per_seq` augmentations; otherwise a random number from 1..max is used.
- `RobustLoader(...)`
  - `enable_augmentations()` / `disable_augmentations()` to toggle augments (handy for finetuning).
  - Preserves the original sequence length L for all transforms.
- `H5Dataset(filepath, batch_size, lower_case=False, transpose=False)`
  - Enhanced dataset class with DataModule-like functionality.
  - `get_train_dataset()`, `get_val_dataset()`, `get_test_dataset()` methods.
  - Seamless integration with PyTorch Lightning.

## Tips and gotchas
- If you use lazy modules (e.g., `nn.LazyLinear`), ensure train/val inputs have the same shape on the very first forward. With EvoAug2's length-preserving transforms, this should hold if your base datasets share the same L.
- To avoid any lazy-module edge cases during sanity validation, you can set:
```python
pl.Trainer(num_sanity_val_steps=0, ...)
```
- Alternatively, replace lazy layers with fixed-shape heads (e.g., global average pooling + `nn.Linear`).
- The `RandomInsertion` augmentation may be commented out in the example if it causes issues in your environment.
- Use the two-stage training workflow for optimal performance: pretrain with augmentations, then fine-tune on original data.

## Two-stage training workflow (recommended)
1. **Pretrain with EvoAug2 augmentations** using `RobustLoader` (100 epochs).
2. **Fine-tune on the original data** (5 epochs, lower learning rate).
3. **Control comparison** with model trained on original data only.

This follows the EvoAug methodology and typically improves robustness and generalization.

## Reference
- Paper: "EvoAug: improving generalization and interpretability of genomic deep neural networks with evolution-inspired data augmentations" (Genome Biology, 2023).
- For questions: koo@cshl.edu 