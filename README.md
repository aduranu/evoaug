## EvoAug2: Evolution-inspired sequence augmentations as a DataLoader

EvoAug2 provides evolution-inspired data augmentations for genomic sequences and a simple way to use them with any PyTorch model via a drop-in DataLoader. It replaces the old model-wrapper approach with a lightweight dataset/loader that applies augmentations on-the-fly. The core design goal is to keep the output sequence length exactly equal to the input length L for every augmentation.

- **Where to look**:
  - `evoaug/augment.py`: augmentation implementations (length-preserving)
  - `evoaug/evoaug2.py`: `AugmentedGenomicDataset` and `RobustLoader`
  - `evoaug2_deepstarr_training.py`: example training script with Lightning

## What’s new
- **Loader-first design**: `RobustLoader` wraps any base dataset and applies augmentations stochastically per mini-batch, no model wrapper needed.
- **Length-preserving augmentations**:
  - **RandomDeletion**: deletes a contiguous segment and pads equally with random DNA to keep length L.
  - **RandomInsertion**: inserts random DNA at a chosen index, then trims equally from both sequence ends so final length is exactly L.
  - **RandomTranslocation**, **RandomInversion**, **RandomRC**, **RandomMutation**, **RandomNoise** all preserve input shape.
- **Safer indexing**: all per-sequence lengths/indices are handled robustly for consistent slicing across PyTorch versions.

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
Wrap your existing dataset with `RobustLoader` for training, and disable augmentations for validation/test. Because augmentations preserve length, train/val/test shapes will match, including during sanity validation.

```python
import lightning.pytorch as pl
from evoaug.evoaug2 import RobustLoader
from evoaug import augment

augment_list = [
    augment.RandomDeletion(delete_min=0, delete_max=30),
    augment.RandomTranslocation(shift_min=0, shift_max=20),
    augment.RandomInsertion(insert_min=0, insert_max=20),
    augment.RandomRC(rc_prob=0.0),
    augment.RandomMutation(mut_frac=0.05),
    augment.RandomNoise(noise_mean=0.0, noise_std=0.3),
]

class AugmentedDataModule(pl.LightningDataModule):
    def __init__(self, base_dataset, augment_list, max_augs_per_seq=2, hard_aug=True):
        super().__init__()
        self.base_dataset = base_dataset
        self.augment_list = augment_list
        self.max_augs_per_seq = max_augs_per_seq
        self.hard_aug = hard_aug

    def train_dataloader(self):
        base_dataset = self.base_module.train_dataloader().dataset
        return RobustLoader(
            base_dataset=base_dataset,
            augment_list=self.augment_list,
            max_augs_per_seq=self.max_augs_per_seq,
            hard_aug=self.hard_aug,
            batch_size=self.base_module.batch_size,
            shuffle=True
        )

    def val_dataloader(self):
        return self.base_module.val_dataloader()

    def test_dataloader(self):
        return self.base_module.test_dataloader()

# Training with Lightning
trainer = pl.Trainer(
    max_epochs=100,
    accelerator="auto",
    devices="auto"
)

trainer.fit(model, datamodule=AugmentedDataModule(
    base_module=your_base_datamodule,
    augment_list=augment_list,
    max_augs_per_seq=2,
    hard_aug=True
))

# Finetune on original data (no augmentations)
trainer.fit(model, datamodule=your_base_datamodule)
```

## Using RobustLoader with a vanilla PyTorch training loop
```python
from evoaug.evoaug2 import RobustLoader
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

## API overview
- `AugmentedGenomicDataset(base_dataset, augment_list, max_augs_per_seq=0, hard_aug=True, apply_augmentations=True)`
  - Wraps an existing dataset and applies augmentations on-the-fly.
  - `max_augs_per_seq`: maximum number of augmentations per sequence.
  - `hard_aug=True`: always apply exactly `max_augs_per_seq` augmentations; otherwise a random number from 1..max is used.
- `RobustLoader(...)`
  - `enable_augmentations()` / `disable_augmentations()` to toggle augments (handy for finetuning).
  - Preserves the original sequence length L for all transforms.

## Tips and gotchas
- If you use lazy modules (e.g., `nn.LazyLinear`), ensure train/val inputs have the same shape on the very first forward. With EvoAug2’s length-preserving transforms, this should hold if your base datasets share the same L.
- To avoid any lazy-module edge cases during sanity validation, you can set:
```python
pl.Trainer(num_sanity_val_steps=0, ...)
```
- Alternatively, replace lazy layers with fixed-shape heads (e.g., global average pooling + `nn.Linear`).

## Two-stage training workflow (recommended)
1. Pretrain with EvoAug2 augmentations using `RobustLoader`.
2. Finetune on the original data (augmentations disabled or standard loader).

This follows the EvoAug methodology and typically improves robustness and generalization.

## Reference
- Paper: “EvoAug: improving generalization and interpretability of genomic deep neural networks with evolution-inspired data augmentations” (Genome Biology, 2023).
- For questions: koo@cshl.edu 