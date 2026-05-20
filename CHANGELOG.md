# Changelog

All notable changes to this project will be documented in this file. This
project adheres to [Semantic Versioning](https://semver.org/).

## [2.0.0] — Unreleased

### Highlights

EvoAug 2.0 replaces the v1 `RobustModel` Lightning-module wrapper with a
**DataLoader-first** API: `RobustLoader` is a drop-in
`torch.utils.data.DataLoader` that applies length-preserving augmentations
on-the-fly. The new design works with **any** PyTorch model — no
`LightningModule` subclass required — and lets you toggle augmentations at the
loader level, which makes the two-stage *pretrain → fine-tune* workflow a
single line of code (`loader.disable_augmentations()`).

### Added

- `evoaug.RobustLoader` — `DataLoader` subclass that applies augmentations
  on-the-fly. Supports `.enable_augmentations()` / `.disable_augmentations()`.
- `evoaug.AugmentedGenomicDataset` — `Dataset` wrapper holding the
  augmentation pipeline (used internally by `RobustLoader`; usable
  standalone if you need a `Dataset` instead of a `DataLoader`).
- `evoaug.utils.H5Dataset`, `evoaug.utils.H5DataModule` — HDF5-backed
  datasets exposing train/val/test splits.
- `evoaug.utils` — evaluation helpers (Pearson, Spearman, AUROC, AUPRC, MSE)
  and other training utilities.
- `evoaug.model_zoo` — reference architectures: `DeepSTARR`,
  `DeepSTARRModel`, `Basset`, generic `CNN`.
- `examples/lightning_module.py` and `examples/vanilla_pytorch.py` —
  end-to-end example scripts for both Lightning and vanilla PyTorch.
- Sphinx documentation tree (`docs/`) ready for Read the Docs.

### Changed

- Augmentation classes (`RandomDeletion`, `RandomInsertion`,
  `RandomTranslocation`, `RandomRC`, `RandomMutation`, `RandomNoise`) gained
  full NumPy-style docstrings. Class names and constructor signatures are
  identical to v1 — **no migration needed** for code that constructs
  augmentations.
- Packaging modernized: added `pyproject.toml`, expanded dependency declarations
  (`scipy`, `h5py`, `scikit-learn`), added `[examples]`, `[docs]`, `[full]`
  extras.

### Deprecated

- `evoaug.RobustModel`, `evoaug.load_model_from_checkpoint`,
  `evoaug.augment_max_len` and the `evoaug.evoaug` submodule are preserved
  for backwards compatibility but emit `DeprecationWarning` on access. The
  v1 implementation has moved to `evoaug.legacy`.

### Fixed

- `RandomInsertion`: off-by-one in the insertion-window index calculation
  that could clip the last position.
- `RandomTranslocation`: tightened shift bookkeeping so the output is
  guaranteed `(N, A, L)` regardless of shift direction.

### Migration guide

#### Before (v1)

```python
from evoaug import evoaug, augment

augment_list = [augment.RandomMutation(mut_frac=0.05), ...]

robust_model = evoaug.RobustModel(
    model, criterion=loss, optimizer=opt,
    augment_list=augment_list, max_augs_per_seq=2, hard_aug=True,
)
trainer.fit(robust_model, datamodule=dm)
```

#### After (v2)

```python
from evoaug import RobustLoader, augment

augment_list = [augment.RandomMutation(mut_frac=0.05), ...]

train_loader = RobustLoader(
    base_dataset=train_ds,
    augment_list=augment_list,
    max_augs_per_seq=2,
    hard_aug=True,
    batch_size=128,
    shuffle=True,
)
trainer.fit(model, train_dataloaders=train_loader)

# Stage 2 — same loader, augmentations off
train_loader.disable_augmentations()
trainer_ft.fit(model, train_dataloaders=train_loader)
```

The base dataset must yield `(sequence, target)` pairs where `sequence` has
shape `(A, L)`; `RobustLoader` batches them into `(N, A, L)` tensors and
applies augmentations during iteration.

If you cannot migrate right away, the v1 API still works:

```python
from evoaug.legacy import RobustModel       # preferred
from evoaug import RobustModel              # works, emits DeprecationWarning
from evoaug.evoaug import RobustModel       # works, emits DeprecationWarning
```

## [1.0.6] — 2023

The previous stable release. See git history for details.
