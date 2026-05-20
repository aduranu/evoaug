# EvoAug

[![PyPI version](https://badge.fury.io/py/evoaug.svg)](https://pypi.org/project/evoaug/)
[![Documentation Status](https://readthedocs.org/projects/evoaug/badge/?version=latest)](https://evoaug.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

EvoAug is a PyTorch package to pretrain sequence-based deep learning models for
regulatory genomics with evolution-inspired data augmentations, followed by
fine-tuning on the original, unperturbed sequence data.

**Starting in v2.0**, the recommended API is `RobustLoader` — a drop-in
`torch.utils.data.DataLoader` that applies augmentations on-the-fly. The v1
`RobustModel` Lightning-wrapper interface remains available under
`evoaug.legacy` for backwards compatibility. See
[CHANGELOG.md](CHANGELOG.md) for the migration guide.

This work follows from *"EvoAug: improving generalization and interpretability
of genomic deep neural networks with evolution-inspired data augmentations"*
(Lee, Tang, Toneyan, and Koo, **Genome Biology**, 2023).

For questions, email: koo@cshl.edu

<img src="fig/augmentations.png" alt="augmentations" width="500"/>

<img src="fig/overview.png" alt="overview" width="500"/>

## Install

```bash
pip install evoaug
```

### From source

```bash
git clone https://github.com/p-koo/evoaug.git
cd evoaug
pip install -e .
```

### Optional extras

```bash
pip install evoaug[examples]   # matplotlib, seaborn, jupyter
pip install evoaug[docs]       # sphinx + rtd theme
pip install evoaug[full]       # everything above
```

## Dependencies

```text
torch >= 1.9.0
lightning >= 2.0.0
numpy >= 1.20.0
scipy >= 1.7.0
h5py >= 3.1.0
scikit-learn >= 1.0.0
```

## Quick start (v2)

```python
from evoaug import RobustLoader, augment

augment_list = [
    augment.RandomDeletion(delete_min=0, delete_max=20),
    augment.RandomRC(rc_prob=0.5),
    augment.RandomMutation(mut_frac=0.05),
]

loader = RobustLoader(
    base_dataset=your_dataset,        # yields (sequence, target), sequence shape (A, L)
    augment_list=augment_list,
    max_augs_per_seq=2,
    hard_aug=True,
    batch_size=128,
    shuffle=True,
)

for x, y in loader:
    # x has shape (N, A, L), augmentations already applied
    ...
```

All augmentations are length-preserving: input shape `(N, A, L)` always returns
output shape `(N, A, L)`.

### Two-stage training (recommended)

```python
# Stage 1 — pretrain with augmentations
loader.enable_augmentations()
trainer.fit(model, train_dataloaders=loader)

# Stage 2 — fine-tune on original data with the same loader
loader.disable_augmentations()
trainer_ft.fit(model, train_dataloaders=loader)
```

This mirrors the EvoAug methodology and typically improves robustness and
generalization.

## Legacy v1 API (`RobustModel`)

The pre-2.0 Lightning-wrapper API is preserved under `evoaug.legacy`:

```python
from evoaug.legacy import RobustModel

robust_model = RobustModel(
    model,
    criterion=loss,
    optimizer=optimizer_dict,
    augment_list=augment_list,
    max_augs_per_seq=2,
    hard_aug=True,
    inference_aug=False,
)
trainer.fit(robust_model, datamodule=data_module)
```

Old import paths (`from evoaug import RobustModel`,
`from evoaug.evoaug import RobustModel`) continue to work in v2.x but emit a
`DeprecationWarning`. They will be removed in a future major release.

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

## Package layout

```
evoaug/
├── augment.py         # Length-preserving augmentations
├── loader.py          # RobustLoader, AugmentedGenomicDataset (v2 API)
├── utils.py           # H5Dataset, evaluation helpers
├── model_zoo.py       # DeepSTARR, Basset, generic CNN
└── legacy/            # v1 RobustModel API
    └── evoaug.py
```

## Documentation

Full documentation:
[evoaug.readthedocs.io](https://evoaug.readthedocs.io/en/latest/).

## Examples

- `examples/lightning_module.py` — full Lightning training loop with two-stage
  workflow.
- `examples/vanilla_pytorch.py` — minimal PyTorch training loop, no Lightning.

## Reference

```bibtex
@article{lee2023evoaug,
  title={EvoAug: improving generalization and interpretability of genomic deep neural networks with evolution-inspired data augmentations},
  author={Lee, Nicholas Keone and Tang, Ziqi and Toneyan, Shushan and Koo, Peter K},
  journal={Genome Biology},
  volume={24},
  number={1},
  pages={105},
  year={2023},
  publisher={Springer}
}
```
