"""EvoAug: evolution-inspired augmentations for genomic sequence models.

The v2 API exposes :class:`~evoaug.loader.RobustLoader` — a drop-in
``torch.utils.data.DataLoader`` that applies length-preserving
augmentations on-the-fly — alongside the augmentation library
(:mod:`evoaug.augment`), an H5 dataset utility (:mod:`evoaug.utils`),
and reference model architectures (:mod:`evoaug.model_zoo`).

The v1 API (``RobustModel`` and helpers, a ``pl.LightningModule``
wrapper) remains available under :mod:`evoaug.legacy` and is also
re-exported here with a :class:`DeprecationWarning` for backwards
compatibility with existing code.

Example
-------
v2::

    from evoaug import RobustLoader, augment
    loader = RobustLoader(
        base_dataset=my_dataset,
        augment_list=[augment.RandomMutation(mut_frac=0.05)],
        max_augs_per_seq=2,
        hard_aug=True,
        batch_size=64,
    )

v1 (legacy)::

    from evoaug.legacy import RobustModel
"""

from . import augment, model_zoo, utils
from .loader import RobustLoader, AugmentedGenomicDataset

__version__ = "2.0.0"

__all__ = [
    "RobustLoader",
    "AugmentedGenomicDataset",
    "augment",
    "utils",
    "model_zoo",
]


# ---- v1 compatibility shim --------------------------------------------------
# evoaug <= 1.0.6 exposed `RobustModel`, `load_model_from_checkpoint`, and
# `augment_max_len` at the top level (and under `evoaug.evoaug`). They now
# live in `evoaug.legacy`; resolving them here keeps existing code working
# while nudging users to migrate.

_LEGACY_NAMES = {"RobustModel", "load_model_from_checkpoint", "augment_max_len"}


def __getattr__(name):
    if name in _LEGACY_NAMES:
        import warnings

        warnings.warn(
            f"evoaug.{name} moved to evoaug.legacy in v2.0. "
            f"Import it as `from evoaug.legacy import {name}` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        from . import legacy

        return getattr(legacy, name)
    raise AttributeError(f"module 'evoaug' has no attribute {name!r}")
