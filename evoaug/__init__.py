"""
EvoAug2: Evolution-Inspired Data Augmentation for Genomic Sequences

EvoAug2 is a PyTorch-based library that provides evolution-inspired data augmentation
techniques for genomic sequence data. It implements various augmentation strategies
that mimic biological mutations and evolutionary processes while maintaining sequence
length consistency.

The library is designed to work seamlessly with PyTorch Lightning and can be used
for training robust deep learning models on genomic data.

Classes
-------
AugmentBase
    Base class for all augmentation operations.
RandomDeletion
    Randomly deletes contiguous stretches of nucleotides.
RandomInsertion
    Randomly inserts random DNA sequences.
RandomTranslocation
    Applies circular shifts to sequences.
RandomInversion
    Inverts and reverse-complements sequence regions.
RandomMutation
    Introduces point mutations at random positions.
RandomRC
    Applies reverse-complement transformations.
RandomNoise
    Adds Gaussian noise to sequences.
AugmentedGenomicDataset
    Dataset wrapper that applies augmentations on-the-fly.
RobustLoader
    DataLoader with built-in augmentation capabilities.

Examples
--------
Basic usage with a single augmentation:

    >>> from evoaug.augment import RandomMutation
    >>> from evoaug.evoaug import RobustLoader
    >>> 
    >>> # Create augmentation
    >>> mutation = RandomMutation(mut_frac=0.05)
    >>> 
    >>> # Use with RobustLoader
    >>> loader = RobustLoader(
    ...     base_dataset=your_dataset,
    ...     augment_list=[mutation],
    ...     max_augs_per_seq=1
    ... )

Advanced usage with multiple augmentations:

    >>> from evoaug.augment import (
    ...     RandomDeletion, RandomInsertion, RandomTranslocation,
    ...     RandomMutation, RandomNoise
    ... )
    >>> 
    >>> augment_list = [
    ...     RandomDeletion(delete_min=0, delete_max=20),
    ...     RandomInsertion(insert_min=0, insert_max=20),
    ...     RandomTranslocation(shift_min=0, shift_max=20),
    ...     RandomMutation(mut_frac=0.05),
    ...     RandomNoise(noise_std=0.2)
    ... ]
    >>> 
    >>> loader = RobustLoader(
    ...     base_dataset=your_dataset,
    ...     augment_list=augment_list,
    ...     max_augs_per_seq=2,
    ...     hard_aug=True
    ... )

Integration with PyTorch Lightning:

    >>> import pytorch_lightning as pl
    >>> from evoaug.evoaug import AugmentedGenomicDataset
    >>> 
    >>> class AugmentedDataModule(pl.LightningDataModule):
    ...     def __init__(self, base_dataset, augment_list):
    ...         super().__init__()
    ...         self.base_dataset = base_dataset
    ...         self.augment_list = augment_list
    ...     
    ...     def train_dataloader(self):
    ...         train_dataset = AugmentedGenomicDataset(
    ...             self.base_dataset, self.augment_list, max_augs_per_seq=2
    ...         )
    ...         return DataLoader(train_dataset, batch_size=32, shuffle=True)
    ...     
    ...     def val_dataloader(self):
    ...         val_dataset = AugmentedGenomicDataset(
    ...             self.base_dataset, self.augment_list, max_augs_per_seq=0
    ...         )
    ...         return DataLoader(val_dataset, batch_size=32, shuffle=False)

Notes
-----
- All augmentations preserve the original sequence length L
- Augmentations can be enabled/disabled at runtime
- The library is designed to work with one-hot encoded DNA sequences
- Supports both single sequences and batched data
- Integrates seamlessly with PyTorch Lightning workflows

References
----------
This library implements augmentation strategies inspired by:
- Biological mutation processes
- Evolutionary sequence variations
- Sequence motif preservation techniques

For more information, see the documentation at:
https://evoaug2.readthedocs.io/
"""

# Version information
__version__ = "2.0.0"
__author__ = "EvoAug2 Development Team"
__email__ = "evoaug2@example.com"

# Import main classes and functions
from .augment import (
    AugmentBase,
    RandomDeletion,
    RandomInsertion,
    RandomTranslocation,
    RandomInversion,
    RandomMutation,
    RandomRC,
    RandomNoise
)

from .evoaug import (
    AugmentedGenomicDataset,
    RobustLoader
)

# Define what gets imported with "from evoaug import *"
__all__ = [
    # Base classes
    'AugmentBase',
    
    # Augmentation classes
    'RandomDeletion',
    'RandomInsertion', 
    'RandomTranslocation',
    'RandomInversion',
    'RandomMutation',
    'RandomRC',
    'RandomNoise',
    
    # Dataset and DataLoader classes
    'AugmentedGenomicDataset',
    'RobustLoader',
]

# Package metadata
__package_name__ = "evoaug2"
__description__ = "Evolution-Inspired Data Augmentation for Genomic Sequences"
__keywords__ = [
    "genomics", "data-augmentation", "deep-learning", "pytorch", 
    "bioinformatics", "sequence-analysis", "evolution", "mutations"
]
__classifiers__ = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Topic :: Scientific/Engineering :: Bio-Informatics",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Software Development :: Libraries :: Python Modules",
]

# Import version info
try:
    from ._version import version as __version__
except ImportError:
    pass
