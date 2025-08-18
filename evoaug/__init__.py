"""
EvoAug2: Evolution-inspired sequence augmentations as a DataLoader.

EvoAug2 provides evolution-inspired data augmentations for genomic sequences and a simple
way to use them with any PyTorch model via a drop-in DataLoader. It replaces the old
model-wrapper approach with a lightweight dataset/loader that applies augmentations on-the-fly.

The core design goal is to keep the output sequence length exactly equal to the input
length L for every augmentation, ensuring consistent batch shapes during training.

Classes
-------
AugmentBase
    Base class for all augmentation classes.
RandomDeletion
    Randomly deletes contiguous stretches and pads with random DNA.
RandomInsertion
    Randomly inserts random DNA and trims to maintain length.
RandomTranslocation
    Randomly shifts sequences using circular roll transformations.
RandomInversion
    Randomly inverts contiguous stretches with reverse-complement.
RandomMutation
    Randomly mutates nucleotides according to mutation fraction.
RandomRC
    Randomly applies reverse-complement transformations.
RandomNoise
    Randomly adds Gaussian noise to sequences.
AugmentedGenomicDataset
    Dataset wrapper that applies augmentations on-the-fly.
RobustLoader
    DataLoader with built-in EvoAug augmentations.

Examples
--------
Basic usage with PyTorch Lightning:

    >>> from evoaug.augment import RandomDeletion, RandomInsertion
    >>> from evoaug.evoaug import RobustLoader
    >>> 
    >>> augment_list = [
    ...     RandomDeletion(delete_min=0, delete_max=20),
    ...     RandomInsertion(insert_min=0, insert_max=20)
    ... ]
    >>> 
    >>> train_loader = RobustLoader(
    ...     base_dataset=your_dataset,
    ...     augment_list=augment_list,
    ...     max_augs_per_seq=2,
    ...     hard_aug=True,
    ...     batch_size=32
    ... )

Basic usage with vanilla PyTorch:

    >>> for epoch in range(num_epochs):
    ...     for x, y in train_loader:
    ...         # x has shape (N, A, L) - same as input
    ...         y_hat = model(x)
    ...         loss = criterion(y_hat, y)
    ...         loss.backward()
    ...         optimizer.step()

Notes
-----
- All augmentations preserve input sequence length L
- Augmentations can be enabled/disabled at runtime
- Each sequence receives different random augmentations
- Compatible with PyTorch Lightning DataModules and vanilla PyTorch
"""

# Import main classes for easy access
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

# Package metadata
__version__ = "2.0.0"
__author__ = "EvoAug Team"
__email__ = "koo@cshl.edu"

# Define what gets imported with "from evoaug import *"
__all__ = [
    # Base classes
    "AugmentBase",
    
    # Augmentation classes
    "RandomDeletion",
    "RandomInsertion", 
    "RandomTranslocation",
    "RandomInversion",
    "RandomMutation",
    "RandomRC",
    "RandomNoise",
    
    # Dataset and DataLoader classes
    "AugmentedGenomicDataset",
    "RobustLoader",
]
