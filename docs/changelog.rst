Changelog
=========

All notable changes to EvoAug2 will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

[Unreleased]
------------

**Added**:
* Comprehensive Sphinx documentation
* Enhanced H5Dataset with DataModule-like functionality
* Improved checkpoint handling in training scripts
* Performance comparison plotting functionality

**Changed**:
* Updated all docstrings to Sphinx-compatible format
* Refactored codebase for better maintainability
* Enhanced package metadata and setup configuration

**Fixed**:
* Length preservation issues in RandomInsertion
* Batch consistency problems across augmentations
* Storage resizing errors during collation

[2.0.0] - 2024-01-XX
---------------------

**Added**:
* Complete rewrite with PyTorch Lightning integration
* New DataLoader-based architecture replacing model wrapper approach
* RobustLoader class for seamless augmentation integration
* AugmentedGenomicDataset for on-the-fly augmentation
* Length-preserving augmentation algorithms
* Runtime augmentation control (enable/disable)
* Comprehensive test suite
* PyTorch Lightning DataModule integration
* Support for multiple augmentations per sequence
* Configurable augmentation strategies

**Changed**:
* Replaced model wrapper with lightweight DataLoader approach
* Improved sequence length preservation across all augmentations
* Enhanced batch consistency and error handling
* Better integration with PyTorch ecosystem
* Optimized augmentation algorithms for performance

**Deprecated**:
* Old model wrapper approach (evoaug2.py)
* Previous augmentation implementations

**Removed**:
* Model wrapper classes
* Old augmentation implementations
* Deprecated utility functions

**Fixed**:
* Sequence length inconsistencies
* Batch shape mismatches
* Memory inefficiencies in augmentation
* Integration issues with PyTorch Lightning

**Security**:
* Improved input validation
* Better error handling for malformed inputs

[1.0.0] - 2023-XX-XX
---------------------

**Added**:
* Initial release of EvoAug
* Basic augmentation functionality
* Model wrapper approach
* Support for genomic sequence data

**Changed**:
* N/A (initial release)

**Deprecated**:
* N/A (initial release)

**Removed**:
* N/A (initial release)

**Fixed**:
* N/A (initial release)

**Security**:
* N/A (initial release)

Migration Guide
--------------

From Version 1.x to 2.0.0
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Breaking Changes**:

1. **Import Changes**:
   ```python
   # Old (v1.x)
   from evoaug2 import RobustModel
   
   # New (v2.0.0)
   from evoaug.evoaug import RobustLoader
   ```

2. **Usage Changes**:
   ```python
   # Old (v1.x)
   model = RobustModel(base_model, augmentations)
   trainer.fit(model, dataloader)
   
   # New (v2.0.0)
   loader = RobustLoader(dataset, augmentations)
   trainer.fit(model, train_dataloaders=loader)
   ```

3. **DataModule Integration**:
   ```python
   # Old (v1.x)
   # Required custom DataModule setup
   
   # New (v2.0.0)
   class AugmentedDataModule(pl.LightningDataModule):
       def train_dataloader(self):
           return RobustLoader(self.dataset, self.augmentations)
   ```

**Migration Steps**:

1. Update imports to use new module structure
2. Replace RobustModel with RobustLoader
3. Update training loops to use new DataLoader approach
4. Modify DataModule implementations if needed
5. Test augmentation functionality with new implementation

**Benefits of Migration**:

* Better PyTorch Lightning integration
* Improved performance and memory efficiency
* More flexible augmentation control
* Better error handling and debugging
* Enhanced testability and maintainability

Version Compatibility
--------------------

**Python Versions**:
* Python 3.8+ (recommended: 3.9+)

**PyTorch Versions**:
* PyTorch 1.9.0+ (recommended: 1.12.0+)

**PyTorch Lightning Versions**:
* PyTorch Lightning 1.5.0+ (recommended: 2.0.0+)

**Other Dependencies**:
* NumPy 1.20.0+
* SciPy 1.7.0+
* scikit-learn 1.0.0+
* h5py 3.1.0+

Support Policy
-------------

**Version Support**:
* Current major version: Full support
* Previous major version: Bug fixes only
* Older versions: No support

**Long Term Support (LTS)**:
* Version 2.0.0: LTS until 2025-12-31
* Security updates and critical bug fixes
* No new features

**End of Life**:
* Version 1.x: End of life (2024-01-01)
* No further updates or support

For questions about version compatibility or migration, please:
* Check the documentation
* Open a GitHub issue
* Start a GitHub discussion 