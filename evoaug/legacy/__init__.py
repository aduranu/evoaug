"""Legacy v1 API: PyTorch Lightning model-wrapper interface.

These names were the public API of evoaug <= 1.0.6. They remain available
for backward compatibility but new code should use ``evoaug.RobustLoader``.
"""

from .evoaug import RobustModel, load_model_from_checkpoint, augment_max_len

__all__ = ["RobustModel", "load_model_from_checkpoint", "augment_max_len"]
