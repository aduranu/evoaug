"""Backward-compatibility shim for evoaug <= 1.0.6.

The v1 API previously lived in ``evoaug/evoaug.py``. It has moved to
:mod:`evoaug.legacy` in v2.0; this module re-exports the same names so
``from evoaug.evoaug import RobustModel`` (as written in the v1 README
and many downstream notebooks) continues to work.

A :class:`DeprecationWarning` is raised on import to nudge users toward
the new location.
"""

import warnings as _warnings

_warnings.warn(
    "evoaug.evoaug is deprecated in v2.0 and will be removed in a future "
    "release. Import the legacy v1 API from evoaug.legacy instead, or "
    "migrate to evoaug.RobustLoader.",
    DeprecationWarning,
    stacklevel=2,
)

from .legacy.evoaug import (  # noqa: E402, F401
    RobustModel,
    load_model_from_checkpoint,
    augment_max_len,
)

__all__ = ["RobustModel", "load_model_from_checkpoint", "augment_max_len"]
