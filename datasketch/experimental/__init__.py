"""Deprecated experimental module.

.. deprecated::
    The `datasketch.experimental` module is deprecated and will be removed in a future version.
    Please use `datasketch.aio` instead:

    Old: ``from datasketch.experimental import AsyncMinHashLSH``
    New: ``from datasketch.aio import AsyncMinHashLSH``

    Or simply: ``from datasketch import AsyncMinHashLSH``
"""

import warnings

warnings.warn(
    "datasketch.experimental is deprecated. "
    "Use 'from datasketch.aio import AsyncMinHashLSH' or "
    "'from datasketch import AsyncMinHashLSH' instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from new location for backward compatibility
from datasketch.aio import AsyncMinHashLSH  # noqa: E402

__all__ = ["AsyncMinHashLSH"]
