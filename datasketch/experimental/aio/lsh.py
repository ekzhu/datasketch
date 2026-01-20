"""Deprecated experimental aio lsh module.

.. deprecated::
    The `datasketch.experimental.aio.lsh` module is deprecated and will be removed in a future version.
    Please use `datasketch.aio.lsh` instead:

    Old: ``from datasketch.experimental.aio.lsh import AsyncMinHashLSH``
    New: ``from datasketch.aio import AsyncMinHashLSH``
"""

import warnings

warnings.warn(
    "datasketch.experimental.aio.lsh is deprecated. "
    "Use 'from datasketch.aio import AsyncMinHashLSH' instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from new location for backward compatibility
from datasketch.aio.lsh import (
    AsyncMinHashLSH,
    AsyncMinHashLSHDeleteSession,
    AsyncMinHashLSHInsertionSession,
)

__all__ = [
    "AsyncMinHashLSH",
    "AsyncMinHashLSHInsertionSession",
    "AsyncMinHashLSHDeleteSession",
]
