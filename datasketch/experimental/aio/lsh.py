"""Deprecated experimental aio lsh module.

.. deprecated::
    The `datasketch.experimental.aio.lsh` module is deprecated and will be removed in a future version.
    Please use `datasketch.aio.lsh` instead:

    Old: ``from datasketch.experimental.aio.lsh import AsyncMinHashLSH``
    New: ``from datasketch.aio import AsyncMinHashLSH``
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Visible to static analyzers so they know `__all__` is satisfied.
    # Not imported at runtime - the real dispatch happens in __getattr__.
    from datasketch.aio.lsh import (
        AsyncMinHashLSH,
        AsyncMinHashLSHDeleteSession,
        AsyncMinHashLSHInsertionSession,
    )

__all__ = [
    "AsyncMinHashLSH",
    "AsyncMinHashLSHDeleteSession",
    "AsyncMinHashLSHInsertionSession",
]

_DEPRECATED = frozenset(__all__)


def __getattr__(name):
    # Lazy warning via PEP 562: fires exactly once per attribute access on the
    # deprecated module, and we cache the resolved symbol back into globals()
    # so the warning is emitted once per process (see the long comment in
    # datasketch/experimental/__init__.py for rationale).
    if name in _DEPRECATED:
        import warnings

        warnings.warn(
            "datasketch.experimental.aio.lsh is deprecated. Use 'from datasketch.aio import AsyncMinHashLSH' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        import datasketch.aio.lsh as _new

        value = getattr(_new, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
