"""Deprecated experimental aio module.

.. deprecated::
    The `datasketch.experimental.aio` module is deprecated and will be removed in a future version.
    Please use `datasketch.aio` instead:

    Old: ``from datasketch.experimental.aio import AsyncMinHashLSH``
    New: ``from datasketch.aio import AsyncMinHashLSH``
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Visible to static analyzers so they know `__all__` is satisfied.
    # Not imported at runtime - the real dispatch happens in __getattr__.
    from datasketch.aio import (
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
    # PEP 562: emit the warning lazily on attribute access so that merely
    # importing the parent package (e.g. as an intermediate step of
    # `from datasketch.experimental.aio.lsh import ...`) does not fire
    # a second, redundant warning. See the long comment in
    # datasketch/experimental/__init__.py for why we cache into globals().
    if name in _DEPRECATED:
        import warnings

        warnings.warn(
            "datasketch.experimental.aio is deprecated. Use 'from datasketch.aio import AsyncMinHashLSH' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        import datasketch.aio as _new

        value = getattr(_new, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
