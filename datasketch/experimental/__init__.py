"""Deprecated experimental module.

.. deprecated::
    The `datasketch.experimental` module is deprecated and will be removed in a future version.
    Please use `datasketch.aio` instead:

    Old: ``from datasketch.experimental import AsyncMinHashLSH``
    New: ``from datasketch.aio import AsyncMinHashLSH``

    Or simply: ``from datasketch import AsyncMinHashLSH``
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Visible to static analyzers so they know `__all__` is satisfied.
    # Not imported at runtime - the real dispatch happens in __getattr__.
    from datasketch.aio import AsyncMinHashLSH

__all__ = ["AsyncMinHashLSH"]


def __getattr__(name):
    # PEP 562: only emit the DeprecationWarning when the user actually pulls a
    # symbol out of this package, not on every `import datasketch.experimental`.
    # This avoids the noisy triple-warning that fired when each intermediate
    # __init__.py warned eagerly.
    #
    # We cache the resolved symbol back into globals() so subsequent accesses
    # bypass __getattr__. This matters for two reasons:
    #   1. `from pkg import x` internally performs both `hasattr(pkg, x)` and
    #      `getattr(pkg, x)`, so without caching __getattr__ fires twice.
    #   2. It makes the warning a one-shot per process, which is the normal
    #      expectation for deprecation warnings.
    if name == "AsyncMinHashLSH":
        import warnings

        warnings.warn(
            "datasketch.experimental is deprecated. "
            "Use 'from datasketch.aio import AsyncMinHashLSH' or "
            "'from datasketch import AsyncMinHashLSH' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        from datasketch.aio import AsyncMinHashLSH

        globals()[name] = AsyncMinHashLSH
        return AsyncMinHashLSH
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
