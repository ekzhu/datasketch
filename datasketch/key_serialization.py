"""Restricted serialization helpers for LSH keys stored in backends."""

from __future__ import annotations

import io
import pickle
import pickletools
from collections.abc import Hashable
from typing import Any

_FORBIDDEN_OPCODES = {
    "BINPERSID",
    "BUILD",
    "EXT1",
    "EXT2",
    "EXT4",
    "GLOBAL",
    "INST",
    "NEWOBJ",
    "NEWOBJ_EX",
    "OBJ",
    "PERSID",
    "REDUCE",
    "STACK_GLOBAL",
}


def _is_hashable(key: Any) -> bool:
    """Return whether ``key`` can actually be hashed.

    ``isinstance(key, Hashable)`` is insufficient for containers such as a
    tuple that contains a list: the tuple exposes ``__hash__`` but calling it
    still raises ``TypeError``.
    """
    if not isinstance(key, Hashable):
        return False
    try:
        hash(key)
    except TypeError:
        return False
    return True


class _RestrictedUnpickler(pickle.Unpickler):
    """Unpickler that forbids importing every global callable and class."""

    def find_class(self, module: str, name: str) -> Any:
        global_name = f"{module}.{name}"
        raise pickle.UnpicklingError(f"global {global_name!r} is forbidden in an LSH key")

    def persistent_load(self, pid: Any) -> Any:
        raise pickle.UnpicklingError("persistent IDs are forbidden in an LSH key")


def loads_key(payload: bytes) -> Hashable:
    """Load a backend key without allowing global object construction."""
    try:
        for opcode, _argument, _position in pickletools.genops(payload):
            if opcode.name in _FORBIDDEN_OPCODES:
                raise pickle.UnpicklingError(f"opcode {opcode.name!r} is forbidden in an LSH key")
        key = _RestrictedUnpickler(io.BytesIO(payload)).load()
    except (EOFError, AttributeError, TypeError, ValueError) as error:
        raise pickle.UnpicklingError("invalid serialized LSH key") from error
    if not _is_hashable(key):
        raise pickle.UnpicklingError("serialized LSH key is not hashable")
    return key


def dumps_key(key: Hashable) -> bytes:
    """Serialize a key and ensure the restricted loader can read it back."""
    if not _is_hashable(key):
        raise TypeError("LSH keys must be hashable")
    payload = pickle.dumps(key)
    try:
        loads_key(payload)
    except pickle.UnpicklingError as error:
        raise TypeError(
            "prepickle=True supports keys made from primitive built-in types only; "
            f"custom type {type(key).__name__!r} cannot be stored safely"
        ) from error
    return payload
