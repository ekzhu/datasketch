from __future__ import annotations

import struct
from collections.abc import Iterable
from typing import Optional

import numpy as np

from datasketch.minhash import (
    _SCHEME_AFFINE32,
    _SCHEME_AFFINE64,
    _SCHEME_CODES,
    _SCHEME_CODES_INV,
    _SCHEME_LEGACY,
    _VALID_SCHEMES,
    MinHash,
)

# Byte-format notes: legacy payloads have no scheme field and are identified
# by a non-negative number-of-hash-values field, while the affine formats
# store the negated number followed by a scheme code byte. This keeps legacy
# sketches bit-identical to (and readable by) versions before 2.0.0.
# struct format character of one hash value, per scheme.
_SCHEME_VALUE_FMTS = {
    _SCHEME_LEGACY: "I",
    _SCHEME_AFFINE32: "I",
    _SCHEME_AFFINE64: "Q",
}


class LeanMinHash(MinHash):
    """Lean MinHash is MinHash with a smaller memory footprint
    and faster deserialization, but with its internal state frozen
    -- no `update()`.

    Lean MinHash inherits all methods from :class:`datasketch.MinHash`.
    It does not store the `permutations` and the `hashfunc` needed for updating.
    If a MinHash does not need further updates, convert it into a lean MinHash
    to save memory.

    Example:
        To create a lean MinHash from an existing MinHash:

        .. code-block:: python

            lean_minhash = LeanMinHash(minhash)

            # You can compute the Jaccard similarity between two lean MinHash
            lean_minhash.jaccard(lean_minhash2)

            # Or between a lean MinHash and a MinHash
            lean_minhash.jaccard(minhash2)

        To create a lean MinHash from the hash values, seed, and scheme of an
        existing MinHash:

        .. code-block:: python

            lean_minhash = LeanMinHash(
                seed=minhash.seed,
                hashvalues=minhash.hashvalues,
                scheme=minhash.scheme,
            )

        To create a MinHash from a lean MinHash:

        .. code-block:: python

            minhash = MinHash(
                seed=lean_minhash.seed,
                hashvalues=lean_minhash.hashvalues,
                scheme=lean_minhash.scheme,
            )

            # Or if you want to prevent further updates on minhash
            # from affecting the state of lean_minhash
            minhash = MinHash(
                seed=lean_minhash.seed,
                hashvalues=lean_minhash.digest(),
                scheme=lean_minhash.scheme,
            )

    Note:
        Lean MinHash can also be used in :class:`datasketch.MinHashLSH`,
        :class:`datasketch.MinHashLSHForest`, and :class:`datasketch.MinHashLSHEnsemble`.

    Args:
        minhash (optional): The :class:`datasketch.MinHash` object used to
            initialize the LeanMinHash. If this is not set, then `seed`
            and `hashvalues` must be set.
        seed (optional): The random seed that controls the set of random
            permutation functions generated for this LeanMinHash. This parameter
            must be used together with `hashvalues`.
        hashvalues (optional): The hash values used to inititialize the state
            of the LeanMinHash. This parameter must be used together with
            `seed`.
        scheme (optional): The permutation scheme of the MinHash the
            `hashvalues` were taken from. Required when initializing from
            `seed` and `hashvalues` (use ``"legacy"`` for hash values created
            by datasketch before 2.0.0), because hash values carry no trace
            of the scheme that produced them. When `minhash` is set the
            scheme is taken from the MinHash object instead, and this
            argument may only repeat it.

    """

    __slots__ = ("hashvalues", "scheme", "seed")

    def _initialize_slots(self, seed, hashvalues, scheme=_SCHEME_LEGACY):
        """Initialize the slots of the LeanMinHash.

        Args:
            seed (int): The random seed controls the set of random
                permutation functions generated for this LeanMinHash.
            hashvalues (Iterable): The hash values is the internal state of the LeanMinHash.
            scheme (str): The permutation scheme of the hash values.

        """
        if scheme not in _VALID_SCHEMES:
            raise ValueError("scheme must be one of %s, got %r" % (", ".join(_VALID_SCHEMES), scheme))
        self.seed = seed
        self.scheme = scheme
        self.hashvalues = self._parse_hashvalues(hashvalues)
        if scheme != _SCHEME_LEGACY and len(self.hashvalues) == 0:
            # An empty sketch would serialize with a hash value count of 0,
            # which the deserializer cannot tell apart from the legacy format
            # (identified by a non-negative count).
            raise ValueError("hashvalues must not be empty")

    def __init__(
        self,
        minhash: MinHash = None,
        seed: Optional[int] = None,
        hashvalues: Optional[Iterable] = None,
        scheme: Optional[str] = None,
    ):
        if minhash is not None:
            if scheme is not None and scheme != minhash.scheme:
                raise ValueError(
                    "scheme %r conflicts with the scheme %r of the given MinHash" % (scheme, minhash.scheme)
                )
            self._initialize_slots(minhash.seed, minhash.hashvalues, minhash.scheme)
        elif hashvalues is not None and seed is not None:
            if scheme is None:
                # Hash values carry no trace of the scheme that produced
                # them, so a default here would silently mislabel pre-2.0.0
                # values and defeat the cross-scheme comparison guards.
                raise ValueError(
                    "scheme must be specified explicitly when initializing from existing "
                    "hash values: pass the scheme of the MinHash they came from, or "
                    "scheme='legacy' for hash values created by datasketch before 2.0.0."
                )
            self._initialize_slots(seed, hashvalues, scheme)
        else:
            raise ValueError(
                "Init parameters cannot be None: make sure to set either minhash or both of hash values and seed"
            )

    def update(self, b) -> None:
        """Not available on a LeanMinHash.
        Calling it raises a TypeError.
        """
        raise TypeError("Cannot update a LeanMinHash")

    def copy(self) -> LeanMinHash:
        lmh = object.__new__(LeanMinHash)
        lmh._initialize_slots(self.seed, self.hashvalues, self.scheme)
        return lmh

    def _value_fmt(self) -> str:
        return _SCHEME_VALUE_FMTS[self.scheme]

    def bytesize(self, byteorder="@") -> int:
        """Compute the byte size after serialization.

        Args:
            byteorder (str, optional): This is byte order of the serialized data. Use one
                of the `byte order characters
                <https://docs.python.org/3/library/struct.html#byte-order-size-and-alignment>`_:
                ``@``, ``=``, ``<``, ``>``, and ``!``.
                Default is ``@`` -- the native order.

        Returns:
            int: Size in number of bytes after serialization.

        """
        if self.scheme == _SCHEME_LEGACY:
            # 8 bytes for the seed, 4 bytes for the number of hash values,
            # and 4 bytes for each hash value.
            return struct.calcsize("%sqi%dI" % (byteorder, len(self)))
        # The affine formats add a 1-byte scheme code after the number of
        # hash values, and store each hash value in 4 ("affine32") or
        # 8 ("affine64") bytes.
        return struct.calcsize("%sqiB%d%s" % (byteorder, len(self), self._value_fmt()))

    def serialize(self, buf, byteorder="@") -> None:
        """Serialize this lean MinHash and store the result in an allocated buffer.

        Args:
            buf (buffer): `buf` must implement the `buffer`_ interface.
                One such example is the built-in `bytearray`_ class.
            byteorder (str, optional): This is byte order of the serialized data. Use one
                of the `byte order characters
                <https://docs.python.org/3/library/struct.html#byte-order-size-and-alignment>`_:
                ``@``, ``=``, ``<``, ``>``, and ``!``.
                Default is ``@`` -- the native order.

        This is preferred over using `pickle`_ if the serialized lean MinHash needs
        to be used by another program in a different programming language.

        The serialization schema for the ``"legacy"`` scheme (identical to
        versions before 2.0.0):
            1. The first 8 bytes is the seed integer
            2. The next 4 bytes is the number of hash values
            3. The rest is the serialized hash values, each uses 4 bytes

        The serialization schema for the affine schemes:
            1. The first 8 bytes is the seed integer
            2. The next 4 bytes is the **negated** number of hash values
               (a negative value marks the post-2.0.0 format)
            3. The next byte is the scheme code (1 for ``"affine32"``,
               2 for ``"affine64"``)
            4. The rest is the serialized hash values, each uses 4 bytes
               for ``"affine32"`` and 8 bytes for ``"affine64"``

        Example:
            To serialize a single lean MinHash into a `bytearray`_ buffer.

            .. code-block:: python

                buf = bytearray(lean_minhash.bytesize())
                lean_minhash.serialize(buf)

            To serialize multiple lean MinHash into a `bytearray`_ buffer.

            .. code-block:: python

                # assuming lean_minhashs is a list of LeanMinHash with the same size
                size = lean_minhashs[0].bytesize()
                buf = bytearray(size * len(lean_minhashs))
                for i, lean_minhash in enumerate(lean_minhashs):
                    lean_minhash.serialize(buf[i * size :])

        .. _`buffer`: https://docs.python.org/3/c-api/buffer.html
        .. _`bytearray`: https://docs.python.org/3.6/library/functions.html#bytearray
        .. _`byteorder`: https://docs.python.org/3/library/struct.html

        """
        if len(buf) < self.bytesize(byteorder):
            raise ValueError(
                "The buffer does not have enough space for holding this MinHash."
            )
        if self.scheme == _SCHEME_LEGACY:
            fmt = "%sqi%dI" % (byteorder, len(self))
            struct.pack_into(fmt, buf, 0, self.seed, len(self), *self.hashvalues)
        else:
            fmt = "%sqiB%d%s" % (byteorder, len(self), self._value_fmt())
            struct.pack_into(fmt, buf, 0, self.seed, -len(self), _SCHEME_CODES[self.scheme], *self.hashvalues)

    @classmethod
    def deserialize(cls, buf, byteorder="@") -> LeanMinHash:
        """Deserialize a lean MinHash from a buffer.

        Buffers written by versions before 2.0.0 (which had no scheme field)
        deserialize with ``scheme="legacy"``.

        Args:
            buf (buffer): `buf` must implement the `buffer`_ interface.
                One such example is the built-in `bytearray`_ class.
            byteorder (str. optional): This is byte order of the serialized data. Use one
                of the `byte order characters
                <https://docs.python.org/3/library/struct.html#byte-order-size-and-alignment>`_:
                ``@``, ``=``, ``<``, ``>``, and ``!``.
                Default is ``@`` -- the native order.

        Return:
            datasketch.LeanMinHash: The deserialized lean MinHash

        Example:
            To deserialize a lean MinHash from a buffer.

            .. code-block:: python

                lean_minhash = LeanMinHash.deserialize(buf)

        """
        fmt_seed_size = "%sqi" % byteorder
        try:
            seed, num_perm = struct.unpack_from(fmt_seed_size, buf, 0)
        except TypeError:
            buf = memoryview(buf)
            seed, num_perm = struct.unpack_from(fmt_seed_size, buf, 0)
        if num_perm >= 0:
            scheme = _SCHEME_LEGACY
            offset = struct.calcsize(fmt_seed_size)
        else:
            num_perm = -num_perm
            (scheme_code,) = struct.unpack_from(byteorder + "B", buf, struct.calcsize(fmt_seed_size))
            if scheme_code not in _SCHEME_CODES_INV:
                raise ValueError("Unknown permutation scheme code: %d" % scheme_code)
            scheme = _SCHEME_CODES_INV[scheme_code]
            # The 0-count value entry aligns the offset without consuming data
            # (only relevant for the native byte order "@").
            offset = struct.calcsize("%sqiB0%s" % (byteorder, _SCHEME_VALUE_FMTS[scheme]))
        fmt_hash = "%s%d%s" % (byteorder, num_perm, _SCHEME_VALUE_FMTS[scheme])
        hashvalues = struct.unpack_from(fmt_hash, buf, offset)
        lmh = object.__new__(LeanMinHash)
        lmh._initialize_slots(seed, hashvalues, scheme)
        return lmh

    def __getstate__(self):
        buf = bytearray(self.bytesize())
        if self.scheme == _SCHEME_LEGACY:
            fmt = "qi%dI" % len(self)
            struct.pack_into(fmt, buf, 0, self.seed, len(self), *self.hashvalues)
        else:
            fmt = "qiB%d%s" % (len(self), self._value_fmt())
            struct.pack_into(fmt, buf, 0, self.seed, -len(self), _SCHEME_CODES[self.scheme], *self.hashvalues)
        return buf

    def __setstate__(self, buf):
        try:
            seed, num_perm = struct.unpack_from("qi", buf, 0)
        except TypeError:
            buf = memoryview(buf)
            seed, num_perm = struct.unpack_from("qi", buf, 0)
        if num_perm >= 0:
            scheme = _SCHEME_LEGACY
            offset = struct.calcsize("qi")
        else:
            num_perm = -num_perm
            (scheme_code,) = struct.unpack_from("B", buf, struct.calcsize("qi"))
            if scheme_code not in _SCHEME_CODES_INV:
                raise ValueError("Unknown permutation scheme code: %d" % scheme_code)
            scheme = _SCHEME_CODES_INV[scheme_code]
            offset = struct.calcsize("qiB0%s" % _SCHEME_VALUE_FMTS[scheme])
        hashvalues = struct.unpack_from("%d%s" % (num_perm, _SCHEME_VALUE_FMTS[scheme]), buf, offset)
        self._initialize_slots(seed, hashvalues, scheme)

    def __hash__(self) -> int:
        return hash((self.scheme, self.seed, tuple(self.hashvalues)))

    @classmethod
    def union(cls, *lmhs: LeanMinHash) -> LeanMinHash:
        """Create a new lean MinHash by unioning multiple lean MinHash."""
        if len(lmhs) < 2:
            raise ValueError("Cannot union less than 2 MinHash")
        num_perm = len(lmhs[0])
        seed = lmhs[0].seed
        scheme = lmhs[0].scheme
        if any((seed != m.seed or num_perm != len(m) or scheme != m.scheme) for m in lmhs):
            raise ValueError(
                "The unioning MinHash must have the same seed, number of permutation functions and scheme."
            )
        hashvalues = np.minimum.reduce([m.hashvalues for m in lmhs])

        lmh = object.__new__(LeanMinHash)
        lmh._initialize_slots(seed, hashvalues, scheme)
        return lmh
