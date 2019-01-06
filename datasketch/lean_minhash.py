import struct
import numpy as np

from datasketch import MinHash

class LeanMinHash(MinHash):
    '''Lean MinHash is MinHash with a smaller memory footprint
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

        To create a MinHash from a lean MinHash:

        .. code-block:: python

            minhash = MinHash(seed=lean_minhash.seed,
                              hashvalues=lean_minhash.hashvalues)

            # Or if you want to prevent further updates on minhash
            # from affecting the state of lean_minhash
            minhash = MinHash(seed=lean_minhash.seed,
                              hashvalues=lean_minhash.digest())

    Note:
        Lean MinHash can also be used in :class:`datasketch.MinHashLSH`,
        :class:`datasketch.MinHashLSHForest`, and :class:`datasketch.MinHashLSHEnsemble`.

    Args:
        minhash: The :class:`datasketch.MinHash` object used to initialize the LeanMinHash.
    '''

    __slots__ = ('seed', 'hashvalues')

    def _initialize_slots(self, seed, hashvalues):
        '''Initialize the slots of the LeanMinHash.

        Args:
            seed (int): The random seed controls the set of random
                permutation functions generated for this LeanMinHash.
            hashvalues: The hash values is the internal state of the LeanMinHash.
        '''
        self.seed = seed
        self.hashvalues = self._parse_hashvalues(hashvalues)

    def __init__(self, minhash):
        self._initialize_slots(minhash.seed, minhash.hashvalues)

    def update(self, b):
        '''This method is not available on a LeanMinHash.
        Calling it raises a TypeError.
        '''
        raise TypeError("Cannot update a LeanMinHash")

    def copy(self):
        lmh = object.__new__(LeanMinHash)
        lmh._initialize_slots(*self.__slots__)
        return lmh

    def bytesize(self, byteorder='@'):
        '''Compute the byte size after serialization.

        Args:
            byteorder (str, optional): This is byte order of the serialized data. Use one
                of the `byte order characters
                <https://docs.python.org/3/library/struct.html#byte-order-size-and-alignment>`_:
                ``@``, ``=``, ``<``, ``>``, and ``!``.
                Default is ``@`` -- the native order.

        Returns:
            int: Size in number of bytes after serialization.
        '''
        # Use 8 bytes to store the seed integer
        seed_size = struct.calcsize(byteorder+'q')
        # Use 4 bytes to store the number of hash values
        length_size = struct.calcsize(byteorder+'i')
        # Use 4 bytes to store each hash value as we are using the lower 32 bit
        hashvalue_size = struct.calcsize(byteorder+'I')
        return seed_size + length_size + len(self) * hashvalue_size

    def serialize(self, buf, byteorder='@'):
        '''
        Serialize this lean MinHash and store the result in an allocated buffer.

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

        The serialization schema:
            1. The first 8 bytes is the seed integer
            2. The next 4 bytes is the number of hash values
            3. The rest is the serialized hash values, each uses 4 bytes

        Example:
            To serialize a single lean MinHash into a `bytearray`_ buffer.

            .. code-block:: python

                buf = bytearray(lean_minhash.bytesize())
                lean_minhash.serialize(buf)

            To serialize multiple lean MinHash into a `bytearray`_ buffer.

            .. code-block:: python

                # assuming lean_minhashs is a list of LeanMinHash with the same size
                size = lean_minhashs[0].bytesize()
                buf = bytearray(size*len(lean_minhashs))
                for i, lean_minhash in enumerate(lean_minhashs):
                    lean_minhash.serialize(buf[i*size:])

        .. _`buffer`: https://docs.python.org/3/c-api/buffer.html
        .. _`bytearray`: https://docs.python.org/3.6/library/functions.html#bytearray
        .. _`byteorder`: https://docs.python.org/3/library/struct.html
        '''
        if len(buf) < self.bytesize():
            raise ValueError("The buffer does not have enough space\
                    for holding this MinHash.")
        fmt = "%sqi%dI" % (byteorder, len(self))
        struct.pack_into(fmt, buf, 0,
                self.seed, len(self), *self.hashvalues)

    @classmethod
    def deserialize(cls, buf, byteorder='@'):
        '''
        Deserialize a lean MinHash from a buffer.

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
        '''
        fmt_seed_size = "%sqi" % byteorder
        fmt_hash = byteorder + "%dI"
        try:
            seed, num_perm = struct.unpack_from(fmt_seed_size, buf, 0)
        except TypeError:
            seed, num_perm = struct.unpack_from(fmt_seed_size, buffer(buf), 0)
        offset = struct.calcsize(fmt_seed_size)
        try:
            hashvalues = struct.unpack_from(fmt_hash % num_perm, buf, offset)
        except TypeError:
            hashvalues = struct.unpack_from(fmt_hash % num_perm, buffer(buf), offset)
        lmh = object.__new__(LeanMinHash)
        lmh._initialize_slots(seed, hashvalues)
        return lmh

    def __getstate__(self):
        buf = bytearray(self.bytesize())
        fmt = "qi%dI" % len(self)
        struct.pack_into(fmt, buf, 0,
                self.seed, len(self), *self.hashvalues)
        return buf

    def __setstate__(self, buf):
        try:
            seed, num_perm = struct.unpack_from('qi', buf, 0)
        except TypeError:
            seed, num_perm = struct.unpack_from('qi', buffer(buf), 0)
        offset = struct.calcsize('qi')
        try:
            hashvalues = struct.unpack_from('%dI' % num_perm, buf, offset)
        except TypeError:
            hashvalues = struct.unpack_from('%dI' % num_perm, buffer(buf), offset)
        self._initialize_slots(seed, hashvalues)

    def __hash__(self):
        return hash((self.seed, tuple(self.hashvalues)))

    @classmethod
    def union(cls, *lmhs):
        if len(lmhs) < 2:
            raise ValueError("Cannot union less than 2 MinHash")
        num_perm = len(lmhs[0])
        seed = lmhs[0].seed
        if any((seed != m.seed or num_perm != len(m)) for m in lmhs):
            raise ValueError("The unioning MinHash must have the\
                    same seed, number of permutation functions.")
        hashvalues = np.minimum.reduce([m.hashvalues for m in lmhs])

        lmh = object.__new__(LeanMinHash)
        lmh._initialize_slots(seed, hashvalues)
        return lmh
