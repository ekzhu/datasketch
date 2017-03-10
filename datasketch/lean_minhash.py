import struct
import numpy as np

from datasketch import MinHash

class LeanMinHash(MinHash):
    '''Lean MinHash is MinHash with a smaller memory footprint
    and faster deserialization, but with its internal state frozen
    -- no `update()`.

    Lean MinHash inherits all methods from :class:`datasketch.MinHash`.
    It does not store the `permutations` and the `hashobj` needed for updating.
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
        Lean MinHash can also be used in :class:`datasketch.MinHashLSH`
        and :class:`datasketch.MinHashLSHForest`.
    
    Args:
        MinHash: The MinHash object used to initialize the LeanMinHash.
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

    @classmethod
    def deserialize(cls, buf):
        try:
            seed, num_perm = struct.unpack_from('qi', buf, 0)
        except TypeError:
            seed, num_perm = struct.unpack_from('qi', buffer(buf), 0)
        offset = struct.calcsize('qi')
        try:
            hashvalues = struct.unpack_from('%dI' % num_perm, buf, offset)
        except TypeError:
            hashvalues = struct.unpack_from('%dI' % num_perm, buffer(buf), offset)
        lmh = object.__new__(LeanMinHash)
        lmh._initialize_slots(seed, hashvalues)
        return lmh

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


    @classmethod
    def union(cls, *lmhs):
        if len(lmhs) < 2:
            raise ValueError("Cannot union less than 2 MinHash")
        num_perm = len(lmhs[0])
        seed = lmhs[0].seed
        if any((seed != m.seed or num_perm != len(m)) for m in lmhs):
            raise ValueError("The unioning MinHash must have the\
                    same seed, number of permutation functions and hashobj")
        hashvalues = np.minimum.reduce([m.hashvalues for m in lmhs])

        lmh = object.__new__(LeanMinHash)
        lmh._initialize_slots(seed, hashvalues)
        return lmh
