import struct
import numpy as np

from datasketch import MinHash

class LeanMinHash(MinHash):
    '''LeanMinHash is a MinHash which doesn't store the permutations and the 
    hashobj needed for updating. It trades the update() functionality for a faster
    deserialization and a smaller memory footprint. If a MinHash won't need further updates
    and needs to be serialized, create a LeanMinHash out of it and serialize that instead.
    
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

    def copy(self):
        '''
        Returns:
            datasketch.LeanMinHash: A copy of this LeanMinHash by exporting its
                state.
        '''
        lmh = object.__new__(LeanMinHash)
        lmh._initialize_slots(*self.__slots__)
        return lmh

    def update(self, b):
        '''This method is not available on a LeanMinHash.
        '''
        raise TypeError("Cannot update a LeanMinHash")

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
        '''Create a LeanMinHash which is the union of the LeanMinHash objects passed as arguments.

        Args:
            *mhs: The LeanMinHash objects to be united. The argument list length is variable,
                but must be at least 2.
        
        Returns:
            datasketch.LeanMinHash: A new union LeanMinHash.
        '''
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