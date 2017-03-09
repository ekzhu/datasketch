import struct
from hashlib import sha1
import numpy as np

from datasketch import MinHash

class LeanMinHash(MinHash):

    __slots__ = ('seed', 'hashvalues', 'hashobj')

    def _initialize_slots(self, seed, hashvalues, hashobj):
        self.seed = seed
        self.hashvalues = self._parse_hashvalues(hashvalues)
        self.hashobj = hashobj

    def __init__(self, minhash):
        self._initialize_slots(minhash.seed, minhash.hashvalues, minhash.hashobj)

    def copy(self):
        '''
        Create a copy of this LeanMinHash by exporting its state.
        '''
        imh = object.__new__(LeanMinHash)
        imh._initialize_slots(*self.__slots__)
        return imh

    def update(self, b):
        raise TypeError("Cannot update a LeanMinHash")

    @classmethod
    def deserialize(cls, buf):
        '''
        Reconstruct a LeanMinHash from a byte buffer.
        This is more efficient than using the pickle.loads on the pickled
        bytes.
        '''
        try:
            seed, num_perm = struct.unpack_from('qi', buf, 0)
        except TypeError:
            seed, num_perm = struct.unpack_from('qi', buffer(buf), 0)
        offset = struct.calcsize('qi')
        try:
            hashvalues = struct.unpack_from('%dI' % num_perm, buf, offset)
        except TypeError:
            hashvalues = struct.unpack_from('%dI' % num_perm, buffer(buf), offset)
        imh = object.__new__(LeanMinHash)
        imh._initialize_slots(seed, hashvalues, sha1)
        return imh

    def __setstate__(self, buf):
        '''
        This function is called when unpickling the MinHash.
        Initialize the object with data in the buffer.
        Note that the input buffer is not the same as the input to the
        Python pickle.loads function.
        '''
        try:
            seed, num_perm = struct.unpack_from('qi', buf, 0)
        except TypeError:
            seed, num_perm = struct.unpack_from('qi', buffer(buf), 0)
        offset = struct.calcsize('qi')
        try:
            hashvalues = struct.unpack_from('%dI' % num_perm, buf, offset)
        except TypeError:
            hashvalues = struct.unpack_from('%dI' % num_perm, buffer(buf), offset)
        self._initialize_slots(seed, hashvalues, sha1)


    @classmethod
    def union(cls, *mhs):
        '''
        Return the union LeanMinHash of multiple MinHash
        '''
        if len(mhs) < 2:
            raise ValueError("Cannot union less than 2 MinHash")
        num_perm = len(mhs[0])
        seed = mhs[0].seed
        hashobj = mhs[0].hashobj
        if any((seed, num_perm, hashobj) != (m.seed, len(m), m.hashobj) for m in mhs):
            raise ValueError("The unioning MinHash must have the\
                    same seed, number of permutation functions and hashobj")
        hashvalues = np.minimum.reduce([m.hashvalues for m in mhs])

        imh = object.__new__(LeanMinHash)
        imh._initialize_slots(seed, hashvalues, hashobj)
        return imh