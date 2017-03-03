'''
This module implements MinHash - a probabilistic data structure for computing
Jaccard similarity between datasets.

The original MinHash paper:
http://cs.brown.edu/courses/cs253/papers/nearduplicate.pdf
'''

import random, copy, struct
from hashlib import sha1
import numpy as np

# The size of a hash value in number of bytes
hashvalue_byte_size = len(bytes(np.int64(42).data))

# http://en.wikipedia.org/wiki/Mersenne_prime
_mersenne_prime = (1 << 61) - 1
_max_hash = (1 << 32) - 1
_hash_range = (1 << 32)

class MinHash(object):
    '''
    Create a MinHash with `num_perm` number of random permutation
    functions.
    The `seed` parameter controls the set of random permutation functions
    generated for this MinHash.
    Different seed will generate different sets of permutaiton functions.
    The `hashobj` parameter specifies a hash used for generating
    hash value. It must implements the `digest` interface similar to
    hashlib hashes.
    `hashvalues` and `permutations` can be specified for faster
    initialization using existing state from another MinHash.
    '''

    __slots__ = ('permutations', 'hashvalues', 'seed', 'hashobj')

    def __init__(self, num_perm=128, seed=1, hashobj=sha1,
            hashvalues=None, permutations=None):
        if num_perm > _hash_range:
            # Because 1) we don't want the size to be too large, and
            # 2) we are using 4 bytes to store the size value
            raise ValueError("Cannot have more than %d number of\
                    permutation functions" % _hash_range)
        self.seed = seed
        self.hashobj = hashobj
        # Initialize hash values
        if hashvalues is not None:
            self.hashvalues = self._parse_hashvalues(hashvalues)
        else:
            self.hashvalues = self._init_hashvalues(num_perm)
        # Initalize permutation function parameters
        if permutations is not None:
            self.permutations = permutations
        else:
            generator = random.Random()
            generator.seed(self.seed)
            # Create parameters for a random bijective permutation function
            # that maps a 32-bit hash value to another 32-bit hash value.
            # http://en.wikipedia.org/wiki/Universal_hashing
            self.permutations = np.array([(generator.randint(1, _mersenne_prime),
                                           generator.randint(0, _mersenne_prime))
                                          for _ in range(num_perm)], dtype=np.uint64).T
        if len(self) != len(self.permutations[0]):
            raise ValueError("Numbers of hash values and permutations mismatch")

    def _init_hashvalues(self, num_perm):
        return np.ones(num_perm, dtype=np.uint64)*_max_hash

    def _parse_hashvalues(self, hashvalues):
        return np.array(hashvalues, dtype=np.uint64)

    def __len__(self):
        '''
        Return the size of the MinHash
        '''
        return len(self.hashvalues)

    def __eq__(self, other):
        '''
        Check equivalence between MinHash
        '''
        return self.seed == other.seed and \
                np.array_equal(self.hashvalues, other.hashvalues)

    def is_empty(self):
        '''
        Check if the current MinHash is empty - at the state of just
        initialized.
        '''
        if np.any(self.hashvalues != _max_hash):
            return False
        return True

    def clear(self):
        '''
        Clear the current state of the Minhash.
        '''
        self.hashvalues = self._init_hashvalues(len(self))

    def copy(self):
        '''
        Create a copy of this MinHash by exporting its state.
        '''
        return MinHash(seed=self.seed, hashvalues=self.digest(),
                permutations=self.permutations)

    def update(self, b):
        '''
        Update the Minhash with a new data value in bytes.
        '''
        hv = struct.unpack('<I', self.hashobj(b).digest()[:4])[0]
        a, b = self.permutations
        phv = np.bitwise_and((a * hv + b) % _mersenne_prime, np.uint64(_max_hash))
        self.hashvalues = np.minimum(phv, self.hashvalues)

    def digest(self):
        '''
        Returns the hash values.
        '''
        return copy.copy(self.hashvalues)

    def merge(self, other):
        '''
        Merge the other MinHash with this one, making this the union
        of both.
        '''
        if other.seed != self.seed:
            raise ValueError("Cannot merge MinHash with\
                    different seeds")
        if len(self) != len(other):
            raise ValueError("Cannot merge MinHash with\
                    different numbers of permutation functions")
        self.hashvalues = np.minimum(other.hashvalues, self.hashvalues)

    def count(self):
        '''
        Estimate the cardinality count.
        See: http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=365694
        '''
        k = len(self)
        return np.float(k) / np.sum(self.hashvalues / np.float(_max_hash)) - 1.0

    def jaccard(self, other):
        '''
        Estimate the Jaccard similarity (resemblance) between this Minhash
        and the other.
        '''
        if other.seed != self.seed:
            raise ValueError("Cannot compute Jaccard given MinHash with\
                    different seeds")
        if len(self) != len(other):
            raise ValueError("Cannot compute Jaccard given MinHash with\
                    different numbers of permutation functions")
        return np.float(np.count_nonzero(self.hashvalues==other.hashvalues)) /\
                np.float(len(self))

    def bytesize(self):
        '''
        Returns the size of this MinHash in bytes.
        To be used in serialization.
        '''
        # Use 8 bytes to store the seed integer
        seed_size = struct.calcsize('q')
        # Use 4 bytes to store the number of hash values
        length_size = struct.calcsize('i')
        # Use 4 bytes to store each hash value as we are using the lower 32 bit
        hashvalue_size = struct.calcsize('I')
        return seed_size + length_size + len(self) * hashvalue_size

    def serialize(self, buf):
        '''
        Serializes this MinHash into bytes, store in `buf`.
        This is more efficient than using pickle.dumps on the object.
        '''
        if len(buf) < self.bytesize():
            raise ValueError("The buffer does not have enough space\
                    for holding this MinHash.")
        fmt = "qi%dI" % len(self)
        struct.pack_into(fmt, buf, 0,
                self.seed, len(self), *self.hashvalues)

    @classmethod
    def deserialize(cls, buf):
        '''
        Reconstruct a MinHash from a byte buffer.
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
        return cls(num_perm=num_perm, seed=seed, hashvalues=hashvalues)

    def __getstate__(self):
        '''
        This function is called when pickling the MinHash.
        Returns a bytearray which will then be pickled.
        Note that the bytes returned by the Python pickle.dumps is not
        the same as the buffer returned by this function.
        '''
        buf = bytearray(self.bytesize())
        fmt = "qi%dI" % len(self)
        struct.pack_into(fmt, buf, 0,
                self.seed, len(self), *self.hashvalues)
        return buf

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
        self.__init__(num_perm=num_perm, seed=seed, hashvalues=hashvalues)

    @classmethod
    def union(cls, *mhs):
        '''
        Return the union MinHash of multiple MinHash
        '''
        if len(mhs) < 2:
            raise ValueError("Cannot union less than 2 MinHash")
        num_perm = len(mhs[0])
        seed = mhs[0].seed
        if any(seed != m.seed for m in mhs) or \
                any(num_perm != len(m) for m in mhs):
            raise ValueError("The unioning MinHash must have the\
                    same seed and number of permutation functions")
        hashvalues = np.minimum.reduce([m.hashvalues for m in mhs])
        return cls(num_perm=num_perm, seed=seed, hashvalues=hashvalues)
