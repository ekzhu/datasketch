'''
This module implements MinHash - a probabilistic data structure for computing
Jaccard similarity between datasets.

The original MinHash paper:
http://cs.brown.edu/courses/cs253/papers/nearduplicate.pdf
'''

import random, struct, math

# http://en.wikipedia.org/wiki/Mersenne_prime
_mersenne_prime = (1 << 61) - 1
_max_hash = (1 << 32) - 1
_hash_range = (1 << 32)


def _create_permutation():
    '''
    Create parameters for a random bijective permutation function
    that maps a 32-bit hash value to another 32-bit hash value.
    http://en.wikipedia.org/wiki/Universal_hashing
    '''
    a = random.randint(1, _max_hash)
    b = random.randint(0, _max_hash)
    return (a, b)


_permutation_func = lambda x, a, b: ((a * x + b) % _mersenne_prime) % _hash_range


class MinHash(object):
    '''
    The MinHash object.
    '''

    __slots__ = ('permutations', 'hashvalues', 'seed')

    def __init__(self, num_perm=128, seed=1):
        '''
        Create a MinHash object with `num_perm` number of random
        permutation functions.
        The `seed` parameter controls the set of random permutation functions
        generated for this MinHash object.
        Different seed will generate different sets of permutaiton functions.
        '''
        if num_perm <= 0:
            raise MinHashException("Cannot have non-positive number of\
                    permutation functions")
        if num_perm > _hash_range:
            # Because 1) we don't want the size to be too large, and
            # 2) we are using 4 bytes to store the size value
            raise MinHashException("Cannot have more than %d number of\
                    permutation functions" % _hash_range)
        self.hashvalues = [_max_hash for _ in range(num_perm)]
        self.seed = seed
        random.seed(self.seed)
        self.permutations = [_create_permutation() for _ in range(num_perm)]
    
    def is_empty(self):
        '''
        Check if the current MinHash object is empty - at the state of just
        initialized.
        '''
        if any(v != _max_hash for v in self.hashvalues):
            return False
        return True

    def digest(self, hashobj):
        '''
        Digest a hash object that implemented `digest` as in hashlib,
        and has size at least 4 bytes.
        '''
        # Digest the hash object to get the hash value
        hv = struct.unpack('<I', hashobj.digest()[:4])[0]
        for i, (a, b) in enumerate(self.permutations):
            phv = _permutation_func(hv, a, b)
            if phv < self.hashvalues[i]:
                self.hashvalues[i] = phv

    def merge(self, other):
        '''
        Merge the other MinHash object with this one, making this the union
        of both.
        '''
        if other.seed != self.seed:
            raise MinHashException("Cannot merge MinHash objects with\
                    different seeds")
        if len(other.permutations) != len(self.permutations):
            raise MinHashException("Cannot merge MinHash objects with\
                    different numbers of permutation functions")
        for i, v in enumerate(other.hashvalues):
            if v < self.hashvalues[i]:
                self.hashvalues[i] = v

    def count(self):
        '''
        Estimate the cardinality count.
        See: http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=365694 
        '''
        k = len(self.hashvalues)
        return float(k) / sum(float(v)/float(_max_hash) for v in self.hashvalues) - 1.0

    def bytesize(self):
        '''
        Returns the size of this MinHash object in bytes.
        To be used in serialization.
        '''
        # Use 8 bytes to store the seed integer
        seed_size = struct.calcsize('q')
        # Use 4 bytes to store the number of hash values
        length_size = struct.calcsize('i')
        # Use 4 bytes to store each hash value as we are using 32 bit
        hashvalue_size = struct.calcsize('I')
        return seed_size + length_size + len(self.hashvalues) * hashvalue_size

    def serialize(self, buffer):
        '''
        Serializes this MinHash object into bytes, store in `buffer`.
        This is more efficient than using pickle.dumps on the object.
        '''
        if len(buffer) < self.bytesize():
            raise MinHashException("The buffer does not have enough space\
                    for holding this MinHash object.")
        fmt = "qi%dI" % len(self.hashvalues)
        struct.pack_into(fmt, buffer, 0,
                self.seed, len(self.hashvalues), *self.hashvalues)

    @classmethod
    def deserialize(cls, buffer):
        '''
        Reconstruct a MinHash object from a byte buffer.
        This is more efficient than using the pickle.loads on the pickled
        bytes.
        '''
        seed, num_perm = struct.unpack_from('qi', buffer, 0)
        mh = cls(num_perm=num_perm, seed=seed)
        offset = struct.calcsize('qi')
        for i in range(num_perm):
            mh.hashvalues[i] = struct.unpack_from('I', buffer, offset)[0]
            offset += struct.calcsize('I')
        return mh

    def __getstate__(self):
        '''
        This function is called when pickling the MinHash object.
        Returns a bytearray which will then be pickled.
        Note that the bytes returned by the Python pickle.dumps is not
        the same as the buffer returned by this function.
        '''
        buffer = bytearray(self.bytesize())
        self.serialize(buffer)
        return buffer

    def __setstate__(self, buffer):
        '''
        This function is called when unpickling the MinHash object.
        Initialize the object with data in the buffer.
        Note that the input buffer is not the same as the input to the
        Python pickle.loads function.
        '''
        seed, num_perm = struct.unpack_from('qi', buffer, 0)
        self.__init__(num_perm=num_perm, seed=seed)
        offset = struct.calcsize('qi')
        for i in range(num_perm):
            self.hashvalues[i] = struct.unpack_from('I', buffer, offset)[0]
            offset += struct.calcsize('I')

    @classmethod
    def union(cls, *mhs):
        '''
        Return the union MinHash of multiple MinHash objects
        '''
        if len(mhs) < 2:
            raise MinHashException("Cannot union less than 2 MinHash sketches")
        num_perm = len(mhs[0].permutations)
        seed = mhs[0].seed
        if any(seed != m.seed for m in mhs) or \
                any(num_perm != len(m.permutations) for m in mhs):
            raise MinHashException("The unioning MinHash objects must have the\
                    same seed and number of permutation functions")
        mh = cls(num_perm=num_perm, seed=seed)
        mh.hashvalues = [min(*vs) for vs in zip(*[m.hashvalues for m in mhs])]
        return mh

    def __eq__(self, other):
        '''
        Check equivalence between MinHash objects
        '''
        if self.seed != other.seed:
            return False
        if len(self.permutations) != len(other.permutations) or\
                len(self.hashvalues) != len(other.hashvalues):
            return False
        if any(t1 != t2 for t1, t2 in zip(self.permutations, other.permutations)):
            return False
        if any(v1 != v2 for v1, v2 in zip(self.hashvalues, other.hashvalues)):
            return False
        return True


def jaccard(*mhs):
    '''
    Compute Jaccard similarity measure for multiple of MinHash objects.
    '''
    if len(mhs) < 2:
        raise MinHashException("Less than 2 MinHash objects were given")
    seed = mhs[0].seed
    if any(seed != m.seed for m in mhs):
        raise MinHashException("Cannot compare MinHash objects with\
                different seeds")
    num_perm = len(mhs[0].permutations)
    if any(num_perm != len(m.permutations) for m in mhs):
        raise MinHashException("Cannot compare MinHash objects with\
                different numbers of permutation functions")
    intersection = 0
    for i in range(num_perm):
        phv = mhs[0].hashvalues[i]
        if all(phv == m.hashvalues[i] for m in mhs):
            intersection += 1
    return float(intersection) / float(num_perm)


class MinHashException(Exception):
    pass
