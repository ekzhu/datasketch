'''
This module implements MinHash - a probabilistic data structure for computing
Jaccard similarity between datasets.

The original MinHash paper:
http://cs.brown.edu/courses/cs253/papers/nearduplicate.pdf
'''

import random, struct

# http://en.wikipedia.org/wiki/Mersenne_prime
_mersenne_prime = (1 << 61) - 1
_max_hash = (1 << 32) - 1
_hash_range = (1 << 32)


def _create_permutation():
    '''
    Create a random bijective permutation function that maps a 32-bit
    hash value to another 32-bit hash value.
    http://en.wikipedia.org/wiki/Universal_hashing
    '''
    a = random.randint(1, _max_hash)
    b = random.randint(0, _max_hash)
    return lambda x : ((a * x + b) % _mersenne_prime) % _hash_range


class MinHash(object):
    '''
    The MinHash object.
    '''

    __slots__ = ('permutations', 'hashvalues', 'seed')

    def __init__(self, seed, num_perm):
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

    def digest(self, hashobj):
        '''
        Digest a hash object that implemented `hexdigest` as in hashlib,
        and has size at least 4 bytes.
        '''
        # Digest the hash object to get the hash value
        hv = int(hashobj.hexdigest()[:8], 16)
        for i, p in enumerate(self.permutations):
            phv = p(hv)
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


    def serialize(self, buffer, offset):
        '''
        Serializes this MinHash object into bytes, store in `buffer`
        starting at `offset` position.
        The size of `buffer` must equal to the size returned by
        the `bytesize` method.
        '''
        if len(buffer) - offset < self.bytesize():
            raise MinHashException("The buffer does not have enough space\
                    for holding this MinHash object.")
        fmt = "qi%dI" % len(self.hashvalues)
        struct.pack_into(fmt, buffer, offset,
                self.seed, len(self.hashvalues), *self.hashvalues)

    @classmethod
    def deserialize(cls, buffer, offset):
        '''
        Reconstruct a MinHash object from a byte buffer starting at `offset`.
        '''
        seed, num_perm = struct.unpack_from('qi', buffer, offset)
        mh = cls(seed, num_perm)
        offset = offset + struct.calcsize('qi')
        for i in range(num_perm):
            mh.hashvalues[i] = struct.unpack_from('I', buffer, offset)[0]
            offset += struct.calcsize('I')
        return mh


def jaccard(mhs):
    '''
    Compute Jaccard similarity measure for a list of MinHash objects.
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
