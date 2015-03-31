'''
Implments the MinHash algorithm.
'''

import random

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
    The MinHash signature
    '''

    __slots__ = ('permutations', 'hashvalues', 'seed')

    def __init__(self, seed, num_perm):
        if num_perm <= 0:
            raise MinHashException("Cannot have non-positive number of\
                    permutation functions")
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

    def serialize(self):
        '''
        Serializes into bytes
        '''
        pass

    def deserialize(serialized):
        '''
        Reconstruct and reset this object from a serialized one.
        '''
        pass


def jaccard(minhashs):
    '''
    Compute Jaccard similarity measure for a list of MinHash objects.
    '''
    if len(minhashs) < 2:
        raise MinHashException("Less than 2 MinHash objects were given")
    seed = minhashs[0].seed
    if any(seed != m.seed for m in minhashs):
        raise MinHashException("Cannot compare MinHash objects with\
                different seeds")
    num_perm = len(minhash[0].permutations)
    if any(num_perm != len(m.permutations) for m in minhashs):
        raise MinHashException("Cannot compare MinHash objects with\
                different numbers of permutation functions")
    intersection = 0
    for i in xrange(num_perm):
        phv = minhashs[0].hashvalues[i]
        if all(phv == m.hashvalues[i] for m in minhashs):
            intersection += 1
    return float(intersection) / float(num_perm)


class MinHashException(Exception):
    pass
