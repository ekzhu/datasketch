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
    '''MinHash is a probabilistic data structure for computing 
    `Jaccard similarity`_ between sets.
 
    Args:
        num_perm (int, optional): Number of random permutation functions.
            It will be ignored if `hashvalues` is not None.
        seed (int, optional): The random seed controls the set of random 
            permutation functions generated for this MinHash.
        hashobj (optional): The hash function used by this MinHash. 
            It must implements
            the `digest()` method similar to hashlib_ hash functions, such
            as `hashlib.sha1`.
        hashvalues (`numpy.array` or `list`, optional): The hash values is 
            the internal state of the MinHash. It can be specified for faster 
            initialization using the existing state from another MinHash. 
        permutations (optional): The permutation function parameters. This argument
            can be specified for faster initialization using the existing
            state from another MinHash.
    
    Note:
        To save memory usage, consider using :class:`datasketch.LeanMinHash`.
    
    .. _`Jaccard similarity`: https://en.wikipedia.org/wiki/Jaccard_index
    .. _hashlib: https://docs.python.org/3.5/library/hashlib.html
    '''

    __slots__ = ('permutations', 'hashvalues', 'seed', 'hashobj')

    def __init__(self, num_perm=128, seed=1, hashobj=sha1,
            hashvalues=None, permutations=None):
        if hashvalues is not None:
            num_perm = len(hashvalues)
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

    def update(self, b):
        '''Update this MinHash with a new value.
        
        Args:
            b (bytes): The value of type `bytes`.
            
        Example:
            To update with a new string value:
            
            .. code-block:: python

                minhash.update("new value".encode('utf-8'))
        '''
        hv = struct.unpack('<I', self.hashobj(b).digest()[:4])[0]
        a, b = self.permutations
        phv = np.bitwise_and((a * hv + b) % _mersenne_prime, np.uint64(_max_hash))
        self.hashvalues = np.minimum(phv, self.hashvalues)

    def jaccard(self, other):
        '''Estimate the `Jaccard similarity`_ (resemblance) between the sets
        represented by this MinHash and the other.
        
        Args:
            other (datasketch.MinHash): The other MinHash.
            
        Returns:
            float: The Jaccard similarity, which is between 0.0 and 1.0.
        '''
        if other.seed != self.seed:
            raise ValueError("Cannot compute Jaccard given MinHash with\
                    different seeds")
        if len(self) != len(other):
            raise ValueError("Cannot compute Jaccard given MinHash with\
                    different numbers of permutation functions")
        return np.float(np.count_nonzero(self.hashvalues==other.hashvalues)) /\
                np.float(len(self))

    def count(self):
        '''Estimate the cardinality count based on the technique described in
        `this paper <http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=365694>`_.
        
        Returns:
            int: The estimated cardinality of the set represented by this MinHash.
        '''
        k = len(self)
        return np.float(k) / np.sum(self.hashvalues / np.float(_max_hash)) - 1.0

    def merge(self, other):
        '''Merge the other MinHash with this one, making this one the union
        of both.
        
        Args:
            other (datasketch.MinHash): The other MinHash.
        '''
        if other.seed != self.seed:
            raise ValueError("Cannot merge MinHash with\
                    different seeds")
        if len(self) != len(other):
            raise ValueError("Cannot merge MinHash with\
                    different numbers of permutation functions")
        self.hashvalues = np.minimum(other.hashvalues, self.hashvalues)

    def digest(self):
        '''Export the hash values, which is the internal state of the
        MinHash.
        
        Returns:
            numpy.array: The hash values which is a Numpy array.
        '''
        return copy.copy(self.hashvalues)

    def is_empty(self):
        '''
        Returns: 
            bool: If the current MinHash is empty - at the state of just
                initialized.
        '''
        if np.any(self.hashvalues != _max_hash):
            return False
        return True

    def clear(self):
        '''
        Clear the current state of the MinHash.
        All hash values are reset.
        '''
        self.hashvalues = self._init_hashvalues(len(self))

    def copy(self):
        '''
        Returns:
            datasketch.MinHash: A copy of this MinHash by exporting its
                state.
        '''
        return MinHash(seed=self.seed, hashvalues=self.digest(),
                permutations=self.permutations)

    def __len__(self):
        '''
        Returns:
            int: The number of hash values.
        '''
        return len(self.hashvalues)

    def __eq__(self, other):
        '''
        Returns:
            bool: If their seeds and hash values are both equal then two
                are equivalent.
        '''
        return self.seed == other.seed and \
                np.array_equal(self.hashvalues, other.hashvalues)

    def bytesize(self):
        # Use 8 bytes to store the seed integer
        seed_size = struct.calcsize('q')
        # Use 4 bytes to store the number of hash values
        length_size = struct.calcsize('i')
        # Use 4 bytes to store each hash value as we are using the lower 32 bit
        hashvalue_size = struct.calcsize('I')
        return seed_size + length_size + len(self) * hashvalue_size

    def serialize(self, buf):
        if len(buf) < self.bytesize():
            raise ValueError("The buffer does not have enough space\
                    for holding this MinHash.")
        fmt = "qi%dI" % len(self)
        struct.pack_into(fmt, buf, 0,
                self.seed, len(self), *self.hashvalues)

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
        return cls(num_perm=num_perm, seed=seed, hashvalues=hashvalues)

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
        self.__init__(num_perm=num_perm, seed=seed, hashvalues=hashvalues)

    @classmethod
    def union(cls, *mhs):
        '''Create a MinHash which is the union of the MinHash objects passed as arguments.

        Args:
            *mhs: The MinHash objects to be united. The argument list length is variable,
                but must be at least 2.
        
        Returns:
            datasketch.MinHash: A new union MinHash.
        '''
        if len(mhs) < 2:
            raise ValueError("Cannot union less than 2 MinHash")
        num_perm = len(mhs[0])
        seed = mhs[0].seed
        if any((seed != m.seed or num_perm != len(m)) for m in mhs):
            raise ValueError("The unioning MinHash must have the\
                    same seed and number of permutation functions")
        hashvalues = np.minimum.reduce([m.hashvalues for m in mhs])
        permutations = mhs[0].permutations
        return cls(num_perm=num_perm, seed=seed, hashvalues=hashvalues,
                permutations=permutations)
