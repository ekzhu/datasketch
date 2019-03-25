import random, copy, struct
import warnings
import numpy as np

from datasketch.hashfunc import sha1_hash32

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
        hashfunc (optional): The hash function used by this MinHash.
            It takes the input passed to the `update` method and
            returns an integer that can be encoded with 32 bits.
            The default hash function is based on SHA1 from hashlib_.
        hashobj (**deprecated**): This argument is deprecated since version
            1.4.0. It is a no-op and has been replaced by `hashfunc`.
        hashvalues (`numpy.array` or `list`, optional): The hash values is
            the internal state of the MinHash. It can be specified for faster
            initialization using the existing state from another MinHash.
        permutations (optional): The permutation function parameters. This argument
            can be specified for faster initialization using the existing
            state from another MinHash.

    Note:
        To save memory usage, consider using :class:`datasketch.LeanMinHash`.

    Note:
        Since version 1.1.1, MinHash will only support serialization using
        `pickle`_. ``serialize`` and ``deserialize`` methods are removed,
        and are supported in :class:`datasketch.LeanMinHash` instead.
        MinHash serialized before version 1.1.1 cannot be deserialized properly
        in newer versions (`need to migrate? <https://github.com/ekzhu/datasketch/issues/18>`_).

    Note:
        Since version 1.1.3, MinHash uses Numpy's random number generator
        instead of Python's built-in random package. This change makes the
        hash values consistent across different Python versions.
        The side-effect is that now MinHash created before version 1.1.3 won't
        work (i.e., ``jaccard``, ``merge`` and ``union``)
        with those created after.

    .. _`Jaccard similarity`: https://en.wikipedia.org/wiki/Jaccard_index
    .. _hashlib: https://docs.python.org/3.5/library/hashlib.html
    .. _`pickle`: https://docs.python.org/3/library/pickle.html
    '''

    def __init__(self, num_perm=128, seed=1,
            hashfunc=sha1_hash32,
            hashobj=None, # Deprecated.
            hashvalues=None, permutations=None):
        if hashvalues is not None:
            num_perm = len(hashvalues)
        if num_perm > _hash_range:
            # Because 1) we don't want the size to be too large, and
            # 2) we are using 4 bytes to store the size value
            raise ValueError("Cannot have more than %d number of\
                    permutation functions" % _hash_range)
        self.seed = seed
        # Check the hash function.
        if not callable(hashfunc):
            raise ValueError("The hashfunc must be a callable.")
        self.hashfunc = hashfunc
        # Check for use of hashobj and issue warning.
        if hashobj is not None:
            warnings.warn("hashobj is deprecated, use hashfunc instead.",
                    DeprecationWarning)
        # Initialize hash values
        if hashvalues is not None:
            self.hashvalues = self._parse_hashvalues(hashvalues)
        else:
            self.hashvalues = self._init_hashvalues(num_perm)
        # Initalize permutation function parameters
        if permutations is not None:
            self.permutations = permutations
        else:
            generator = np.random.RandomState(self.seed)
            # Create parameters for a random bijective permutation function
            # that maps a 32-bit hash value to another 32-bit hash value.
            # http://en.wikipedia.org/wiki/Universal_hashing
            self.permutations = np.array([(generator.randint(1, _mersenne_prime, dtype=np.uint64),
                                           generator.randint(0, _mersenne_prime, dtype=np.uint64))
                                          for _ in range(num_perm)], dtype=np.uint64).T
        if len(self) != len(self.permutations[0]):
            raise ValueError("Numbers of hash values and permutations mismatch")

    def _init_hashvalues(self, num_perm):
        return np.ones(num_perm, dtype=np.uint64)*_max_hash

    def _parse_hashvalues(self, hashvalues):
        return np.array(hashvalues, dtype=np.uint64)

    def update(self, b):
        '''Update this MinHash with a new value.
        The value will be hashed using the hash function specified by
        the `hashfunc` argument in the constructor.

        Args:
            b: The value to be hashed using the hash function specified.

        Example:
            To update with a new string value (using the default SHA1 hash
            function, which requires bytes as input):

            .. code-block:: python

                minhash = Minhash()
                minhash.update("new value".encode('utf-8'))

            We can also use a different hash function, for example, `pyfarmhash`:

            .. code-block:: python

                import farmhash
                def _hash_32(b):
                    return farmhash.hash32(b)
                minhash = MinHash(hashfunc=_hash_32)
                minhash.update("new value")
        '''
        hv = self.hashfunc(b)
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
        :returns: datasketch.MinHash -- A copy of this MinHash by exporting its state.
        '''
        return MinHash(seed=self.seed, hashfunc=self.hashfunc,
                hashvalues=self.digest(),
                permutations=self.permutations)

    def __len__(self):
        '''
        :returns: int -- The number of hash values.
        '''
        return len(self.hashvalues)

    def __eq__(self, other):
        '''
        :returns: bool -- If their seeds and hash values are both equal then two are equivalent.
        '''
        return type(self) is type(other) and \
            self.seed == other.seed and \
            np.array_equal(self.hashvalues, other.hashvalues)

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
