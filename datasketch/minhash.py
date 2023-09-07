from __future__ import annotations
import copy
from typing import Callable, Generator, Iterable, List, Optional, Tuple
import warnings
import numpy as np

from datasketch.hashfunc import sha1_hash32

# The size of a hash value in number of bytes
hashvalue_byte_size = len(bytes(np.int64(42).data))

# http://en.wikipedia.org/wiki/Mersenne_prime
_mersenne_prime = np.uint64((1 << 61) - 1)
_max_hash = np.uint64((1 << 32) - 1)
_hash_range = 1 << 32


class MinHash(object):
    """MinHash is a probabilistic data structure for computing
    `Jaccard similarity`_ between sets.

    Args:
        num_perm (int): Number of random permutation functions.
            It will be ignored if `hashvalues` is not None.
        seed (int): The random seed controls the set of random
            permutation functions generated for this MinHash.
        hashfunc (Callable): The hash function used by
            this MinHash.
            It takes the input passed to the :meth:`update` method and
            returns an integer that can be encoded with 32 bits.
            The default hash function is based on SHA1 from hashlib_.
            Users can use `farmhash` for better performance.
            See the example in :meth:`update`.
        hashobj (**deprecated**): This argument is deprecated since version
            1.4.0. It is a no-op and has been replaced by `hashfunc`.
        hashvalues (Optional[Iterable]): The hash values is
            the internal state of the MinHash. It can be specified for faster
            initialization using the existing :attr:`hashvalues` of another MinHash.
        permutations (Optional[Tuple[Iterable, Iterable]]): The permutation
            function parameters as a tuple of two lists. This argument
            can be specified for faster initialization using the existing
            :attr:`permutations` from another MinHash.

    Note:
        To save memory usage, consider using :class:`datasketch.LeanMinHash`.

    Note:
        Since version 1.1.1, MinHash will only support serialization using
        pickle_. ``serialize`` and ``deserialize`` methods are removed,
        and are supported in :class:`datasketch.LeanMinHash` instead.
        MinHash serialized before version 1.1.1 cannot be deserialized properly
        in newer versions (`need to migrate? <https://github.com/ekzhu/datasketch/issues/18>`_).

    Note:
        Since version 1.1.3, MinHash uses Numpy's random number generator
        instead of Python's built-in random package. This change makes the
        hash values consistent across different Python versions.
        The side-effect is that now MinHash created before version 1.1.3 won't
        work (i.e., :meth:`jaccard`, :meth:`merge` and :meth:`union`)
        with those created after.

    .. _`Jaccard similarity`: https://en.wikipedia.org/wiki/Jaccard_index
    .. _hashlib: https://docs.python.org/3.5/library/hashlib.html
    .. _`pickle`: https://docs.python.org/3/library/pickle.html
    """

    def __init__(
        self,
        num_perm: int = 128,
        seed: int = 1,
        hashfunc: Callable = sha1_hash32,
        hashobj: Optional[object] = None,  # Deprecated.
        hashvalues: Optional[Iterable] = None,
        permutations: Optional[Tuple[Iterable, Iterable]] = None,
    ) -> None:
        if hashvalues is not None:
            num_perm = len(hashvalues)
        if num_perm > _hash_range:
            # Because 1) we don't want the size to be too large, and
            # 2) we are using 4 bytes to store the size value
            raise ValueError(
                "Cannot have more than %d number of\
                    permutation functions"
                % _hash_range
            )
        self.seed = seed
        self.num_perm = num_perm
        # Check the hash function.
        if not callable(hashfunc):
            raise ValueError("The hashfunc must be a callable.")
        self.hashfunc = hashfunc
        # Check for use of hashobj and issue warning.
        if hashobj is not None:
            warnings.warn(
                "hashobj is deprecated, use hashfunc instead.", DeprecationWarning
            )
        # Initialize hash values
        if hashvalues is not None:
            self.hashvalues = self._parse_hashvalues(hashvalues)
        else:
            self.hashvalues = self._init_hashvalues(num_perm)
        # Initalize permutation function parameters
        if permutations is not None:
            self.permutations = permutations
        else:
            self.permutations = self._init_permutations(num_perm)
        if len(self) != len(self.permutations[0]):
            raise ValueError("Numbers of hash values and permutations mismatch")

    def _init_hashvalues(self, num_perm: int) -> np.ndarray:
        return np.ones(num_perm, dtype=np.uint64) * _max_hash

    def _init_permutations(self, num_perm: int) -> np.ndarray:
        # Create parameters for a random bijective permutation function
        # that maps a 32-bit hash value to another 32-bit hash value.
        # http://en.wikipedia.org/wiki/Universal_hashing
        gen = np.random.RandomState(self.seed)
        return np.array(
            [
                (
                    gen.randint(1, _mersenne_prime, dtype=np.uint64),
                    gen.randint(0, _mersenne_prime, dtype=np.uint64),
                )
                for _ in range(num_perm)
            ],
            dtype=np.uint64,
        ).T

    def _parse_hashvalues(self, hashvalues):
        return np.array(hashvalues, dtype=np.uint64)

    def update(self, b) -> None:
        """Update this MinHash with a new value.
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
        """
        hv = self.hashfunc(b)
        a, b = self.permutations
        phv = np.bitwise_and((a * hv + b) % _mersenne_prime, _max_hash)
        self.hashvalues = np.minimum(phv, self.hashvalues)

    def update_batch(self, b: Iterable) -> None:
        """Update this MinHash with new values.
        The values will be hashed using the hash function specified by
        the `hashfunc` argument in the constructor.

        Args:
            b (Iterable): Values to be hashed using the hash function specified.

        Example:
            To update with new string values (using the default SHA1 hash
            function, which requires bytes as input):

            .. code-block:: python

                minhash = Minhash()
                minhash.update_batch([s.encode('utf-8') for s in ["token1", "token2"]])
        """
        hv = np.array([self.hashfunc(_b) for _b in b], dtype=np.uint64, ndmin=2).T
        a, b = self.permutations
        phv = (hv * a + b) % _mersenne_prime & _max_hash
        self.hashvalues = np.vstack([phv, self.hashvalues]).min(axis=0)

    def jaccard(self, other: MinHash) -> float:
        """Estimate the `Jaccard similarity`_ (resemblance) between the sets
        represented by this MinHash and the other.

        Args:
            other (MinHash): The other MinHash.

        Returns:
            float: The Jaccard similarity, which is between 0.0 and 1.0.

        Raises:
            ValueError: If the two MinHashes have different numbers of
                permutation functions or different seeds.
        """
        if other.seed != self.seed:
            raise ValueError(
                "Cannot compute Jaccard given MinHash with\
                    different seeds"
            )
        if len(self) != len(other):
            raise ValueError(
                "Cannot compute Jaccard given MinHash with\
                    different numbers of permutation functions"
            )
        return float(np.count_nonzero(self.hashvalues == other.hashvalues)) / float(
            len(self)
        )

    def count(self) -> float:
        """Estimate the cardinality count based on the technique described in
        `this paper <http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=365694>`_.

        Returns:
            int: The estimated cardinality of the set represented by this MinHash.
        """
        k = len(self)
        return float(k) / np.sum(self.hashvalues / float(_max_hash)) - 1.0

    def merge(self, other: MinHash) -> None:
        """Merge the other MinHash with this one, making this one the union
        of both.

        Args:
            other (MinHash): The other MinHash.

        Raises:
            ValueError: If the two MinHashes have different numbers of
                permutation functions or different seeds.
        """
        if other.seed != self.seed:
            raise ValueError(
                "Cannot merge MinHash with\
                    different seeds"
            )
        if len(self) != len(other):
            raise ValueError(
                "Cannot merge MinHash with\
                    different numbers of permutation functions"
            )
        self.hashvalues = np.minimum(other.hashvalues, self.hashvalues)

    def digest(self) -> np.ndarray:
        """Export the hash values, which is the internal state of the
        MinHash.

        Returns:
            numpy.ndarray: The hash values which is a Numpy array.
        """
        return copy.copy(self.hashvalues)

    def is_empty(self) -> bool:
        """
        Returns:
            bool: If the current MinHash is empty - at the state of just
                initialized.
        """
        if np.any(self.hashvalues != _max_hash):
            return False
        return True

    def clear(self) -> None:
        """
        Clear the current state of the MinHash.
        All hash values are reset.
        """
        self.hashvalues = self._init_hashvalues(len(self))

    def copy(self) -> MinHash:
        """
        Returns:
            MinHash: a copy of this MinHash by exporting its state.
        """
        return MinHash(
            seed=self.seed,
            hashfunc=self.hashfunc,
            hashvalues=self.digest(),
            permutations=self.permutations,
        )

    def __len__(self) -> int:
        """
        Returns:
            int: The number of hash values.
        """
        return len(self.hashvalues)

    def __eq__(self, other: MinHash) -> bool:
        """
        Returns:
            bool: If their seeds and hash values are both equal then two are equivalent.
        """
        return (
            type(self) is type(other)
            and self.seed == other.seed
            and np.array_equal(self.hashvalues, other.hashvalues)
        )

    @classmethod
    def union(cls, *mhs: MinHash) -> MinHash:
        """Create a MinHash which is the union of the MinHash objects passed as arguments.

        Args:
            *mhs (MinHash): The MinHash objects to be united. The argument list length is variable,
                but must be at least 2.

        Returns:
            MinHash: a new union MinHash.

        Raises:
            ValueError: If the number of MinHash objects passed as arguments is less than 2,
                or if the MinHash objects passed as arguments have different seeds or
                different numbers of permutation functions.

        Example:

            .. code-block:: python

                from datasketch import MinHash
                import numpy as np

                m1 = MinHash(num_perm=128)
                m1.update_batch(np.random.randint(low=0, high=30, size=10))

                m2 = MinHash(num_perm=128)
                m2.update_batch(np.random.randint(low=0, high=30, size=10))

                # Union m1 and m2.
                m = MinHash.union(m1, m2)
        """
        if len(mhs) < 2:
            raise ValueError("Cannot union less than 2 MinHash")
        num_perm = len(mhs[0])
        seed = mhs[0].seed
        if any((seed != m.seed or num_perm != len(m)) for m in mhs):
            raise ValueError(
                "The unioning MinHash must have the\
                    same seed and number of permutation functions"
            )
        hashvalues = np.minimum.reduce([m.hashvalues for m in mhs])
        permutations = mhs[0].permutations
        return cls(
            num_perm=num_perm,
            seed=seed,
            hashvalues=hashvalues,
            permutations=permutations,
        )

    @classmethod
    def bulk(cls, b: Iterable, **minhash_kwargs) -> List[MinHash]:
        """Compute MinHashes in bulk. This method avoids unnecessary
        overhead when initializing many minhashes by reusing the initialized
        state.

        Args:
            b (Iterable): An Iterable of lists of bytes, each list is
                hashed in to one MinHash in the output.
            **minhash_kwargs: Keyword arguments used to initialize MinHash,
                will be used for all minhashes.

        Returns:
            List[datasketch.MinHash]: A list of computed MinHashes.

        Example:

            .. code-block:: python

                from datasketch import MinHash
                data = [[b'token1', b'token2', b'token3'],
                        [b'token4', b'token5', b'token6']]
                minhashes = MinHash.bulk(data, num_perm=64)

        """
        return list(cls.generator(b, **minhash_kwargs))

    @classmethod
    def generator(cls, b: Iterable, **minhash_kwargs) -> Generator[MinHash, None, None]:
        """Compute MinHashes in a generator. This method avoids unnecessary
        overhead when initializing many minhashes by reusing the initialized
        state.

        Args:
            b (Iterable): An Iterable of lists of bytes, each list is
                hashed in to one MinHash in the output.
            minhash_kwargs: Keyword arguments used to initialize MinHash,
                will be used for all minhashes.

        Returns:
            Generator[MinHash, None, None]: a generator of computed MinHashes.

        Example:

            .. code-block:: python

                from datasketch import MinHash
                data = [[b'token1', b'token2', b'token3'],
                        [b'token4', b'token5', b'token6']]
                for minhash in MinHash.generator(data, num_perm=64):
                    # do something useful
                    minhash

        """
        m = cls(**minhash_kwargs)
        for _b in b:
            _m = m.copy()
            _m.update_batch(_b)
            yield _m
