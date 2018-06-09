import pickle
from datasketch.aiostorage import (_random_name, async_ordered_storage, async_unordered_storage)
from datasketch.storage import unordered_storage

_integration_precision = 0.001


def _integration(f, a, b):
    p = _integration_precision
    area = 0.0
    x = a
    while x < b:
        area += f(x + 0.5 * p) * p
        x += p
    return area, None


try:
    from scipy.integrate import quad as integrate
except ImportError:
    # For when no scipy installed
    integrate = _integration


def _false_positive_probability(threshold, b, r):
    _probability = lambda s: 1 - (1 - s ** float(r)) ** float(b)
    a, err = integrate(_probability, 0.0, threshold)
    return a


def _false_negative_probability(threshold, b, r):
    _probability = lambda s: 1 - (1 - (1 - s ** float(r)) ** float(b))
    a, err = integrate(_probability, threshold, 1.0)
    return a


def _optimal_param(threshold, num_perm, false_positive_weight, false_negative_weight):
    '''
    Compute the optimal `MinHashLSH` parameter that minimizes the weighted sum
    of probabilities of false positive and false negative.
    '''
    min_error = float("inf")
    opt = (0, 0)
    for b in range(1, num_perm + 1):
        max_r = int(num_perm / b)
        for r in range(1, max_r + 1):
            fp = _false_positive_probability(threshold, b, r)
            fn = _false_negative_probability(threshold, b, r)
            error = fp * false_positive_weight + fn * false_negative_weight
            if error < min_error:
                min_error = error
                opt = (b, r)
    return opt


async def create_AsyncMinHashLSH(*args, **kwargs):
    async_lsh = AsyncMinHashLSH(*args, **kwargs)
    await async_lsh.create_storages()
    return async_lsh


class AsyncMinHashLSH(object):
    """
    The :ref:`minhash_lsh` index.
    It supports query with `Jaccard similarity`_ threshold.
    Reference: `Chapter 3, Mining of Massive Datasets
    <http://www.mmds.org/>`_.

    Args:
        threshold (float): The Jaccard similarity threshold between 0.0 and
            1.0. The initialized MinHash LSH will be optimized for the threshold by
            minizing the false positive and false negative.
        num_perm (int, optional): The number of permutation functions used
            by the MinHash to be indexed. For weighted MinHash, this
            is the sample size (`sample_size`).
        weights (tuple, optional): Used to adjust the relative importance of
            minimizing false positive and false negative when optimizing
            for the Jaccard similarity threshold.
            `weights` is a tuple in the format of
            :code:`(false_positive_weight, false_negative_weight)`.
        params (tuple, optional): The LSH parameters (i.e., number of bands and size
            of each bands). This is used to bypass the parameter optimization
            step in the constructor. `threshold` and `weights` will be ignored
            if this is given.
        storage_config (dict, optional): Type of storage service to use for storing
            hashtables and keys.

    Note:
        `weights` must sum to 1.0, and the format is
        (false positive weight, false negative weight).
        For example, if minimizing false negative (or maintaining high recall) is more
        important, assign more weight toward false negative: weights=(0.4, 0.6).
        Try to live with a small difference between weights (i.e. < 0.5).
    """

    def __init__(self, threshold=0.9, num_perm=128, weights=(0.5, 0.5),
                 params=None, storage_config=None):
        self._storage_config = storage_config if storage_config else {'type': 'aioredis',
                                                                      'redis': {'host': 'localhost', 'port': 6379}}
        assert storage_config['type'] == 'aioredis', 'Wrong storage type. See documentation for storage types'
        if threshold > 1.0 or threshold < 0.0:
            raise ValueError("threshold must be in [0.0, 1.0]")
        if num_perm < 2:
            raise ValueError("Too few permutation functions")
        if any(w < 0.0 or w > 1.0 for w in weights):
            raise ValueError("Weight must be in [0.0, 1.0]")
        if sum(weights) != 1.0:
            raise ValueError("Weights must sum to 1.0")
        self.h = num_perm
        if params is not None:
            self.b, self.r = params
            if self.b * self.r > num_perm:
                raise ValueError("The product of b and r must be less than num_perm")
        else:
            false_positive_weight, false_negative_weight = weights
            self.b, self.r = _optimal_param(threshold, num_perm,
                                            false_positive_weight, false_negative_weight)

    async def create_storages(self):
        basename = _random_name(11)
        self.hashranges = [(i * self.r, (i + 1) * self.r) for i in range(self.b)]
        self.hashtables = [await async_unordered_storage(self._storage_config, name=basename + b'_bucket_' + bytes([i]))
                           for i in range(self.b)]
        self.keys = await async_ordered_storage(self._storage_config, name=basename + b'_keys')

    async def init_storages(self):
        for t in self.hashtables:
            await t.create_redis()
        await self.keys.create_redis()

    async def close(self):
        for t in self.hashtables:
            await t.close()
        await self.keys.close()

    async def insert(self, key, minhash):
        """
        Insert a unique key to the index, together
        with a MinHash (or weighted MinHash) of the set referenced by
        the key.

        Args:
            key (hashable): The unique identifier of the set.
            minhash (datasketch.MinHash): The MinHash of the set.
        """
        await self._insert(key, minhash, check_duplication=True, buffer=False)

    def insertion_session(self):
        """
        Create a asynchronous context manager for fast insertion into this index.

        Returns:
            datasketch.lsh.AsyncMinHashLSHInsertionSession
        """
        return AsyncMinHashLSHInsertionSession(self)

    async def _insert(self, key, minhash, check_duplication=True, buffer=False):
        if len(minhash) != self.h:
            raise ValueError("Expecting minhash with length %d, got %d" % (self.h, len(minhash)))
        if check_duplication and await self.has_key(key):
            raise ValueError("The given key already exists")
        Hs = [self._H(minhash.hashvalues[start:end]) for start, end in self.hashranges]
        key = pickle.dumps(key)
        await self.keys.insert(key, *Hs, buffer=buffer)
        for H, hashtable in zip(Hs, self.hashtables):
            await hashtable.insert(H, key, buffer=buffer)

    async def query(self, minhash):
        """
        Giving the MinHash of the query set, retrieve
        the keys that references sets with Jaccard
        similarities greater than the threshold.

        Args:
            minhash (datasketch.MinHash): The MinHash of the query set.

        Returns:
            `list` of keys.
        """
        if len(minhash) != self.h:
            raise ValueError("Expecting minhash with length %d, got %d"
                             % (self.h, len(minhash)))
        candidates = set()
        for (start, end), hashtable in zip(self.hashranges, self.hashtables):
            H = self._H(minhash.hashvalues[start:end])
            for key in await hashtable.get(H):
                candidates.add(key)
        return [pickle.loads(key) for key in candidates]

    async def has_key(self, key):
        """
        Args:
            key (hashable): The unique identifier of a set.

        Returns:
            bool: True only if the key exists in the index.
        """
        return await self.keys.has_key(pickle.dumps(key))
        # return await self.keys.has_key(key)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def remove(self, key):
        """
        Remove the key from the index.

        Args:
            key (hashable): The unique identifier of a set.

        """
        if not await self.has_key(key):
            raise ValueError("The given key does not exist")
        key = pickle.dumps(key)
        for H, hashtable in zip(await self.keys.get(key), self.hashtables):
            await hashtable.remove_val(H, key)
            if not await hashtable.get(H):
                await hashtable.remove(H)
        await self.keys.remove(key)

    async def is_empty(self):
        """
        Returns:
            bool: Check if the index is empty.
        """
        for t in self.hashtables:
            if await t.size() == 0:
                return True
        return False

    @staticmethod
    def _H(hs):
        return bytes(hs.byteswap().data)

    async def _query_b(self, minhash, b):
        if len(minhash) != self.h:
            raise ValueError("Expecting minhash with length %d, got %d"
                             % (self.h, len(minhash)))
        if b > len(self.hashtables):
            raise ValueError("b must be less or equal to the number of hash tables")
        candidates = set()
        for (start, end), hashtable in zip(self.hashranges[:b], self.hashtables[:b]):
            H = self._H(minhash.hashvalues[start:end])
            if await hashtable.has_key(H):
                for key in await hashtable.get(H):
                    candidates.add(key)

        return {pickle.loads(key) for key in candidates}

    async def get_counts(self):
        """
        Returns a list of length ``self.b`` with elements representing the
        number of keys stored under each bucket for the given permutation.
        """
        counts = [await hashtable.itemcounts() for hashtable in self.hashtables]
        return counts

    async def get_subset_counts(self, *keys):
        """
        Returns the bucket allocation counts (see :ref:`get_counts` above)
        restricted to the list of keys given.

        Args:
            keys (hashable) : the keys for which to get the bucket allocation
                counts
        """
        key_set = [pickle.dumps(key) for key in set(keys)]
        hashtables = [unordered_storage({'type': 'dict'}) for _ in range(self.b)]
        Hss = await self.keys.getmany(*key_set)
        for key, Hs in zip(key_set, Hss):
            for H, hashtable in zip(Hs, hashtables):
                await hashtable.insert(H, key)
        return [await hashtable.itemcounts() for hashtable in hashtables]


class AsyncMinHashLSHInsertionSession:
    """Asynchronous Context manager for batch insertion of documents into a MinHashLSH."""

    def __init__(self, lsh):
        self.lsh = lsh

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.lsh.keys.empty_buffer()
        for hashtable in self.lsh.hashtables:
            await hashtable.empty_buffer()

    async def insert(self, key, minhash, check_duplication=True):
        """
        Insert a unique key to the index, together
        with a MinHash (or weighted MinHash) of the set referenced by
        the key.

        Args:
            :param minhash: (datasketch.MinHash) The MinHash of the set.
            :param key: (hashable) The unique identifier of the set.
            :param check_duplication: (default=True)
        """
        await self.lsh._insert(key, minhash, check_duplication=check_duplication, buffer=True)