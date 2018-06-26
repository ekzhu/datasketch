import asyncio
import pickle
from datasketch.async.storage import (async_ordered_storage,
                                      async_unordered_storage,)

from datasketch.lsh import _optimal_param
from datasketch.storage import unordered_storage, _random_name


class AsyncMinHashLSH(object):
    """
    The :ref:`minhash_lsh` index.
    It supports query with `Jaccard similarity`_ threshold.
    Reference: `Chapter 3, Mining of Massive Datasets
    <http://www.mmds.org/>`_.

    Args:
        threshold (float): The Jaccard similarity threshold between 0.0 and
            1.0. The initialized AsyncMinHash LSH will be optimized for the
            threshold by minizing the false positive and false negative.
        num_perm (int, optional): The number of permutation functions used
            by the AsyncMinHash to be indexed. For weighted AsyncMinHash, this
            is the sample size (`sample_size`).
        weights (tuple, optional): Used to adjust the relative importance of
            minimizing false positive and false negative when optimizing
            for the Jaccard similarity threshold.
            `weights` is a tuple in the format of
            :code:`(false_positive_weight, false_negative_weight)`.
        params (tuple, optional): The LSH parameters (i.e., number of bands and
            size of each bands). This is used to bypass the parameter
            optimization step in the constructor. `threshold` and `weights`
            will be ignored if this is given.
        storage_config (dict, optional): Type of storage service to use for
            storing hashtables and keys.

    Note:
        `weights` must sum to 1.0, and the format is
        (false positive weight, false negative weight).
        For example, if minimizing false negative (or maintaining high recall)
        is more important, assign more weight toward false negative:
        weights=(0.4, 0.6). Try to live with a small difference between weights
        (i.e. < 0.5).
    """

    def __init__(self, threshold=0.9, num_perm=128, weights=(0.5, 0.5),
                 params=None, storage_config=None, prepickle=None, batch_size=10000):

        if storage_config is None:
            storage_config = {
                'type': 'aioredis',
                'redis': {'host': 'localhost', 'port': 6379}
            }
        self._storage_config = storage_config
        self._batch_size = batch_size
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
                raise ValueError("The product of b and r must be less than "
                                 "num_perm")
        else:
            false_positive_weight, false_negative_weight = weights
            self.b, self.r = _optimal_param(threshold, num_perm, false_positive_weight, false_negative_weight)

        if prepickle is None:
            self.prepickle = storage_config['type'] == 'aioredis'
        else:
            self.prepickle = prepickle

        self.hashranges = [(i * self.r, (i + 1) * self.r)
                           for i in range(self.b)]
        self.hashtables = None
        self.keys = None

        self._lock = asyncio.Lock()
        self._initialized = False
        self._basename = _random_name(11)

    async def __async_init(self):
        async with self._lock:
            if not self._initialized:
                await self.init_storages()
            self._initialized = True
        return self

    def __await__(self):
        return self.__async_init().__await__()

    async def __aenter__(self):
        return await self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_initialized'] = False
        state.pop('_lock')
        return state

    def __setstate__(self, state):
        state['_lock'] = asyncio.Lock()
        self.__dict__ = state

    async def _create_storages(self):
        if self._storage_config['type'] == 'aioredis':
            name_ordered = self._basename + b'_keys'
            fs = (
                async_unordered_storage(
                    config=self._storage_config,
                    batch_size=self._batch_size,
                    name=self._basename + b'_bucket_'+bytes([i]),
                )
                for i in range(self.b)
            )
        else:
            name_ordered = ''.join([self._basename.decode('utf-8'), '_keys'])
            fs = (
                async_unordered_storage(
                    config=self._storage_config,
                    batch_size=self._batch_size,
                    name=''.join([self._basename.decode('utf-8'), '_bucket_', str(i)])
                )
                for i in range(self.b)
            )

        self.hashtables = await asyncio.gather(*fs)
        self.keys = await async_ordered_storage(self._storage_config,
                                                batch_size=self._batch_size,
                                                name=name_ordered)

    async def init_storages(self):
        if self.keys is None:
            await self._create_storages()

        if not self.keys.initialized:
            await self.keys

        fs = (ht for ht in self.hashtables if not ht.initialized)
        await asyncio.gather(*fs)

    async def close(self):
        async with self._lock:
            for t in self.hashtables:
                await t.close()

            if self.keys is not None:
                await self.keys.close()

            self._initialized = False

    async def insert(self, key, minhash):
        """
        Insert a unique key to the index, together
        with a AsyncMinHash (or weighted AsyncMinHash) of the set referenced by
        the key.

        Args:
            key (hashable): The unique identifier of the set.
            minhash (datasketch.AsyncMinHash): The AsyncMinHash of the set.
        """
        await self._insert(key, minhash, check_duplication=True, buffer=False)

    def insertion_session(self):
        """
        Create a asynchronous context manager for fast insertion into this
        index.

        Returns:
            datasketch.lsh.AsyncMinHashLSHInsertionSession
        """
        return AsyncMinHashLSHInsertionSession(self)

    async def _insert(self, key, minhash, check_duplication=True,
                      buffer=False):
        if len(minhash) != self.h:
            raise ValueError("Expecting minhash with length %d, "
                             "got %d" % (self.h, len(minhash)))
        if check_duplication and await self.has_key(key):
            raise ValueError("The given key already exists")
        Hs = [self._H(minhash.hashvalues[start:end])
              for start, end in self.hashranges]
        if self.prepickle:
            key = pickle.dumps(key)
        await self.keys.insert(key, *Hs, buffer=buffer)
        fs = (hashtable.insert(H, key, buffer=buffer) for H, hashtable in zip(Hs, self.hashtables))
        await asyncio.gather(*fs)

    async def query(self, minhash):
        """
        Giving the AsyncMinHash of the query set, retrieve
        the keys that references sets with Jaccard
        similarities greater than the threshold.

        Args:
            minhash (datasketch.AsyncMinHash): The AsyncMinHash of the query set.

        Returns:
            `list` of keys.
        """
        if len(minhash) != self.h:
            raise ValueError("Expecting minhash with length %d, "
                             "got %d" % (self.h, len(minhash)))
        candidates = set()
        for (start, end), hashtable in zip(self.hashranges, self.hashtables):
            H = self._H(minhash.hashvalues[start:end])
            for key in await hashtable.get(H):
                candidates.add(key)

        if self.prepickle:
            return [pickle.loads(key) for key in candidates]
        else:
            return list(candidates)

    async def has_key(self, key):
        """
        Args:
            key (hashable): The unique identifier of a set.

        Returns:
            bool: True only if the key exists in the index.
        """
        if self.prepickle:
            key = pickle.dumps(key)
        return await self.keys.has_key(key)

    async def remove(self, key):
        """
        Remove the key from the index.

        Args:
            key (hashable): The unique identifier of a set.

        """
        if not await self.has_key(key):
            raise ValueError("The given key does not exist")
        if self.prepickle:
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
            raise ValueError("b must be less or equal to the number "
                             "of hash tables")
        candidates = set()
        for (start, end), hashtable in zip(self.hashranges[:b], self.hashtables[:b]):
            H = self._H(minhash.hashvalues[start:end])
            if await hashtable.has_key(H):
                for key in await hashtable.get(H):
                    candidates.add(key)

        if self.prepickle:
            return {pickle.loads(key) for key in candidates}
        else:
            return candidates

    async def get_counts(self):
        """
        Returns a list of length ``self.b`` with elements representing the
        number of keys stored under each bucket for the given permutation.
        """
        fs = (hashtable.itemcounts() for hashtable in self.hashtables)
        counts = await asyncio.gather(*fs)
        return counts

    async def get_subset_counts(self, *keys):
        """
        Returns the bucket allocation counts (see :ref:`get_counts` above)
        restricted to the list of keys given.

        Args:
            keys (hashable) : the keys for which to get the bucket allocation
                counts
        """
        if self.prepickle:
            key_set = [pickle.dumps(key) for key in set(keys)]
        else:
            key_set = list(set(keys))
        hashtables = [unordered_storage({'type': 'dict'}) for _ in range(self.b)]
        Hss = await self.keys.getmany(*key_set)
        for key, Hs in zip(key_set, Hss):
            for H, hashtable in zip(Hs, hashtables):
                hashtable.insert(H, key)
        return [hashtable.itemcounts() for hashtable in hashtables]


class AsyncMinHashLSHInsertionSession:
    """
    Asynchronous Context manager for batch insertion of documents
    into a MinHashLSH.
    """

    def __init__(self, lsh):
        self.lsh = lsh

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def close(self):
        await self.lsh.keys.empty_buffer()
        for hashtable in self.lsh.hashtables:
            await hashtable.empty_buffer()

    async def insert(self, key, minhash, check_duplication=True):
        """
        Insert a unique key to the index, together
        with a AsyncMinHash (or weighted AsyncMinHash) of the set referenced by
        the key.

        Args:
            :param minhash: (datasketch.AsyncMinHash) The AsyncMinHash
                of the set.
            :param key: (hashable) The unique identifier of the set.
            :param check_duplication: (default=True)
        """
        await self.lsh._insert(key, minhash,
                               check_duplication=check_duplication, buffer=True)
