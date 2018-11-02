import sys

if sys.version_info < (3, 6):
    raise ImportError("Can't use AsyncMinHashLSH module. Python version should be >=3.6")

import asyncio

from typing import Tuple, Dict
from itertools import chain
from datasketch.experimental.aio.storage import (async_ordered_storage, async_unordered_storage, )

from datasketch.lsh import _optimal_param
from datasketch.storage import unordered_storage, _random_name


class AsyncMinHashLSH(object):
    """
    Asynchronous MinHashLSH index.

    :param float threshold: see :class:`datasketch.MinHashLSH`.
    :param int num_perm: see :class:`datasketch.MinHashLSH`.
    :param weights: see :class:`datasketch.MinHashLSH`.
    :type weights: tuple(float, float)
    :param tuple params: see :class:`datasketch.MinHashLSH`.
    :param dict storage_config: New type of storage service - aiomongo - to use for storing
                                hashtables and keys are implemented.
                                If storage_config is None aiomongo storage will be used.

    For example usage see :ref:`minhash_lsh_async`.

    Example of supported storage configuration:

    .. code-block:: python

        MONGO = {'type': 'aiomongo', 'basename': 'base_name_1', 'mongo': {'host': 'localhost', 'port': 27017}}

    .. note::
        * The module supports Python version >=3.6, and is currently experimental. So the interface may change slightly in the future.
        * For main functionality of LSH algorithm see :class:`datasketch.MinHashLSH`.
        * For additional information see :ref:`minhash_lsh_at_scale` and :ref:`minhash_lsh_async`
    """

    def __init__(self, threshold: float=0.9, num_perm: int=128, weights: Tuple[float, float] =(0.5, 0.5),
                 params: Tuple[int, int]=None, storage_config: Dict=None):
        if storage_config is None:
            storage_config = {
                'type': 'aiomongo',
                'mongo': {'host': 'localhost', 'port': 27017}
            }
        self._storage_config = storage_config.copy()
        self._storage_config['basename'] = self._storage_config.get('basename', _random_name(11))
        self._basename = self._storage_config['basename']
        self._batch_size = 10000
        self._threshold = threshold
        self._num_perm = num_perm
        self._weights = weights
        self._params = params

        if self._threshold > 1.0 or self._threshold < 0.0:
            raise ValueError("threshold must be in [0.0, 1.0]")
        if self._num_perm < 2:
            raise ValueError("Too few permutation functions")
        if any(w < 0.0 or w > 1.0 for w in self._weights):
            raise ValueError("Weight must be in [0.0, 1.0]")
        if sum(self._weights) != 1.0:
            raise ValueError("Weights must sum to 1.0")
        self.h = self._num_perm
        if self._params is not None:
            self.b, self.r = self._params
            if self.b * self.r > self._num_perm:
                raise ValueError("The product of b and r must be less than "
                                 "num_perm")
        else:
            false_positive_weight, false_negative_weight = self._weights
            self.b, self.r = _optimal_param(self._threshold, self._num_perm,
                                            false_positive_weight, false_negative_weight)

        self.hashranges = [(i * self.r, (i + 1) * self.r)
                           for i in range(self.b)]
        self.hashtables = None
        self.keys = None

        self._lock = asyncio.Lock()
        self._initialized = False

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
        state.pop('hashranges')
        state.pop('hashtables')
        state.pop('keys')
        return state

    def __setstate__(self, state):
        state['_lock'] = asyncio.Lock()
        self.__dict__ = state
        self.__init__(self._threshold, self._num_perm, self._weights, self._params, self._storage_config)

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        if self.keys is not None:
            self.keys.batch_size = value
        else:
            raise AttributeError('AsyncMinHash is not initialized.')

        for t in self.hashtables:
            t.batch_size = value

        self._batch_size = value

    async def _create_storages(self):
        if self._storage_config['type'] == 'aioredis':
            name_ordered = b''.join([self._basename, b'_keys'])
            fs = (
                async_unordered_storage(
                    config=self._storage_config,
                    name=b''.join([self._basename, b'_bucket_', bytes([i])]),
                )
                for i in range(self.b)
            )
        else:
            name_ordered = ''.join([self._basename.decode('utf-8'), '_keys'])
            fs = (
                async_unordered_storage(
                    config=self._storage_config,
                    name=''.join([self._basename.decode('utf-8'), '_bucket_', str(i)])
                )
                for i in range(self.b)
            )

        fs = chain(fs, (async_ordered_storage(self._storage_config,
                                              name=name_ordered),))
        storages = await asyncio.gather(*fs)
        *self.hashtables, self.keys = storages

    async def init_storages(self):
        if self.keys is None:
            await self._create_storages()

        if not self.keys.initialized:
            await self.keys

        fs = (ht for ht in self.hashtables if not ht.initialized)
        await asyncio.gather(*fs)

    async def close(self):
        """
        Cleanup client resources and disconnect from AsyncMinHashLSH storage.
        """
        async with self._lock:
            for t in self.hashtables:
                await t.close()

            if self.keys is not None:
                await self.keys.close()

            self._initialized = False

    async def insert(self, key, minhash, check_duplication=True):
        """
        see :class:`datasketch.MinHashLSH`.
        """
        await self._insert(key, minhash, check_duplication=check_duplication, buffer=False)

    def insertion_session(self, batch_size=10000):
        """
        Create a asynchronous context manager for fast insertion in index.

        :param int batch_size: the size of chunks to use in insert_session mode (default=10000).

        :return: datasketch.experimental.aio.lsh.AsyncMinHashLSHSession

        Example:
            .. code-block:: python

                from datasketch.experimental.aio.lsh import AsyncMinHashLSH
                from datasketch import MinHash

                def chunk(it, size):
                    it = iter(it)
                    return iter(lambda: tuple(islice(it, size)), ())

                _chunked_str = chunk((random.choice(string.ascii_lowercase) for _ in range(10000)), 4)
                seq = frozenset(chain((''.join(s) for s in _chunked_str), ('aahhb', 'aahh', 'aahhc', 'aac', 'kld', 'bhg', 'kkd', 'yow', 'ppi', 'eer')))
                objs = [MinHash(16) for _ in range(len(seq))]
                for e, obj in zip(seq, objs):
                    for i in e:
                        obj.update(i.encode('utf-8'))
                data = [(e, m) for e, m in zip(seq, objs)]

                _storage_config_redis = {'type': 'aiomongo', 'mongo': {'host': 'localhost', 'port': 27017}}
                async def func():
                    async with AsyncMinHashLSH(storage_config=_storage_config_redis, threshold=0.5, num_perm=16) as lsh:
                        async with lsh.insertion_session(batch_size=1000) as session:
                            fs = (session.insert(key, minhash, check_duplication=True) for key, minhash in data)
                            await asyncio.gather(*fs)
        """
        return AsyncMinHashLSHInsertionSession(self, batch_size=batch_size)

    def delete_session(self, batch_size=10000):
        """
        Create a asynchronous context manager for fast removal of keys
        from index.

        :param int batch_size: the size of chunks to use in insert_session mode (default=10000).

        :return: datasketch.experimental.aio.lsh.AsyncMinHashLSHSession

        Example:
            .. code-block:: python

                from datasketch.experimental.aio.lsh import AsyncMinHashLSH
                from datasketch import MinHash

                def chunk(it, size):
                    it = iter(it)
                    return iter(lambda: tuple(islice(it, size)), ())

                _chunked_str = chunk((random.choice(string.ascii_lowercase) for _ in range(10000)), 4)
                seq = frozenset(chain((''.join(s) for s in _chunked_str), ('aahhb', 'aahh', 'aahhc', 'aac', 'kld', 'bhg', 'kkd', 'yow', 'ppi', 'eer')))
                objs = [MinHash(16) for _ in range(len(seq))]
                for e, obj in zip(seq, objs):
                    for i in e:
                        obj.update(i.encode('utf-8'))
                data = [(e, m) for e, m in zip(seq, objs)]

                _storage_config_redis = {'type': 'aiomongo', 'mongo': {'host': 'localhost', 'port': 27017}}
                async def func():
                    async with AsyncMinHashLSH(storage_config=_storage_config_redis, threshold=0.5, num_perm=16) as lsh:
                        async with lsh.insertion_session(batch_size=1000) as session:
                            fs = (session.insert(key, minhash, check_duplication=True) for key, minhash in data)
                            await asyncio.gather(*fs)

                        async with lsh.delete_session(batch_size=3) as session:
                            fs = (session.remove(key) for key in keys_to_remove)
                            await asyncio.gather(*fs)

        """
        return AsyncMinHashLSHDeleteSession(self, batch_size=batch_size)

    async def _insert(self, key, minhash, check_duplication=True, buffer=False):
        if len(minhash) != self.h:
            raise ValueError("Expecting minhash with length %d, "
                             "got %d" % (self.h, len(minhash)))
        if check_duplication and await self.has_key(key):
            raise ValueError("The given key already exists")
        Hs = [self._H(minhash.hashvalues[start:end])
              for start, end in self.hashranges]

        fs = chain((self.keys.insert(key, *Hs, buffer=buffer),),
                   (hashtable.insert(H, key, buffer=buffer) for H, hashtable in zip(Hs, self.hashtables)))
        await asyncio.gather(*fs)

    async def query(self, minhash):
        """
        see :class:`datasketch.MinHashLSH`.
        """
        if len(minhash) != self.h:
            raise ValueError("Expecting minhash with length %d, "
                             "got %d" % (self.h, len(minhash)))

        fs = (hashtable.get(self._H(minhash.hashvalues[start:end]))
              for (start, end), hashtable in zip(self.hashranges, self.hashtables))
        candidates = frozenset(chain.from_iterable(await asyncio.gather(*fs)))

        return list(candidates)

    async def has_key(self, key):
        """
        see :class:`datasketch.MinHashLSH`.
        """
        return await self.keys.has_key(key)

    async def remove(self, key):
        """
        see :class:`datasketch.MinHashLSH`.
        """
        await self._remove(key, buffer=False)

    async def _remove(self, key, buffer=False):
        if not await self.has_key(key):
            raise ValueError("The given key does not exist")

        for H, hashtable in zip(await self.keys.get(key), self.hashtables):
            await hashtable.remove_val(H, key, buffer=buffer)
            if not await hashtable.get(H):
                await hashtable.remove(H, buffer=buffer)

        await self.keys.remove(key, buffer=buffer)

    async def is_empty(self):
        """
        see :class:`datasketch.MinHashLSH`.
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
        fs = []
        for (start, end), hashtable in zip(self.hashranges[:b], self.hashtables[:b]):
            H = self._H(minhash.hashvalues[start:end])
            if await hashtable.has_key(H):
                fs.append(hashtable.get(H))
        candidates = set(chain.from_iterable(await asyncio.gather(*fs)))

        return candidates

    async def get_counts(self):
        """
        see :class:`datasketch.MinHashLSH`.
        """
        fs = (hashtable.itemcounts() for hashtable in self.hashtables)
        return await asyncio.gather(*fs)

    async def get_subset_counts(self, *keys):
        """
        see :class:`datasketch.MinHashLSH`.
        """
        key_set = list(set(keys))
        hashtables = [unordered_storage({'type': 'dict'}) for _ in range(self.b)]
        Hss = await self.keys.getmany(*key_set)
        for key, Hs in zip(key_set, Hss):
            for H, hashtable in zip(Hs, hashtables):
                hashtable.insert(H, key)
        return [hashtable.itemcounts() for hashtable in hashtables]


class AsyncMinHashLSHInsertionSession:
    """
    Context manager for batch insertion.
    """

    def __init__(self, lsh: AsyncMinHashLSH, batch_size: int):
        self.lsh = lsh
        self.lsh.batch_size = batch_size

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def close(self):
        fs = chain((self.lsh.keys.empty_buffer(),),
                   (hashtable.empty_buffer() for hashtable in self.lsh.hashtables))
        await asyncio.gather(*fs)

    async def insert(self, key, minhash, check_duplication=True):
        """
        see :class:`datasketch.MinHashLSH`.
        """
        await self.lsh._insert(key, minhash,
                               check_duplication=check_duplication, buffer=True)


class AsyncMinHashLSHDeleteSession:
    """
    Context manager for batch removal of keys.
    """

    def __init__(self, lsh: AsyncMinHashLSH, batch_size: int):
        self.lsh = lsh
        self.lsh.batch_size = batch_size

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def close(self):
        fs = chain((self.lsh.keys.empty_buffer(),),
                   (hashtable.empty_buffer() for hashtable in self.lsh.hashtables))
        await asyncio.gather(*fs)

    async def remove(self, key):
        """
        Remove key from LSH index
        """
        await self.lsh._remove(key, buffer=True)
