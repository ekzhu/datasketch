import sys

if sys.version_info >= (3, 6):
    python_version_supported = True
    import asyncio
    import pickle
    from itertools import chain
    from datasketch.experimental.aio.storage import (async_ordered_storage, async_unordered_storage, )

    from datasketch.lsh import _optimal_param
    from datasketch.storage import unordered_storage, _random_name
else:
    python_version_supported = False

if python_version_supported:
    class AsyncMinHashLSH(object):
        """
        Asynchronous MinHashLSH index.

        :param float threshold: see :class:`datasketch.MinHashLSH`.
        :param int num_perm: see :class:`datasketch.MinHashLSH`.
        :param weights: see :class:`datasketch.MinHashLSH`.
        :type weights: tuple(float, float)
        :param tuple params: see :class:`datasketch.MinHashLSH`.
        :param dict storage_config: New types of storage service (aioredis, aiomongo) to use for storing hashtables and keys are implemented. If storage_config is None aioredis storage will be used.
        :param bool prepickle: for redis type storage use bytes as keys.
        :param int batch_size: the size of chunks to use in insert_session mode.
        :param byte name: basename. Useful if you want pickle LSH class and spread it into different workers.

        Example:
            There is two ways to create a AsyncMinHash index.
                1. Usual way:

                    .. code-block:: python

                        from datasketch.experimental.aio import AsyncMinHashLSH

                        _storage_config_redis = {'type': 'aioredis', 'redis': {'host': 'localhost', 'port': 6379}}

                        lsh = await AsyncMinHashLSH(storage_config=_storage_config_redis, threshold=0.5, num_perm=16)
                        m1 = MinHash(16)
                        m1.update('a'.encode('utf8'))
                        m2 = MinHash(16)
                        m2.update('b'.encode('utf8'))
                        await lsh.insert('a', m1)
                        await lsh.insert('b', m2)
                        print(await lsh.query(m1))
                        print(await lsh.query(m2))
                        lsh.close()

                2. Context Manager style:

                    .. code-block:: python

                        from datasketch.experimental.aio import AsyncMinHashLSH

                        _storage_config_redis = {'type': 'aioredis', 'redis': {'host': 'localhost', 'port': 6379}}

                        aio with AsyncMinHashLSH(storage_config=_storage_config_redis, threshold=0.5, num_perm=16) as lsh:
                            m1 = MinHash(16)
                            m1.update('a'.encode('utf8'))
                            m2 = MinHash(16)
                            m2.update('b'.encode('utf8'))
                            await lsh.insert('a', m1)
                            await lsh.insert('b', m2)
                            print(await lsh.query(m1))
                            print(await lsh.query(m2))

        .. note::
            * The module supports Python version >=3.6, and is currently experimental. So the interface may change slightly in the future.
            * For main functionality of LSH algorithm see :class:`datasketch.MinHashLSH`.
            * For additional information see :ref:`minhash_lsh_at_scale` and :ref:`minhash_lsh_async`
        """

        def __init__(self, threshold=0.9, num_perm=128, weights=(0.5, 0.5),
                     params=None, storage_config=None, prepickle=None, batch_size=10000, name=None):
            if storage_config is None:
                storage_config = {
                    'type': 'aioredis',
                    'redis': {'host': 'localhost', 'port': 6379}
                }
            self._storage_config = storage_config
            self._batch_size = batch_size
            self._threshold = threshold
            self._num_perm = num_perm
            self._weights = weights
            self._params = params
            if prepickle is None:
                self._prepickle = storage_config['type'] == 'aioredis'
            else:
                self._prepickle = prepickle

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
            if name is None:
                name = _random_name(11)
            self._basename = name

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
            self.__init__(self._threshold, self._num_perm, self._weights, self._params, self._storage_config,
                          self._prepickle,  self._batch_size, self._basename)

        @property
        def batch_size(self):
            return self._batch_size

        @batch_size.setter
        def batch_size(self, value):
            self._batch_size = value

        async def _create_storages(self):
            if self._storage_config['type'] == 'aioredis':
                name_ordered = self._basename + b'_keys'
                fs = (
                    async_unordered_storage(
                        config=self._storage_config,
                        batch_size=self._batch_size,
                        name=self._basename + b'_bucket_' + bytes([i]),
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

            fs = chain(fs, (async_ordered_storage(self._storage_config,
                                                  batch_size=self._batch_size,
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

        def insertion_session(self):
            """
            Create a asynchronous context manager for fast insertion into this index.

            :return: datasketch.experimental.aio.lsh.AsyncMinHashLSHInsertionSession

            Example:
                .. code-block:: python

                    from datasketch.experimental.aio import AsyncMinHashLSH

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

                    _storage_config_redis = {'type': 'aioredis', 'redis': {'host': 'localhost', 'port': 6379}}
                    aio with AsyncMinHashLSH(storage_config=_storage_config_redis, threshold=0.5, num_perm=16, batch_size=1000) as lsh:
                        aio with lsh.insertion_session() as session:
                            fs = (session.insert(key, minhash, check_duplication=False) for key, minhash in data)
                            await asyncio.gather(*fs)

            """
            return AsyncMinHashLSHInsertionSession(self)

        async def _insert(self, key, minhash, check_duplication=True, buffer=False):
            if len(minhash) != self.h:
                raise ValueError("Expecting minhash with length %d, "
                                 "got %d" % (self.h, len(minhash)))
            if check_duplication and await self.has_key(key):
                raise ValueError("The given key already exists")
            Hs = [self._H(minhash.hashvalues[start:end])
                  for start, end in self.hashranges]
            if self._prepickle:
                key = pickle.dumps(key)
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

            if self._prepickle:
                return [pickle.loads(key) for key in candidates]
            else:
                return list(candidates)

        async def has_key(self, key):
            """
            see :class:`datasketch.MinHashLSH`.
            """
            if self._prepickle:
                key = pickle.dumps(key)
            return await self.keys.has_key(key)

        async def remove(self, key):
            """
            see :class:`datasketch.MinHashLSH`.
            """
            if not await self.has_key(key):
                raise ValueError("The given key does not exist")
            if self._prepickle:
                key = pickle.dumps(key)

            fs = chain.from_iterable((hashtable.remove_val(H, key), hashtable.remove(H))
                                     for H, hashtable in zip(await self.keys.get(key), self.hashtables))
            fs = chain((self.keys.remove(key),), fs)
            await asyncio.gather(*fs)

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

            if self._prepickle:
                return {pickle.loads(key) for key in candidates}
            else:
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
            if self._prepickle:
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
        see :func:`~datasketch.experimental.aio.lsh.AsyncMinHashLSH.insertion_session`
        """

        def __init__(self, lsh):
            self.lsh = lsh

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