import os
import random
import string
import asyncio
from abc import ABCMeta

ABC = ABCMeta('ABC', (object,), {})  # compatible with Python 2 *and* 3
try:
    import aioredis
except ImportError:
    aioredis = None


async def async_ordered_storage(config, name=None):
    tp = config['type']
    if tp == 'aioredis':
        return await create_AsyncRedisListStorage(config, name=name)


async def async_unordered_storage(config, name=None):
    tp = config['type']
    if tp == 'aioredis':
        return await create_AsyncRedisSetStorage(config, name=name)


if aioredis is not None:
    async def create_AsyncRedisListStorage(config, name=None):
        redis_async = AsyncRedisListStorage(config, name=name)
        await redis_async.create_redis()
        return redis_async

    async def create_AsyncRedisSetStorage(config, name=None):
        redis_async = AsyncRedisSetStorage(config, name=name)
        await redis_async.create_redis()
        return redis_async

    class AsyncRedisBuffer:
        def __init__(self, aioredis, buffer_size=50000):
            self.buffer_size = buffer_size
            self._command_stack = []
            self._redis = aioredis

        async def execute_command(self, func, *args, **kwargs):
            if len(self._command_stack) >= self.buffer_size:
                await self.execute()
            self._command_stack.append(func(*args, **kwargs))

        async def execute(self):
            if self._command_stack:
                await asyncio.gather(*self._command_stack)
                self._command_stack = []

        async def hset(self, *args, **kwargs):
            await self.execute_command(self._redis.hset, *args, **kwargs)

        async def rpush(self, *args, **kwargs):
            await self.execute_command(self._redis.rpush, *args, **kwargs)

        async def sadd(self, *args, **kwargs):
            await self.execute_command(self._redis.sadd, *args, **kwargs)


    class AsyncRedisStorage:
        """Base class for asynchronous Redis-based storage containers.

        Args:
            config (dict): Redis storage units require a configuration
                of the form::

                    storage_config={
                        'type': 'aioredis',
                        'redis': {'host': 'localhost', 'port': 6379}
                    }

                one can refer to system environment variables via::

                    storage_config={
                        'type': 'aioredis',
                        'redis': {
                            'host': {'env': 'REDIS_HOSTNAME',
                                     'default':'localhost'},
                            'port': 6379}
                        }
                    }

            name (bytes, optional): A prefix to namespace all keys in
                the database pertaining to this storage container.
                If None, a random name will be chosen.
        """

        def __init__(self, config, name: bytes=None, batch_size=50000):
            self.config = config
            self._batch_size = batch_size
            self._redis_param = self._parse_config(self.config['redis'])

            if name is None:
                name = _random_name(11).decode('utf-8')
            self._name = name

        async def create_redis(self):
            db = self.redis_param['db'] if 'db' in self.redis_param else None
            self._redis = await aioredis.create_redis('redis://{host}:{port}'.format(**self.redis_param), db=db)
            self._buffer = AsyncRedisBuffer(self._redis)

        async def close(self):
            await self._buffer.execute()
            self._redis.close()
            await self._redis.wait_closed()

        @property
        def batch_size(self):
            return self._batch_size

        @property
        def redis_param(self):
            return self._redis_param

        def redis_key(self, key):
            return self._name + key

        @staticmethod
        def _parse_config(config):
            cfg = {}
            for key, value in config.items():
                # If the value is a plain str, we will use the value
                # If the value is a dict, we will extract the name of an environment
                # variable stored under 'env' and optionally a default, stored under
                # 'default'.
                # (This is useful if the database relocates to a different host
                # during the lifetime of the LSH object)
                if isinstance(value, dict):
                    if 'env' in value:
                        value = os.getenv(value['env'], value.get('default', None))
                cfg[key] = value
            return cfg

        def __getstate__(self):
            state = self.__dict__.copy()
            # We cannot pickle the connection objects, they get recreated
            # upon unpickling
            state.pop('_redis')
            state.pop('_buffer')
            return state

        def __setstate__(self, state):
            """Important! Initialize storage's after pickle.load/s.

                    lsh = await create_AsyncMinHashLSH(storage_config=self._storage_config, threshold=0.5, num_perm=16)
                    m1 = MinHash(16)
                    m1.update("a".encode("utf8"))
                    m2 = MinHash(16)
                    m2.update("b".encode("utf8"))
                    await lsh.insert("a", m1)
                    await lsh.insert("b", m2)
                    lsh2 = pickle.loads(pickle.dumps(lsh))
                    await lsh2.init_storages()
                    await lsh.close()

                    result = await lsh2.query(m1)
                    self.assertTrue("a" in result)
                    result = await lsh2.query(m2)
                    self.assertTrue("b" in result)
                    await lsh2.close()

               Don't forget to call class close() method after all calculations.
            """
            self.__dict__ = state
            self.__init__(self.config, name=self._name)


    class AsyncRedisListStorage(AsyncRedisStorage):
        def __init__(self, config, name=None):
            AsyncRedisStorage.__init__(self, config, name=name)

        async def keys(self):
            return await self._redis.hkeys(self._name)

        async def redis_keys(self):
            return await self._redis.hvals(self._name)

        async def status(self):
            status = self._parse_config(self.config['redis'])
            status.update({'keyspace_size': await self.size()})
            return status

        async def get(self, key):
            return await self._get_items(self._redis, self.redis_key(key))

        async def getmany(self, *keys):
            _command_stack = [self._redis.lrange(self.redis_key(key), 0, -1) for key in keys]
            return await asyncio.gather(*_command_stack)

        @staticmethod
        async def _get_items(r, k):
            return await r.lrange(k, 0, -1)

        async def remove(self, *keys):
            await self._redis.hdel(self._name, *keys)
            await self._redis.delete(*[self.redis_key(key) for key in keys])

        async def remove_val(self, key, val):
            redis_key = self.redis_key(key)
            await self._redis.lrem(redis_key, val)
            if not await self._redis.exists(redis_key):
                await self._redis.hdel(self._name, redis_key)

        async def insert(self, key, *vals, **kwargs):
            # Using buffer=True outside of an `insertion_session`
            # could lead to inconsistencies, because those
            # insertion will not be processed until the
            # buffer is cleared
            buffer = kwargs.pop('buffer', False)
            if buffer:
                await self._insert(self._buffer, key, *vals)
            else:
                await self._insert(self._redis, key, *vals)

        async def _insert(self, r, key, *values):
            redis_key = self.redis_key(key)
            await r.hset(self._name, key, redis_key)
            await r.rpush(redis_key, *values)

        async def size(self):
            return await self._redis.hlen(self._name)

        async def itemcounts(self):
            ks = await self.keys()
            _command_stack = [self._get_len(self._redis, self.redis_key(k)) for k in ks]
            return dict(zip(ks, await asyncio.gather(*_command_stack)))

        @staticmethod
        async def _get_len(r, k):
            return await r.llen(k)

        async def empty_buffer(self):
            await self._buffer.execute()

        async def has_key(self, key):
            return await self._redis.hexists(self._name, key)


    class AsyncRedisSetStorage(AsyncRedisListStorage):

        def __init__(self, config, name=None):
            AsyncRedisListStorage.__init__(self, config, name=name)

        @staticmethod
        async def _get_items(r, k):
            return await r.smembers(k)

        async def remove_val(self, key, val):
            redis_key = self.redis_key(key)
            await self._redis.srem(redis_key, val)
            if not await self._redis.exists(redis_key):
                await self._redis.hdel(self._name, redis_key)

        async def _insert(self, r, key, *values):
            redis_key = self.redis_key(key)
            await r.hset(self._name, key, redis_key)
            await r.sadd(redis_key, *values)

        @staticmethod
        async def _get_len(r, k):
            return await r.scard(k)


def _random_name(length):
    # For use with Redis, we return bytes
    return ''.join(random.choice(string.ascii_lowercase) for _ in range(length)).encode('utf8')
