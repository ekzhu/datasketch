import os
import asyncio
from itertools import islice
from functools import partial
import copy

from datasketch.storage import UnorderedStorage, OrderedStorage, _random_name
from abc import ABCMeta

ABC = ABCMeta('ABC', (object,), {})  # compatible with Python 2 *and* 3

try:
    import aioredis
except ImportError:
    aioredis = None

try:
    import motor.motor_asyncio
    from pymongo import ReturnDocument
except ImportError:
    motor = None
    ReturnDocument = None


async def async_ordered_storage(config, batch_size, name=None):
    tp = config['type']
    if tp == 'aioredis':
        if aioredis is None:
            raise RuntimeError('aioredis is not installed')
        return await AsyncRedisListStorage(config,
                                           name=name, batch_size=batch_size)
    elif tp == 'aiomongo':
        if motor is None:
            raise RuntimeError('motor is not installed')
        return AsyncMongoListStorage(config,
                                     name=name, batch_size=batch_size)

    raise ValueError('Unknown config ["type"]')


async def async_unordered_storage(config, batch_size, name=None):
    tp = config['type']
    if tp == 'aioredis':
        if aioredis is None:
            raise RuntimeError('aioredis is not installed')
        return await AsyncRedisSetStorage(config,
                                          name=name, batch_size=batch_size)
    elif tp == 'aiomongo':
        if motor is None:
            raise RuntimeError('motor is not installed')
        return AsyncMongoSetStorage(config,
                                    name=name, batch_size=batch_size)

    raise ValueError('Unknown config ["type"]')


if aioredis is not None:
    class AsyncRedisBuffer:
        def __init__(self, aio_redis, batch_size):
            self.batch_size = batch_size
            self._command_stack = tuple()
            self._redis = aio_redis

        async def execute_command(self, func, *args, **kwargs):
            if len(self._command_stack) >= self.batch_size:
                await self.execute()
            self._command_stack += (func(*args, **kwargs),)

        async def execute(self):
            if self._command_stack:
                _buffer = tuple(self._command_stack)
                self._command_stack = tuple()
                await asyncio.gather(*_buffer)

        async def hset(self, *args, **kwargs):
            await self.execute_command(self._redis.hset, *args, **kwargs)

        async def rpush(self, *args, **kwargs):
            await self.execute_command(self._redis.rpush, *args, **kwargs)

        async def sadd(self, *args, **kwargs):
            await self.execute_command(self._redis.sadd, *args, **kwargs)


    class AsyncRedisStorage(object):
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

        def __init__(self, config, name=None, batch_size=50000):
            self.config = config
            self._batch_size = batch_size
            self._redis_param = self._parse_config(self.config['redis'])

            if name is None:
                name = _random_name(11)
            self._name = name

            self._lock = asyncio.Lock()
            self._initialized = False
            self._redis = None
            self._buffer = None

        async def __async_init(self):
            async with self._lock:
                if not self._initialized:
                    await self.create_redis()
                self._initialized = True
            return self

        async def create_redis(self):
            db = self.redis_param['db'] if 'db' in self.redis_param else None
            dsn = 'redis://{host}:{port}'.format(**self.redis_param)
            self._redis = await aioredis.create_redis(dsn, db=db)
            self._buffer = AsyncRedisBuffer(self._redis, self._batch_size)

        def __await__(self):
            return self.__async_init().__await__()

        async def close(self):
            async with self._lock:
                await self._buffer.execute()
                self._redis.close()
                await self._redis.wait_closed()

                self._initialized = False

        @property
        def batch_size(self):
            return self._batch_size

        @property
        def redis_param(self):
            return self._redis_param

        def redis_key(self, key):
            return self._name + key

        @property
        def initialized(self):
            return self._initialized

        @staticmethod
        def _parse_config(config):
            cfg = {}
            for key, value in config.items():
                # If the value is a plain str, we will use the value
                # If the value is a dict, we will extract the name of an
                # environment variable stored under 'env' and optionally
                # a default, stored under 'default'.
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
            state.pop('_lock')
            state['_initialized'] = False
            return state

        def __setstate__(self, state):
            self.__dict__ = state
            self.__init__(self.config, name=self._name)

        async def __aenter__(self):
            return await self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            await self.close()


    class AsyncRedisListStorage(OrderedStorage, AsyncRedisStorage):
        def keys(self):
            return self._redis.hkeys(self._name)

        def redis_keys(self):
            return self._redis.hvals(self._name)

        async def status(self):
            status = self._parse_config(self.config['redis'])
            status.update({'keyspace_size': await self.size()})
            return status

        async def get(self, key):
            return await self._get_items(self._redis, self.redis_key(key))

        async def getmany(self, *keys):
            _command_stack = (self._redis.lrange(self.redis_key(key), 0, -1)
                              for key in keys)
            return await asyncio.gather(*_command_stack)

        @staticmethod
        async def _get_items(r, k):
            return await r.lrange(k, 0, -1)

        async def remove(self, *keys):
            await self._redis.hdel(self._name, *keys)
            await self._redis.delete(*(self.redis_key(key) for key in keys))

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
            _command_stack = (self._get_len(self._redis, self.redis_key(k))
                              for k in ks)
            return dict(zip(ks, await asyncio.gather(*_command_stack)))

        @staticmethod
        async def _get_len(r, k):
            return await r.llen(k)

        async def empty_buffer(self):
            await self._buffer.execute()

        async def has_key(self, key):
            return await self._redis.hexists(self._name, key)


    class AsyncRedisSetStorage(UnorderedStorage, AsyncRedisListStorage):
        async def get(self, key):
            return set(await AsyncRedisListStorage.get(self, key))

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

if motor is not None and ReturnDocument is not None:

    class AsyncMongoBuffer:
        def __init__(self, aio_mongo_collection, batch_size):
            self.batch_size = batch_size
            self._commands_col = 0
            self._mongo_coll = aio_mongo_collection
            self._bulk = self._mongo_coll.initialize_unordered_bulk_op()

        async def execute_command(self, command_name, **kwargs):
            if self._commands_col >= self.batch_size:
                await self.execute()
            self._commands_col += 1
            if command_name == 'find_one_and_update':
                self._bulk.find(kwargs['filter']).upsert().update(kwargs['update'])
            elif command_name == 'insert_one':
                self._bulk.insert(kwargs['document'])

        async def execute(self):
            if self._commands_col:
                task = self._bulk.execute()
                self._bulk, self._commands_col = self._mongo_coll.initialize_unordered_bulk_op(), 0
                await asyncio.gather(task)

        async def insert_one(self, **kwargs):
            await self.execute_command('insert_one', **kwargs)

        async def find_one_and_update(self, **kwargs):
            await self.execute_command('find_one_and_update', **kwargs)


    class AsyncMongoStorage(object):
        """Base class for asynchronous Mongo-based storage containers.

        Args:
            config (dict): Redis storage units require a configuration
                of the form::

                    storage_config={
                        'type': 'aiomongo',
                        'mongo': {'host': 'localhost', 'port': 27017}
                    }

                one can refer to system environment variables via::

                    storage_config={
                        'type': 'aiomongo',
                        'mongo': {
                            'host': {'env': 'MONGO_HOSTNAME',
                                     'default':'localhost'},
                            'port': 27017}
                        }
                    }

            name (bytes, optional): A prefix to namespace all keys in
                the database pertaining to this storage container.
                If None, a random name will be chosen.
        """

        def __init__(self, config, name=None, batch_size=10000):
            assert config['type'] == 'aiomongo', 'Storage type <{}> not supported'.format(config['type'])
            self._config = config
            # self._batch_size = batch_size
            self._mongo_param = self._parse_config(self._config['mongo'])

            self._name = name if name else _random_name(11).decode('utf-8')
            self._collection_name = 'lsh_' + self._name

            db_lsh = self.mongo_param['db'] if 'db' in self.mongo_param else 'db_0'
            if 'username' in self.mongo_param or 'password' in self.mongo_param:
                dsn = 'mongodb://{username}:{password}@{host}:{port}'.format(**self.mongo_param)
            else:
                dsn = 'mongodb://{host}:{port}'.format(**self.mongo_param)

            self._mongo_client = motor.motor_asyncio.AsyncIOMotorClient(dsn)
            self._collection = self._mongo_client[db_lsh][self._collection_name]
            self._initialized = True
            self._buffer = AsyncMongoBuffer(self._collection, batch_size)

        async def close(self):
            await self._buffer.execute()
            self._mongo_client.close()

        @property
        def initialized(self):
            return self._initialized

        @property
        def mongo_param(self):
            return self._mongo_param

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
            state.pop('_mongo_client')
            state.pop('_collection')
            state.pop('_buffer')
            state['_initialized'] = False
            return state

        def __setstate__(self, state):
            self.__dict__ = state
            self.__init__(self._config, name=self._name)


    class AsyncMongoListStorage(OrderedStorage, AsyncMongoStorage):
        async def keys(self):
            return [doc['key'] async for doc in self._collection.find(projection={'_id': False, 'vals': False})]

        async def get(self, key: str):
            doc = await self._collection.find_one({'key': key}, projection={'_id': False})
            return doc['vals'] if doc else []

        async def insert(self, key, *vals, **kwargs):
            buffer = kwargs.pop('buffer', False)
            if buffer:
                await self._insert(self._buffer, key, *vals)
            else:
                await self._insert(self._collection, key, *vals)

        async def _insert(self, obj, key, *values):
            await obj.insert_one(document={'key': key, 'vals': values})

        async def remove(self, *keys):
            for key in keys:
                await self._collection.find_one_and_delete({'key': key}, projection={'_id': False})

        async def remove_val(self, key: str, val):
            return await self._collection.find_one_and_update({'key': key},
                                                              {'$pull': {'vals': val}},
                                                              projection={'_id': False},
                                                              return_document=ReturnDocument.AFTER)

        async def size(self):
            return await self._collection.count()

        async def itemcounts(self):
            return {doc['key']: doc['count'] async for doc in
                    self._collection.aggregate([{'$project': {'key': 1, 'count': {'$size': '$vals'}}}])}

        async def has_key(self, key):
            return True if await self._collection.find_one({'key': key}) else False

        async def status(self):
            status = self._parse_config(self.config['mongo'])
            status.update({'keyspace_size': await self.size()})
            return status

        async def empty_buffer(self):
            await self._buffer.execute()


    class AsyncMongoSetStorage(UnorderedStorage, AsyncMongoListStorage):
        async def get(self, key):
            return frozenset(await AsyncMongoListStorage.get(self, key))

        async def _insert(self, obj, key, *values):
            await obj.find_one_and_update(filter={'key': key}, update={'$addToSet': {'vals': {'$each': values}}},
                                          upsert=True)
