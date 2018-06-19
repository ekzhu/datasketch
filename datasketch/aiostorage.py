import os
import random
import string
import asyncio
import aioredis
import motor.motor_asyncio
from pymongo import ReturnDocument
# from bson.objectid import ObjectId
# from bson import SON

from abc import ABCMeta, abstractmethod

ABC = ABCMeta('ABC', (object,), {})  # compatible with Python 2 *and* 3


async def async_ordered_storage(config, name=None):
    tp = config['type']
    if tp == 'aioredis':
        return await create_AsyncRedisListStorage(config, name=name)
    elif tp == 'aiomongo':
        return AsyncMongoListStorage(config, name=name)


async def async_unordered_storage(config, name=None):
    tp = config['type']
    if tp == 'aioredis':
        return await create_AsyncRedisSetStorage(config, name=name)
    elif tp == 'aiomongo':
        return AsyncMongoSetStorage(config, name=name)


class AsyncStorage(ABC):
    """Base class for key, value containers where the values are sequences."""

    @abstractmethod
    async def keys(self):
        """Return an iterator on keys in storage"""
        return []

    @abstractmethod
    async def get(self, key):
        """Get list of values associated with a key"""
        pass

    async def getmany(self, *keys):
        return [await self.get(key) for key in keys]

    @abstractmethod
    async def insert(self, key, *vals, **kwargs):
        """Add `val` to storage against `key`"""
        pass

    @abstractmethod
    async def remove(self, *keys):
        """Remove `keys` from storage"""
        pass

    @abstractmethod
    async def remove_val(self, key, val):
        """Remove `val` from list of values under `key`"""
        pass

    @abstractmethod
    async def size(self):
        """Return size of storage with respect to number of keys"""
        pass

    @abstractmethod
    async def itemcounts(self, **kwargs):
        """Returns the number of items stored under each key"""
        pass

    @abstractmethod
    async def has_key(self, key):
        """Determines whether the key is in the storage or not"""
        pass

    def status(self):
        return {'keyspace_size': len(self)}

    async def empty_buffer(self):
        pass


class AsyncOrderedStorage(AsyncStorage):
    pass


class AsyncUnorderedStorage(AsyncStorage):
    pass


if aioredis is not None:
    async def create_AsyncRedisListStorage(config, name=None):
        redis_async = AsyncRedisListStorage(config, name=name)
        await redis_async.init_storage()
        return redis_async


    async def create_AsyncRedisSetStorage(config, name=None):
        redis_async = AsyncRedisSetStorage(config, name=name)
        await redis_async.init_storage()
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

        def __init__(self, config, name=None, batch_size=50000):
            self.config = config
            self._batch_size = batch_size
            self._redis_param = self._parse_config(self.config['redis'])

            if name is None:
                name = _random_name(11)
            self._name = name

        async def init_storage(self):
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

                    lsh = await create_AsyncMinHashLSH(storage_config=self._storage_config_redis, threshold=0.5, num_perm=16)
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


    class AsyncRedisListStorage(AsyncOrderedStorage, AsyncRedisStorage):
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


    class AsyncRedisSetStorage(AsyncUnorderedStorage, AsyncRedisListStorage):
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

if motor.motor_asyncio is not None:
    # async def create_AsyncMongoListStorage(_config, name=None):
    #     redis_async = AsyncRedisListStorage(_config, name=name)
    #     await redis_async.init_storage()
    #     return redis_async
    #
    # async def create_AsyncMongoSetStorage(_config, name=None):
    #     redis_async = AsyncRedisSetStorage(_config, name=name)
    #     await redis_async.init_storage()
    #     return redis_async

    # class AsyncMongoBuffer:
    #     def __init__(self, aioredis, buffer_size=50000):
    #         self.buffer_size = buffer_size
    #         self._command_stack = []
    #         self._redis = aioredis
    #
    #     async def execute_command(self, func, *args, **kwargs):
    #         if len(self._command_stack) >= self.buffer_size:
    #             await self.execute()
    #         self._command_stack.append(func(*args, **kwargs))
    #
    #     async def execute(self):
    #         if self._command_stack:
    #             await asyncio.gather(*self._command_stack)
    #             self._command_stack = []
    #
    #     async def hset(self, *args, **kwargs):
    #         await self.execute_command(self._redis.hset, *args, **kwargs)
    #
    #     async def rpush(self, *args, **kwargs):
    #         await self.execute_command(self._redis.rpush, *args, **kwargs)
    #
    #     async def sadd(self, *args, **kwargs):
    #         await self.execute_command(self._redis.sadd, *args, **kwargs)

    class AsyncMongoStorage:
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

        def __init__(self, config, name: str=None):  # collection_name=None, batch_size=50000 , key_name=None
            assert config['type'] == 'aiomongo', 'Storage type <{}> not supported'.format(config['type'])
            self._config = config
            # self._batch_size = batch_size
            self._mongo_param = self._parse_config(self._config['mongo'])

            self._name = name if name else _random_name(11).decode('utf-8')
            self._collection_name = 'lsh_' + self._name

            db_lsh = self.mongo_param if 'db' in self.mongo_param else 'db_0'
            if 'username' in self.mongo_param or 'password' in self.mongo_param:
                conn_str = 'mongodb://{username}:{password}@{host}:{port}'.format(**self.mongo_param)
            else:
                conn_str = 'mongodb://{host}:{port}'.format(**self.mongo_param)

            self._collection = motor.motor_asyncio.AsyncIOMotorClient(conn_str)[db_lsh][self._collection_name]
            # self._mongodb = motor.motor_asyncio.AsyncIOMotorClient(conn_str)[db_lsh]
            # self._collection = self._mongodb[self._collection_name]
            # self._buffer = AsyncMongoBuffer(self._mongodb)

        async def close(self):
            pass
            # self._mongodb.close()

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

        # def mongo_key(self, key):
        #     return self._name + key

        def __getstate__(self):
            state = self.__dict__.copy()
            # We cannot pickle the connection objects, they get recreated
            # upon unpickling
            # state.pop('_mongodb')
            # state.pop('_buffer')
            state.pop('_collection')
            return state

        def __setstate__(self, state):
            self.__dict__ = state
            self.__init__(self._config, name=self._name)


    class AsyncMongoListStorage(AsyncOrderedStorage, AsyncMongoStorage):
        def __init__(self, config, name=None):
            AsyncMongoStorage.__init__(self, config, name=name)

        async def keys(self):
            return [doc['key'] async for doc in self._collection.find(projection={'_id': False, 'vals': False})]

        async def get(self, key: str):
            doc = await self._collection.find_one({'key': key}, projection={'_id': False})  # self.mongo_key(key)
            return doc['vals'] if doc else []

        async def insert(self, key, *vals, **kwargs):
            if await self._collection.find_one({'key': key}):  # self.mongo_key(key)
                await self._collection.find_one_and_update({'key': key},  # self.mongo_key(key)
                                                           {'$push': {'vals': {'$each': vals}}},
                                                           projection={'_id': False},
                                                           return_document=ReturnDocument.AFTER)
            else:
                await self._collection.insert_one({'key': key, 'vals': vals})  # self.mongo_key(key)

        async def remove(self, *keys):
            for key in keys:
                await self._collection.find_one_and_delete({'key': key}, projection={'_id': False})  # self.mongo_key(key)

        async def remove_val(self, key: str, val):
            return await self._collection.find_one_and_update({'key': key},  # self.mongo_key(key)
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


    class AsyncMongoSetStorage(AsyncUnorderedStorage, AsyncMongoListStorage):

        def __init__(self, config, name=None):
            AsyncMongoStorage.__init__(self, config, name=name)

        async def get(self, key):
            return set(await AsyncMongoListStorage.get(self, key))
            # doc = await self._collection.find_one({'key': key}, projection={'_id': False})
            # return doc['vals'] if doc else set()

        async def insert(self, key, *vals, **kwargs):
            if await self._collection.find_one({'key': key}):
                await self._collection.find_one_and_update({'key': key},
                                                           {'$addToSet': {'vals': {'$each': vals}}},
                                                           projection={'_id': False},
                                                           return_document=ReturnDocument.AFTER)
            else:
                await self._collection.insert_one({'key': key, 'vals': list(set(vals))})


def _random_name(length):
    # For use with Redis, we return bytes
    return ''.join(random.choice(string.ascii_lowercase) for _ in range(length)).encode('utf8')
