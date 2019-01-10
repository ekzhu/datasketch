import sys

if sys.version_info < (3, 6):
    raise ImportError("Can't use AsyncMinHashLSH module. Python version should be >=3.6")

import os
import asyncio

from itertools import chain

from datasketch.storage import UnorderedStorage, OrderedStorage, _random_name
from abc import ABCMeta

ABC = ABCMeta('ABC', (object,), {})

try:
    import motor.motor_asyncio
    from pymongo import ReturnDocument
except ImportError:
    motor = None
    ReturnDocument = None


async def async_ordered_storage(config, name=None):
    tp = config['type']
    if tp == 'aiomongo':
        if motor is None:
            raise RuntimeError('motor is not installed')
        return AsyncMongoListStorage(config, name=name)

    raise ValueError('Unknown config ["type"]')


async def async_unordered_storage(config, name=None):
    tp = config['type']
    if tp == 'aiomongo':
        if motor is None:
            raise RuntimeError('motor is not installed')
        return AsyncMongoSetStorage(config, name=name)

    raise ValueError('Unknown config ["type"]')


if motor is not None and ReturnDocument is not None:
    class AsyncMongoBuffer:
        def __init__(self, aio_mongo_collection, batch_size):
            self._batch_size = batch_size
            self._insert_documents_stack = tuple()
            self._delete_by_key_documents_stack = tuple()
            self._delete_by_val_documents_stack = tuple()
            self._mongo_coll = aio_mongo_collection

        @property
        def batch_size(self):
            return self._batch_size

        @batch_size.setter
        def batch_size(self, value):
            self._batch_size = value

        async def execute_command(self, **kwargs):
            command = kwargs.pop('command')
            if command == 'insert':
                if len(self._insert_documents_stack) >= self.batch_size:
                    await self.execute(command)
                self._insert_documents_stack += (kwargs['obj'],)
            elif command == 'delete_by_key':
                if len(self._delete_by_key_documents_stack) >= self.batch_size:
                    await self.execute(command)
                self._delete_by_key_documents_stack += (kwargs['key'],)
            elif command == 'delete_by_val':
                if len(self._delete_by_val_documents_stack) >= self.batch_size:
                    await self.execute(command)
                self._delete_by_val_documents_stack += (kwargs['val'],)

        async def execute(self, command):
            if command == 'insert' and self._insert_documents_stack:
                buffer = self._insert_documents_stack
                self._insert_documents_stack = tuple()
                await self._mongo_coll.insert_many(buffer, ordered=False)
            elif command == 'delete_by_key' and self._delete_by_key_documents_stack:
                buffer = self._delete_by_key_documents_stack
                self._delete_by_key_documents_stack = tuple()
                await self._mongo_coll.delete_many({'key': {'$in': buffer}})
            elif command == 'delete_by_val' and self._delete_by_val_documents_stack:
                buffer = self._delete_by_val_documents_stack
                self._delete_by_val_documents_stack = tuple()
                await self._mongo_coll.delete_many({'vals': {'$in': buffer}})

        async def insert_one(self, **kwargs):
            await self.execute_command(obj=kwargs['document'], command='insert')

        async def delete_many_by_key(self, **kwargs):
            await self.execute_command(key=kwargs['key'], command='delete_by_key')

        async def delete_many_by_val(self, **kwargs):
            await self.execute_command(val=kwargs['val'], command='delete_by_val')


    class AsyncMongoStorage(object):
        """Base class for asynchronous MongoDB-based storage containers.

        :param dict config: MongoDB storage units require a configuration
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

        :param bytes name: see :class:`datasketch.storage.RedisStorage` (default = None).
        """

        def __init__(self, config, name=None):
            assert config['type'] == 'aiomongo', 'Storage type <{}> not supported'.format(config['type'])
            self._config = config
            self._mongo_param = self._parse_config(self._config['mongo'])

            self._name = name if name else _random_name(11).decode('utf-8')
            self._collection_name = 'lsh_' + self._name

            db_lsh = self.mongo_param['db'] if 'db' in self.mongo_param else 'db_0'
            if 'replica_set' in self.mongo_param:
                dsn = 'mongodb://{replica_set_nodes}/?replicaSet={replica_set}'.format(**self.mongo_param)
            elif 'username' in self.mongo_param or 'password' in self.mongo_param:
                dsn = 'mongodb://{username}:{password}@{host}:{port}'.format(**self.mongo_param)
            else:
                dsn = 'mongodb://{host}:{port}'.format(**self.mongo_param)

            additional_args = self.mongo_param.get('args', {})

            self._batch_size = 1000
            self._mongo_client = motor.motor_asyncio.AsyncIOMotorClient(dsn, **additional_args)
            self._collection = self._mongo_client[db_lsh][self._collection_name]
            self._initialized = True
            self._buffer = AsyncMongoBuffer(self._collection, self._batch_size)

        async def close(self):
            fs = (self._buffer.execute(command) for command in ('insert', 'delete_by_key', 'delete_by_val'))
            await asyncio.gather(*fs)
            self._mongo_client.close()

        @property
        def batch_size(self):
            return self._batch_size

        @batch_size.setter
        def batch_size(self, value):
            self._batch_size = value
            self._buffer.batch_size = value

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
            return list(chain.from_iterable([doc['vals'] async for doc in self._collection.find(filter={'key': key},
                                                                                                projection={
                                                                                                    '_id': False,
                                                                                                    'key': False})]))

        async def insert(self, key, *vals, **kwargs):
            buffer = kwargs.pop('buffer', False)
            if buffer:
                await self._insert(self._buffer, key, *vals)
            else:
                await self._insert(self._collection, key, *vals)

        async def _insert(self, obj, key, *values):
            await obj.insert_one(document={'key': key, 'vals': values})

        async def remove(self, *keys, **kwargs):
            buffer = kwargs.pop('buffer', False)
            if buffer:
                fs = (self._buffer.delete_many_by_key(key=key) for key in keys)
                await asyncio.gather(*fs)
            else:
                await self._collection.delete_many({'key': {'$in': keys}})

        async def remove_val(self, key, val, **kwargs):
            pass

        async def size(self):
            return await self._collection.count_documents({})

        async def itemcounts(self):
            return {doc['_id']: doc['count'] async for doc in
                    self._collection.aggregate([{'$group': {'_id': '$key', 'count': {'$sum': 1}}}])}

        async def has_key(self, key):
            return True if await self._collection.find_one({'key': key}) else False

        async def status(self):
            status = self._parse_config(self.config['mongo'])
            status.update({'keyspace_size': await self.size()})
            return status

        async def empty_buffer(self):
            fs = (self._buffer.execute(command) for command in ('insert', 'delete_by_key', 'delete_by_val'))
            await asyncio.gather(*fs)


    class AsyncMongoSetStorage(UnorderedStorage, AsyncMongoListStorage):
        async def get(self, key):
            keys = [doc['vals'] async for doc in self._collection.find(filter={'key': key},
                                                                       projection={'_id': False, 'key': False})]
            return frozenset(keys)

        async def _insert(self, obj, key, *values):
            await obj.insert_one(document={'key': key, 'vals': values[0]})

        async def remove(self, *keys, **kwargs):
            pass

        async def remove_val(self, key, val, **kwargs):
            buffer = kwargs.pop('buffer', False)
            if buffer:
                await self._buffer.delete_many_by_val(val=val)
            else:
                await self._collection.find_one_and_delete({'key': key, 'vals': val})
