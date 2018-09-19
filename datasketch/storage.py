from collections import defaultdict
import os
import random
import string
import psycopg2
import pickle
from pymemcache.client.base import Client

from abc import ABCMeta, abstractmethod
ABC = ABCMeta('ABC', (object,), {}) # compatible with Python 2 *and* 3
try:
    import redis
except ImportError:
    redis = None


def ordered_storage(config, name=None):
    '''Return ordered storage system based on the specified config.

    The canonical example of such a storage container is
    ``defaultdict(list)``. Thus, the return value of this method contains
    keys and values. The values are ordered lists with the last added
    item at the end.

    Args:
        config (dict): Defines the configurations for the storage.
            For in-memory storage, the config ``{'type': 'dict'}`` will
            suffice. For Redis storage, the type should be ``'redis'`` and
            the configurations for the Redis database should be supplied
            under the key ``'redis'``. These parameters should be in a form
            suitable for `redis.Redis`. The parameters may alternatively
            contain references to environment variables, in which case
            literal configuration values should be replaced by dicts of
            the form::

                {'env': 'REDIS_HOSTNAME',
                 'default': 'localhost'}

            For a full example, see :ref:`minhash_lsh_at_scale`

        name (bytes, optional): A reference name for this storage container.
            For dict-type containers, this is ignored. For Redis containers,
            this name is used to prefix keys pertaining to this storage
            container within the database.
    '''
    tp = config['type']
    if tp == 'dict':
        return DictListStorage(config)
    if tp == 'redis':
        return RedisListStorage(config, name=name)
    if tp == 'postgres':
        return PostgresListStorage(config, name=name)
    if tp == 'memcached':
        return MemcachedListStorage(config, name=name)


def unordered_storage(config, name=None):
    '''Return an unordered storage system based on the specified config.

    The canonical example of such a storage container is
    ``defaultdict(set)``. Thus, the return value of this method contains
    keys and values. The values are unordered sets.

    Args:
        config (dict): Defines the configurations for the storage.
            For in-memory storage, the config ``{'type': 'dict'}`` will
            suffice. For Redis storage, the type should be ``'redis'`` and
            the configurations for the Redis database should be supplied
            under the key ``'redis'``. These parameters should be in a form
            suitable for `redis.Redis`. The parameters may alternatively
            contain references to environment variables, in which case
            literal configuration values should be replaced by dicts of
            the form::

                {'env': 'REDIS_HOSTNAME',
                 'default': 'localhost'}

            For a full example, see :ref:`minhash_lsh_at_scale`

        name (bytes, optional): A reference name for this storage container.
            For dict-type containers, this is ignored. For Redis containers,
            this name is used to prefix keys pertaining to this storage
            container within the database.
    '''
    tp = config['type']
    if tp == 'dict':
        return DictSetStorage(config)
    if tp == 'redis':
        return RedisSetStorage(config, name=name)
    if tp == 'postgres':
        return PostgresSetStorage(config, name=name)
    if tp == 'memcached':
        return MemcachedSetStorage(config, name=name)


class Storage(ABC):
    '''Base class for key, value containers where the values are sequences.'''
    def __getitem__(self, key):
        return self.get(key)

    def __delitem__(self, key):
        return self.remove(key)

    def __len__(self):
        return self.size()

    def __iter__(self):
        for key in self.keys():
            yield key

    def __contains__(self, item):
        return self.has_key(item)

    @abstractmethod
    def keys(self):
        '''Return an iterator on keys in storage'''
        return []

    @abstractmethod
    def get(self, key):
        '''Get list of values associated with a key
        
        Returns empty list ([]) if `key` is not found
        '''
        pass

    def getmany(self, *keys):
        return [self.get(key) for key in keys]

    @abstractmethod
    def insert(self, key, *vals, **kwargs):
        '''Add `val` to storage against `key`'''
        pass

    @abstractmethod
    def remove(self, *keys):
        '''Remove `keys` from storage'''
        pass

    @abstractmethod
    def remove_val(self, key, val):
        '''Remove `val` from list of values under `key`'''
        pass

    @abstractmethod
    def size(self):
        '''Return size of storage with respect to number of keys'''
        pass

    @abstractmethod
    def itemcounts(self, **kwargs):
        '''Returns the number of items stored under each key'''
        pass

    @abstractmethod
    def has_key(self, key):
        '''Determines whether the key is in the storage or not'''
        pass

    def status(self):
        return {'keyspace_size': len(self)}

    def empty_buffer(self):
        pass


class OrderedStorage(Storage):

    pass


class UnorderedStorage(Storage):

    pass


class DictListStorage(OrderedStorage):
    '''This is a wrapper class around ``defaultdict(list)`` enabling
    it to support an API consistent with `Storage`
    '''
    def __init__(self, config):
        self._dict = defaultdict(list)

    def keys(self):
        return self._dict.keys()

    def get(self, key):
        return self._dict.get(key, [])

    def remove(self, *keys):
        for key in keys:
            del self._dict[key]

    def remove_val(self, key, val):
        self._dict[key].remove(val)

    def insert(self, key, *vals, **kwargs):
        self._dict[key].extend(vals)

    def size(self):
        return len(self._dict)

    def itemcounts(self, **kwargs):
        '''Returns a dict where the keys are the keys of the container.
        The values are the *lengths* of the value sequences stored
        in this container.
        '''
        return {k: len(v) for k, v in self._dict.items()}

    def has_key(self, key):
        return key in self._dict


class DictSetStorage(UnorderedStorage, DictListStorage):
    '''This is a wrapper class around ``defaultdict(set)`` enabling
    it to support an API consistent with `Storage`
    '''
    def __init__(self, config):
        self._dict = defaultdict(set)

    def get(self, key):
        return self._dict.get(key, set())

    def insert(self, key, *vals, **kwargs):
        self._dict[key].update(vals)


if redis is not None:
    class RedisBuffer(redis.client.Pipeline):
        '''A bufferized version of `redis.pipeline.Pipeline`.

        The only difference from the conventional pipeline object is the
        ``_buffer_size``. Once the buffer is longer than the buffer size,
        the pipeline is automatically executed, and the buffer cleared.
        '''

        def __init__(self, connection_pool, response_callbacks, transaction, buffer_size,
                     shard_hint=None):
            self._buffer_size = buffer_size
            super(RedisBuffer, self).__init__(
                connection_pool, response_callbacks, transaction,
                shard_hint=shard_hint)

        @property
        def buffer_size(self):
            return self._buffer_size

        @buffer_size.setter
        def buffer_size(self, value):
            self._buffer_size = value

        def execute_command(self, *args, **kwargs):
            if len(self.command_stack) >= self._buffer_size:
                self.execute()
            super(RedisBuffer, self).execute_command(*args, **kwargs)


    class RedisStorage:
        '''Base class for Redis-based storage containers.

        Args:
            config (dict): Redis storage units require a configuration
                of the form::

                    storage_config={
                        'type': 'redis',
                        'redis': {'host': 'localhost', 'port': 6379}
                    }

                one can refer to system environment variables via::

                    storage_config={
                        'type': 'redis',
                        'redis': {
                            'host': {'env': 'REDIS_HOSTNAME',
                                     'default':'localhost'},
                            'port': 6379}
                        }
                    }

            name (bytes, optional): A prefix to namespace all keys in
                the database pertaining to this storage container.
                If None, a random name will be chosen.
        '''

        def __init__(self, config, name=None):
            self.config = config
            self._buffer_size = 50000
            redis_param = self._parse_config(self.config['redis'])
            self._redis = redis.Redis(**redis_param)
            self._buffer = RedisBuffer(self._redis.connection_pool,
                                       self._redis.response_callbacks,
                                       transaction=True,
                                       buffer_size=self._buffer_size)
            if name is None:
                name = _random_name(11)
            self._name = name

        @property
        def buffer_size(self):
            return self._buffer_size

        @buffer_size.setter
        def buffer_size(self, value):
            self._buffer_size = value
            self._buffer.buffer_size = value

        def redis_key(self, key):
            return self._name + key

        def _parse_config(self, config):
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
            self.__dict__ = state
            # Reconnect here
            self.__init__(self.config, name=self._name)


    class RedisListStorage(OrderedStorage, RedisStorage):
        def __init__(self, config, name=None):
            RedisStorage.__init__(self, config, name=name)

        def keys(self):
            return self._redis.hkeys(self._name)

        def redis_keys(self):
            return self._redis.hvals(self._name)

        def status(self):
            status = self._parse_config(self.config['redis'])
            status.update(Storage.status(self))
            return status

        def get(self, key):
            return self._get_items(self._redis, self.redis_key(key))

        def getmany(self, *keys):
            pipe = self._redis.pipeline()
            pipe.multi()
            for key in keys:
                self._get_items(pipe, self.redis_key(key))
            return pipe.execute()

        @staticmethod
        def _get_items(r, k):
            return r.lrange(k, 0, -1)

        def remove(self, *keys):
            self._redis.hdel(self._name, *keys)
            self._redis.delete(*[self.redis_key(key) for key in keys])

        def remove_val(self, key, val):
            redis_key = self.redis_key(key)
            self._redis.lrem(redis_key, val)
            if not self._redis.exists(redis_key):
                self._redis.hdel(self._name, redis_key)

        def insert(self, key, *vals, **kwargs):
            # Using buffer=True outside of an `insertion_session`
            # could lead to inconsistencies, because those
            # insertion will not be processed until the
            # buffer is cleared
            buffer = kwargs.pop('buffer', False)
            if buffer:
                self._insert(self._buffer, key, *vals)
            else:
                self._insert(self._redis, key, *vals)

        def _insert(self, r, key, *values):
            redis_key = self.redis_key(key)
            r.hset(self._name, key, redis_key)
            r.rpush(redis_key, *values)

        def size(self):
            return self._redis.hlen(self._name)

        def itemcounts(self):
            pipe = self._redis.pipeline()
            pipe.multi()
            ks = self.keys()
            for k in ks:
                self._get_len(pipe, self.redis_key(k))
            d = dict(zip(ks, pipe.execute()))
            return d

        @staticmethod
        def _get_len(r, k):
            return r.llen(k)

        def has_key(self, key):
            return self._redis.hexists(self._name, key)

        def empty_buffer(self):
            self._buffer.execute()
            # To avoid broken pipes, recreate the connection
            # objects upon emptying the buffer
            self.__init__(self.config, name=self._name)


    class RedisSetStorage(UnorderedStorage, RedisListStorage):
        def __init__(self, config, name=None):
            RedisListStorage.__init__(self, config, name=name)

        @staticmethod
        def _get_items(r, k):
            return r.smembers(k)

        def remove_val(self, key, val):
            redis_key = self.redis_key(key)
            self._redis.srem(redis_key, val)
            if not self._redis.exists(redis_key):
                self._redis.hdel(self._name, redis_key)

        def _insert(self, r, key, *values):
            redis_key = self.redis_key(key)
            r.hset(self._name, key, redis_key)
            r.sadd(redis_key, *values)

        @staticmethod
        def _get_len(r, k):
            return r.scard(k)


class PostgresStorage:
    def __init__(self, config, name=None):
        self._name = name
        self._postgres_param = self._parse_config(config['postgres'])
        self._db_connection = psycopg2.connect(**self._postgres_param)

    def _postgres_key(self, key):
        newkey = self._name + key
        return newkey

    def _keys(self, table):
        sql = "SELECT KEY FROM {0}"
        with self._db_connection.cursor() as cursor:
            cursor.itersize = 100
            cursor.execute(sql.format(table),)
            for res in cursor:
                # Internal datasketch API requires keys method
                # to return key without basename
                yield bytes(res[0]).replace(self._name, b'')
        self._db_connection.commit()

    def _get(self, key, table):
        sql = "SELECT VALUE FROM {0} WHERE KEY=%s"
        with self._db_connection.cursor() as cursor:
            cursor.execute(sql.format(table), (self._postgres_key(key),))
            res = cursor.fetchone()
        self._db_connection.commit()
        if res is None:
            return list()
        res = [bytes(r) for r in res[0]]
        return res

    def _remove(self, table, *keys):
        sql = "DELETE FROM {0} WHERE KEY in (%s)"
        with self._db_connection.cursor() as cursor:
            cursor.execute(sql.format(table), (*[self._postgres_key(key) for key in keys],))
        self._db_connection.commit()

    def _remove_val(self, key, val, table):
        sql = "UPDATE {0} SET VALUE = array_remove(VALUE, %s) WHERE KEY = %s"
        with self._db_connection.cursor() as cursor:
            cursor.execute(sql.format(table), (val, self._postgres_key(key)))
        self._db_connection.commit()

    def _insert(self, key, table, *vals, **kwargs):
        sql = "INSERT INTO {0} VALUES (%s, %s)"
        with self._db_connection.cursor() as cursor:
            cursor.execute(sql.format(table),
                           (self._postgres_key(key), list(vals)))
        self._db_connection.commit()

    def _size(self, table):
        sql = "SELECT count(*) FROM {0}"
        with self._db_connection.cursor() as cursor:
            cursor.execute(sql.format(table),)
            res = int(cursor.fetchone()[0])
        self._db_connection.commit()
        return res

    def itemcounts(self, table, **kwargs):
        '''Returns a dict where the keys are the keys of the container.
        The values are the *lengths* of the value sequences stored
        in this container.
        '''
        # result = dict()
        # with self._db_connection.cursor(
        #         name="datasketch_named_cursor") as cursor:
        #     cursor.itersize = 100
        #     cursor.execute("SELECT KEY, array_length(VALUE, 1) "
        #                    "FROM DATASKETCH_BUCKETS")
        #     for record in cursor:
        #         result[bytes(record[0])] = record[1]
        # self._db_connection.commit()
        raise NotImplementedError

    def _has_key(self, key, table):
        sql = "SELECT KEY FROM {0} WHERE KEY=%s"
        if not type(key) is bytes:
            key = pickle.dumps(key)
        with self._db_connection.cursor() as cursor:
            cursor.execute(sql.format(table),
                           (self._postgres_key(key),))
            res = cursor.fetchall()
        self._db_connection.commit()
        if res:
            return True
        else:
            return False

    def _parse_config(self, config):
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


class PostgresListStorage(PostgresStorage, OrderedStorage):
    '''This is a wrapper class around ``defaultdict(list)`` enabling
    it to support an API consistent with `Storage`
    '''

    def __init__(self, config, name=None):
        PostgresStorage.__init__(self, config, name=name)
        self._table = "LSH_BUCKETS_ORDERED"

    def keys(self):
        for res in self._keys(self._table):
            yield res

    def get(self, key):
        return self._get(key, self._table)

    def remove(self, *keys):
        self._remove( self._table, *keys)

    def remove_val(self, key, val):
        self._remove_val(key, val, self._table)

    def insert(self, key, *vals, **kwargs):
        self._insert(key, self._table, *vals, **kwargs)

    def size(self):
        return self._size(self._table)

    def itemcounts(self, table, **kwargs):
        '''Returns a dict where the keys are the keys of the container.
        The values are the *lengths* of the value sequences stored
        in this container.
        '''

        raise NotImplementedError

    def has_key(self, key):
        return self._has_key(key, self._table)


class PostgresSetStorage(PostgresListStorage, OrderedStorage):
    '''This is a wrapper class around ``defaultdict(list)`` enabling
    it to support an API consistent with `Storage`
    '''
    def __init__(self, config, name=None):
        PostgresStorage.__init__(self, config, name=name)
        self._table = "LSH_BUCKETS_UNORDERED"

    def keys(self):
        for res in self._keys(self._table):
            yield res

    def get(self, key):
        return self._get(key, self._table)

    def remove(self, *keys):
        self._remove( self._table, *keys)

    def remove_val(self, key, val):
        self._remove_val(key, val, self._table)

    def insert(self, key, *vals, **kwargs):
        if self.has_key(key):
            bu = self.get(key)
            bu = set(bu) | set(vals)
            sql = "UPDATE {0} SET VALUE = %s WHERE KEY = %s"
            with self._db_connection.cursor() as cursor:
                cursor.execute(sql.format(self._table),
                               (list(bu), self._postgres_key(key)))
            self._db_connection.commit()
        else:
            self._insert(key, self._table, *vals, **kwargs)

    def size(self):
        return self._size(self._table)

    def itemcounts(self, **kwargs):
        '''Returns a dict where the keys are the keys of the container.
        The values are the *lengths* of the value sequences stored
        in this container.
        '''
        raise NotImplementedError

    def has_key(self, key):
        return self._has_key(key, self._table)


class MemcachedListStorage(OrderedStorage):
    def __init__(self, config, name=None):
        self._name = str(name)
        self._memcached_param = self._parse_config(config['memcached'])
        self._client = Client(self._memcached_param)

    def _memcached_key(self, key):
        if type(key) is bytes:
            key = key.hex()
        elif not type(key) is str:
            key = str(key)
        newkey = self._name + key
        return newkey

    def keys(self):
        hkeys = self._client.get(self._name)
        if hkeys is None:
            return []
        else:
            hkeys = pickle.loads(hkeys)
            return hkeys

    def get(self, key):
        values = self._client.get(self._memcached_key(key))
        if values is None:
            return []
        else:
            values = pickle.loads(values)
            return values

    def remove(self, *keys):
        self._client.delete_many(
            [self._memcached_key(key) for key in keys])
        hkeys = self._client.get(self._name)
        if hkeys is None:
            return
        else:
            hkeys = pickle.loads(hkeys)
            hkeys = [hkey for hkey in hkeys if hkey not in keys]
            hkeys = pickle.dumps(hkeys)
            self._client.set(self._name, hkeys)

    def remove_val(self, key, val):
        values = self._client.get(self._memcached_key(key))
        if values is None:
            return
        else:
            values = pickle.loads(values)
            values.remove(val)
            values = pickle.dumps(values)
            self._client.set(self._memcached_key(key), values)

    def insert(self, key, *vals, **kwargs):
        values = self._client.get(self._memcached_key(key))
        if values is None:
            self._client.set(self._memcached_key(key), vals)
            hkeys = self._client.get(self._name)
            if hkeys is None:
                hkeys = [key]
                hkeys = pickle.dumps(hkeys)
                self._client.set(self._name, hkeys)
            else:
                hkeys = pickle.loads(hkeys)
                hkeys.append(key)
                hkeys = pickle.dumps(hkeys)
                self._client.set(self._name, hkeys)
        else:
            values = pickle.loads(values)
            values.extend(vals)

    def size(self):
        return len(self.keys())

    def itemcounts(self, **kwargs):
        '''Returns a dict where the keys are the keys of the container.
        The values are the *lengths* of the value sequences stored
        in this container.
        '''

        raise NotImplementedError

    def has_key(self, key):
        return not self._client.get(self._memcached_key(key)) is None

    def _parse_config(self, config):
        cfg_tuple = (config['host'], config['port'])
        return cfg_tuple


class MemcachedSetStorage(MemcachedListStorage):
    def __init__(self, config, name=None):
        MemcachedListStorage.__init__(self, config, name=name)

    def get(self, key):
        values = self._client.get(self._memcached_key(key))
        if values is None:
            return {}
        else:
            values = pickle.loads(values)
            return values

    def remove(self, *keys):
        self._client.delete_many(
            [self._memcached_key(key) for key in keys])
        hkeys = self._client.get(self._name)
        hkeys = pickle.loads(hkeys)
        hkeys = hkeys - keys
        hkeys = pickle.dumps(hkeys)
        self._client.set(self._name, hkeys)

    def insert(self, key, *vals, **kwargs):
        values = self._client.get(self._memcached_key(key))
        if values is None:
            self._client.set(self._memcached_key(key), pickle.dumps(set(vals)))
            hkeys = self._client.get(self._name)
            if hkeys is None:
                hkeys = [key]
                hkeys = pickle.dumps(hkeys)
                self._client.set(self._name, hkeys)
            else:
                hkeys = pickle.loads(hkeys)
                hkeys.append(key)
                hkeys = pickle.dumps(hkeys)
                self._client.set(self._name, hkeys)
        else:
            values = pickle.loads(values)
            values.update(vals)

def _random_name(length):
    # For use with Redis, we return bytes
    return ''.join(random.choice(string.ascii_lowercase)
                   for _ in range(length)).encode('utf8')
