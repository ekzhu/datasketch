from collections import defaultdict
import os
import random
import string
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


def _random_name(length):
    # For use with Redis, we return bytes
    return ''.join(random.choice(string.ascii_lowercase)
                   for _ in range(length)).encode('utf8')
