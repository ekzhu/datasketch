from collections import defaultdict
import redis
import os
import random
import string
from abc import ABCMeta, abstractmethod
ABC = ABCMeta('ABC', (object,), {}) # compatible with Python 2 *and* 3


def ordered_storage(config):
    '''Return ordered storage system based on the specified config'''
    tp = config['type']
    if tp == 'dict':
        return DictListStorage(config)
    if tp == 'redis':
        return RedisListStorage(config)


def unordered_storage(config):
    '''Return an unordered storage system based on the specified config'''
    tp = config['type']
    if tp == 'dict':
        return DictSetStorage(config)
    if tp == 'redis':
        return RedisSetStorage(config)


class Storage(ABC):
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
        return {k: len(v) for k, v in self._dict.items()}

    def has_key(self, key):
        return key in self._dict


class DictSetStorage(UnorderedStorage, DictListStorage):
    def __init__(self, config):
        self._dict = defaultdict(set)

    def get(self, key):
        return self._dict.get(key, set())

    def insert(self, key, *vals, **kwargs):
        self._dict[key].update(vals)


class RedisBuffer(redis.client.Pipeline):

    def __init__(self, connection_pool, response_callbacks, transaction,
                 shard_hint=None, buffer_size=50000):
        self.buffer_size = buffer_size
        super(RedisBuffer, self).__init__(
            connection_pool, response_callbacks, transaction,
            shard_hint=shard_hint)

    def execute_command(self, *args, **kwargs):
        if len(self.command_stack) >= self.buffer_size:
            self.execute()
        super(RedisBuffer, self).execute_command(*args, **kwargs)


class RedisStorage:

    def __init__(self, config, name=None):
        self.config = config
        redis_param = self._parse_config(self.config['redis'])
        self._redis = redis.Redis(**redis_param)
        self._buffer = RedisBuffer(self._redis.connection_pool,
                                   self._redis.response_callbacks,
                                   transaction=True)
        if name is None:
            name = _random_name(11)
        self._name = name

    def redis_key(self, key):
        return self._name + key

    def _parse_config(self, config):
        cfg = {}
        for key, value in config.items():
            if isinstance(value, dict):
                if 'env' in value:
                    value = os.getenv(value['env'], value.get('default', None))
            cfg[key] = value
        return cfg

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('_redis')
        state.pop('_buffer')
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self.__init__(self.config, name=self._name)


class RedisListStorage(OrderedStorage, RedisStorage):

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
            pipe.lrange(self.redis_key(key), 0, -1)
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
        self.__init__(self.config, name=self._name)


class RedisSetStorage(UnorderedStorage, RedisListStorage):

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
    return ''.join(random.choice(string.ascii_lowercase)
                   for _ in range(length)).encode('utf8')
