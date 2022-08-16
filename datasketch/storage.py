from collections import defaultdict
import binascii
import collections
import itertools
import operator
import os
import random
import string
from abc import ABCMeta, abstractmethod
ABC = ABCMeta('ABC', (object,), {}) # compatible with Python 2 *and* 3
try:
    import redis
except ImportError:
    redis = None
try:
    import cassandra
    from cassandra import cluster as c_cluster
    from cassandra import concurrent as c_concurrent
    import logging
    logging.getLogger("cassandra").setLevel(logging.ERROR)
except ImportError:
    cassandra = None
    c_cluster = None
    c_concurrent = None


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
    if tp == 'cassandra':
        return CassandraListStorage(config, name=name)


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
    if tp == 'cassandra':
        return CassandraSetStorage(config, name=name)


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

    def add_to_select_buffer(self, keys):
        '''Query keys and add them to internal buffer'''
        if not hasattr(self, '_select_buffer'):
            self._select_buffer = self.getmany(*keys)
        else:
            self._select_buffer.extend(self.getmany(*keys))

    def collect_select_buffer(self):
        '''Return buffered query results'''
        if not hasattr(self, '_select_buffer'):
            return []
        buffer = list(self._select_buffer)
        del self._select_buffer[:]
        return buffer


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


if cassandra is not None:

    class CassandraSharedSession(object):
        """Cassandra session shared across all storage instances."""

        __session = None
        __session_buffer = None
        __session_select_buffer = None

        QUERY_CREATE_KEYSPACE = """
            CREATE KEYSPACE IF NOT EXISTS {keyspace}
            WITH replication = {replication}
        """

        QUERY_DROP_KEYSPACE = "DROP KEYSPACE IF EXISTS {}"

        @classmethod
        def get_session(cls, seeds, **kwargs):
            _ = kwargs
            keyspace = kwargs["keyspace"]
            replication = kwargs["replication"]

            if cls.__session is None:
                # Allow dependency injection
                session = kwargs.get("session")
                if session is None:
                    cluster = c_cluster.Cluster(seeds)
                    session = cluster.connect()
                cls.__session = session
            if cls.__session.keyspace != keyspace:
                if kwargs.get("drop_keyspace", False):
                    cls.__session.execute(cls.QUERY_DROP_KEYSPACE.format(keyspace))
                cls.__session.execute(cls.QUERY_CREATE_KEYSPACE.format(
                    keyspace=keyspace,
                    replication=str(replication),
                ))
                cls.__session.set_keyspace(keyspace)
            return cls.__session

        @classmethod
        def get_buffer(cls):
            if cls.__session_buffer is None:
                cls.__session_buffer = []
            return cls.__session_buffer

        @classmethod
        def get_select_buffer(cls):
            if cls.__session_select_buffer is None:
                cls.__session_select_buffer = []
            return cls.__session_select_buffer


    class CassandraClient(object):
        """Cassandra Client."""

        MIN_TOKEN = -(2 ** 63)

        PAGE_SIZE = 1024

        CONCURRENCY = 100

        QUERY_CREATE_TABLE = """
            CREATE TABLE IF NOT EXISTS {}  (
                key blob,
                value blob,
                ts bigint,
                PRIMARY KEY (key, value)
            ) WITH CLUSTERING ORDER BY (value DESC)
        """

        QUERY_DROP_TABLE = "DROP TABLE IF EXISTS {}"

        QUERY_GET_KEYS = """
            SELECT DISTINCT key, TOKEN(key) as f_token
            FROM {}
            WHERE TOKEN(key) >= ? LIMIT ?
        """

        QUERY_GET_COUNTS = """
            SELECT key, COUNT(value) AS count
            FROM {}
            WHERE key = ?
        """

        QUERY_SELECT = """
            SELECT key, value, ts
            FROM {}
            WHERE key = ?
        """

        QUERY_SELECT_ONE = """
            SELECT key, value
            FROM {}
            WHERE key = ? LIMIT 1
        """

        QUERY_DELETE_KEY = """
            DELETE FROM {}
            WHERE key = ?
        """

        QUERY_DELETE_VAL = """
            DELETE FROM {}
            WHERE key = ? AND value  = ?
        """

        QUERY_UPSERT = """
            UPDATE {}
            SET ts = ?
            WHERE key = ? AND value = ?
        """

        QUERY_INSERT = "INSERT INTO {} (key, value, ts) VALUES (?, ?, ?)"

        def __init__(self, cassandra_params, name, buffer_size):
            """
            Constructor.

            :param dict[str, any] cassandra_params: Cassandra parameters
            :param bytes name: the suffix to be used for the table name
            :param int buffer_size: the buffer size
            """
            self._buffer_size = buffer_size
            self._session = CassandraSharedSession.get_session(**cassandra_params)
            # This timestamp generator allows us to sort different values for the same key
            self._ts = c_cluster.MonotonicTimestampGenerator()
            # Each table (hashtable or key table is handled by a different storage; to increase
            # throughput it is possible to share a single buffer so the chances of a flush
            # are increased.
            if cassandra_params.get("shared_buffer", False):
                self._statements_and_parameters = CassandraSharedSession.get_buffer()
                self._select_statements_and_parameters_with_decoders = CassandraSharedSession.get_select_buffer()
            else:
                self._statements_and_parameters = []
                self._select_statements_and_parameters_with_decoders = []

            # Buckets tables rely on byte strings as keys and normal strings as values.
            # Keys tables have normal strings as keys and byte strings as values.
            # Since both data types can be reduced to byte strings without loss of data, we use
            # only one Cassandra table for both table types (so we can keep one single storage) and
            # we specify different encoders/decoders based on the table type.
            if b'bucket' in name:
                basename, _, ret = name.split(b'_', 2)
                name = basename + b'_bucket_' + binascii.hexlify(ret)
                self._key_decoder = lambda x: x
                self._key_encoder = lambda x: x
                self._val_decoder = lambda x: x.decode('utf-8')
                self._val_encoder = lambda x: x.encode('utf-8')
            else:
                self._key_decoder = lambda x: x.decode('utf-8')
                self._key_encoder = lambda x: x.encode('utf-8')
                self._val_decoder = lambda x: x
                self._val_encoder = lambda x: x
            table_name = 'lsh_' + name.decode('ascii')

            # Drop the table if are instructed to do so
            if cassandra_params.get("drop_tables", False):
                self._session.execute(self.QUERY_DROP_TABLE.format(table_name))
            self._session.execute(self.QUERY_CREATE_TABLE.format(table_name))

            # Prepare all the statements for this table
            self._stmt_insert = self._session.prepare(self.QUERY_INSERT.format(table_name))
            self._stmt_upsert = self._session.prepare(self.QUERY_UPSERT.format(table_name))
            self._stmt_get_keys = self._session.prepare(self.QUERY_GET_KEYS.format(table_name))
            self._stmt_get = self._session.prepare(self.QUERY_SELECT.format(table_name))
            self._stmt_get_one = self._session.prepare(self.QUERY_SELECT_ONE.format(table_name))
            self._stmt_get_count = self._session.prepare(self.QUERY_GET_COUNTS.format(table_name))
            self._stmt_delete_key = self._session.prepare(self.QUERY_DELETE_KEY.format(table_name))
            self._stmt_delete_val = self._session.prepare(self.QUERY_DELETE_VAL.format(table_name))

        @property
        def buffer_size(self):
            """
            Get the buffer size.

            :rtype: int
            :return: the buffer size
            """
            return self._buffer_size

        @buffer_size.setter
        def buffer_size(self, value):
            """
            Set the buffer size and propagate it to the underlying client.

            :param int value: buffer size
            """
            self._buffer_size = value

        @staticmethod
        def split_sequence(iterable, size):
            """
            Generator to split an iterable in chunks of given size.

            :param iterable iterable: the iterable to split
            :param int size: the size of a chunk
            :rtype: generator[iterable]
            :return: a generator
            """
            iterator = iter(iterable)
            item = list(itertools.islice(iterator, size))
            while item:
                yield item
                item = list(itertools.islice(iterator, size))

        def _select(self, statements_and_parameters):
            """
            Execute a list of statements and parameters returning data.

            :param iterable[tuple] statements_and_parameters: list of statements and parameters
            :rtype: list[Row]
            :return: the rows matching the queries
            """
            ret = []
            size = self.CONCURRENCY
            for sub_sequence in CassandraClient.split_sequence(statements_and_parameters, size):
                results = c_concurrent.execute_concurrent(
                    self._session,
                    sub_sequence,
                    concurrency=size,
                )
                for result in results:
                    success, rows = result
                    if success:
                        ret.append(rows)
                    else:
                        raise RuntimeError
            return ret

        def _execute(self, statements_and_parameters):
            """
            Execute a list of statements and parameters NOT returning data.

            :param iterable[tuple] statements_and_parameters: list of statements and parameters
            """
            size = self.CONCURRENCY
            for sub_sequence in CassandraClient.split_sequence(statements_and_parameters, size):
                c_concurrent.execute_concurrent(
                    self._session,
                    sub_sequence,
                    concurrency=size,
                )

        def _buffer(self, statements_and_parameters):
            """
            Buffer (and execute) statements and parameters NOT returning data.

            :param iterable[tuple] statements_and_parameters: list of statements and parameters
            """
            self._statements_and_parameters.extend(statements_and_parameters)
            if len(self._statements_and_parameters) >= self._buffer_size:
                self.empty_buffer()

        def empty_buffer(self):
            """
            Empty the buffer of statements and parameters.
            """
            # copy the underlying list in a python2/3 compatible way
            buffer = list(self._statements_and_parameters)
            # delete the actual elements in a python2/3 compatible way
            del self._statements_and_parameters[:]
            self._execute(buffer)

        def insert(self, key, vals, buffer=False):
            """
            Insert an iterable of values with the same key.

            :param byte|str key: the key
            :param iterable[byte|str] vals: the iterable of values
            :param boolean buffer: whether the insert statements should be buffered
            """
            statements_and_parameters = [
                (self._stmt_insert, (self._key_encoder(key), self._val_encoder(val), self._ts()))
                for val in vals
            ]
            if buffer:
                self._buffer(statements_and_parameters)
            else:
                self._execute(statements_and_parameters)

        def upsert(self, key, vals, buffer=False):
            """
            Upsert an iterable of values with the same key.

            Note: this is used when treating a Cassandra partition as a set. Since we upsert data
                we never store duplicates. In this case the timestamp loses its meaning as we
                are not interested in sorting records anymore (it is a set after all) and we can
                safely overwrite every time we are storing a duplicate.

            :param byte|str key: the key
            :param iterable[byte|str] vals: the iterable of values
            :param boolean buffer: whether the upsert statements should be buffered
            """
            statements_and_parameters = [
                (self._stmt_upsert, (self._ts(), self._key_encoder(key), self._val_encoder(val)))
                for val in vals
            ]
            if buffer:
                self._buffer(statements_and_parameters)
            else:
                self._execute(statements_and_parameters)

        def delete_keys(self, keys, buffer=False):
            """
            Delete a key (and all its values).

            :param iterable[byte|str] keys: the key
            :param boolean buffer: whether the delete statements should be buffered
            """
            statements_and_parameters = [
                (self._stmt_delete_key, (self._key_encoder(key), ))
                for key in keys
            ]
            if buffer:
                self._buffer(statements_and_parameters)
            else:
                self._execute(statements_and_parameters)

        def delete(self, key, val, buffer=False):
            """
            Delete a value from a key.

            :param byte|str key: the key
            :param byte|str val: the value
            :param boolean buffer: whether the delete statement should be buffered
            """
            statements_and_parameters = [
                (self._stmt_delete_val, (self._key_encoder(key), self._val_encoder(val)))
            ]
            if buffer:
                self._buffer(statements_and_parameters)
            else:
                self._execute(statements_and_parameters)

        def get_keys(self):
            """
            Get all the keys.

            Note: selecting all keys in Cassandra via "SELECT DISTINCT key FROM table" is bound to
                time out since all nodes need to be contacted. To avoid this, we paginate through
                all keys using the TOKEN function. In this way we issue several different queries
                which alone can not time out.

            :rtype: set[byte|str]
            :return: the set of all keys
            """
            min_token = self.MIN_TOKEN
            keys = set([])
            while True:
                rows = self._session.execute(self._stmt_get_keys, (min_token, self.PAGE_SIZE))
                if not rows:
                    break
                for r in rows:
                    keys.add(self._key_decoder(r.key))
                    min_token = r.f_token + 1
            return keys

        def add_to_select_buffer(self, keys):
            """
            Buffer query statements and parameters with decoders to be used on returned data.

            :param iterable[byte|str] keys: the keys
            """
            statements_and_parameters_with_decoders = [
                ((self._stmt_get, (self._key_encoder(key),)), (self._key_decoder, self._val_decoder))
                for key in keys
            ]
            self._select_statements_and_parameters_with_decoders.extend(statements_and_parameters_with_decoders)

        def collect_select_buffer(self):
            """
            Perform buffered select queries

            :return: list of list of query results
            """
            if not self._select_statements_and_parameters_with_decoders:
                return []
            # copy the underlying list in a python2/3 compatible way
            buffer = list(self._select_statements_and_parameters_with_decoders)
            # delete the actual elements in a python2/3 compatible way
            del self._select_statements_and_parameters_with_decoders[:]
            statements_and_parameters, decoders = zip(*buffer)

            ret = collections.defaultdict(list)
            query_results = self._select(statements_and_parameters)
            for rows, (key_decoder, val_decoder) in zip(query_results, decoders):
                for row in rows:
                    ret[key_decoder(row.key)].append((val_decoder(row.value), row.ts))
            return [
                [x[0] for x in sorted(v, key=operator.itemgetter(1))] for v in ret.values()
            ]

        def select(self, keys):
            """
            Select all values for the given keys.

            :param iterable[byte|str] keys: the keys
            :rtype: dict[byte|str,list[byte|str]
            :return: a dictionary of lists
            """
            statements_and_parameters = [
                (self._stmt_get, (self._key_encoder(key), ))
                for key in keys
            ]
            ret = collections.defaultdict(list)
            for rows in self._select(statements_and_parameters):
                for row in rows:
                    ret[self._key_decoder(row.key)].append((self._val_decoder(row.value), row.ts))
            return {
                k: [x[0] for x in sorted(v, key=operator.itemgetter(1))]
                for k, v in ret.items()
            }

        def select_count(self, keys):
            """
            Count the values for each of the provided keys.

            :param iterable[byte|str] keys: list of keys
            :rtype: dict[byte|str,int]
            :return: the number of values per key
            """
            statements_and_parameters = [
                (self._stmt_get_count, (self._key_encoder(key), ))
                for key in keys
            ]
            return {
                self._key_decoder(row.key): row.count
                for rows in self._select(statements_and_parameters)
                for row in rows
            }

        def one(self, key):
            """
            Select one single value of the given key.

            :param byte|str key: the key
            :rtype: byte|str|None
            :return: a single value for that key or None if the key does not exist
            """
            rows = self._session.execute(self._stmt_get_one, (self._key_encoder(key),))
            if rows:
                row = next(iter(rows))
                return self._val_decoder(row.value)
            return None


    class CassandraStorage(object):
        """
        Storage implementation using Cassandra.

        Note: like other implementations, each storage has its own client. Unlike other
            implementations, all storage instances share one session and can potentially share the
            same buffer.
        """

        DEFAULT_BUFFER_SIZE = 5000

        def __init__(self, config, name=None, buffer_size=None):
            """
            Constructor.

            :param dict[str, any] config: configuration following the following format:
                {
                    'basename': b'test',
                    'type': 'cassandra',
                    'cassandra': {
                        'seeds': ['127.0.0.1'],
                        'keyspace': 'lsh_test',
                        'replication': {
                            'class': 'SimpleStrategy',
                            'replication_factor': '1'
                        },
                        'drop_keyspace': True,
                        'drop_tables': True,
                        'shared_buffer': False,
                    }
                }
            :param bytes name: the name
            :param int buffer_size: the buffer size
            """
            self._config = config
            if buffer_size is None:
                buffer_size = CassandraStorage.DEFAULT_BUFFER_SIZE
            cassandra_param = self._parse_config(self._config['cassandra'])
            self._name = name if name else _random_name(11).decode('utf-8')
            self._buffer_size = buffer_size
            self._client = CassandraClient(cassandra_param, name, self._buffer_size)

        @staticmethod
        def _parse_config(config):
            """
            Parse a configuration dictionary, optionally fetching data from env variables.

            :param dict[str, any] config: the configuration
            :rtype: dict[str, str]
            :return: the parse configuration
            """
            cfg = {}
            for key, value in config.items():
                if isinstance(value, dict):
                    if 'env' in value:
                        value = os.getenv(value['env'], value.get('default', None))
                cfg[key] = value
            return cfg

        @property
        def buffer_size(self):
            """
            Get the buffer size.

            :rtype: int
            :return: the buffer size
            """
            return self._buffer_size

        @buffer_size.setter
        def buffer_size(self, value):
            """
            Set the buffer size and propagate it to the underlying client.

            :param int value: buffer size
            """
            self._buffer_size = value
            self._client.buffer_size = value

        def __getstate__(self):
            """
            Get a pickable state by removing unpickable objects.

            :rtype: dict[str, any]
            :return: the state
            """
            state = self.__dict__.copy()
            state.pop('_client')
            return state

        def __setstate__(self, state):
            """
            Set the state by reconnecting ephemeral objects.

            :param dict[str, any] state: the state to restore
            """
            self.__dict__ = state
            self.__init__(self._config, name=self._name, buffer_size=self._buffer_size)


    class CassandraListStorage(OrderedStorage, CassandraStorage):
        """
        OrderedStorage storage implementation using Cassandra as backend.

        Note: Since we need to (i) select and delete values by both 'key' and by 'key and value',
            and (ii) allow duplicate values, we store a monotonically increasing timestamp as
            additional value.
        """

        def keys(self):
            """Implement interface."""
            return self._client.get_keys()

        def get(self, key):
            """Implement interface."""
            return self._client.select([key]).get(key, [])

        def getmany(self, *keys):
            """Implement interface."""
            return self._client.select(keys).values()

        def add_to_select_buffer(self, keys):
            """Implement interface."""
            self._client.add_to_select_buffer(keys)

        def collect_select_buffer(self):
            """Implement interface."""
            return self._client.collect_select_buffer()

        def insert(self, key, *vals, **kwargs):
            """Implement interface."""
            buffer = kwargs.pop('buffer', False)
            self._client.insert(key, vals, buffer)

        def remove(self, *keys, **kwargs):
            """Implement interface."""
            buffer = kwargs.pop('buffer', False)
            self._client.delete_keys(keys, buffer)

        def remove_val(self, key, val, **kwargs):
            """Implement interface."""
            buffer = kwargs.pop('buffer', False)
            self._client.delete(key, val, buffer)

        def size(self):
            """Implement interface."""
            return len(self.keys())

        def itemcounts(self):
            """Implement interface."""
            return self._client.select_count(self.keys())

        def has_key(self, key):
            """Implement interface."""
            return self._client.one(key) is not None

        def empty_buffer(self):
            """Implement interface."""
            self._client.empty_buffer()


    class CassandraSetStorage(UnorderedStorage, CassandraListStorage):
        """
        OrderedStorage storage implementation using Cassandra as backend.

        Note: since we are interested in keeping duplicates or ordered data, we upsert the data
            ignoring what the timestamp actually means.
        """

        def get(self, key):
            """Implement interface and override super-class."""
            return set(super(CassandraSetStorage, self).get(key))

        def insert(self, key, *vals, **kwargs):
            """Implement interface and override super-class."""
            buffer = kwargs.pop('buffer', False)
            self._client.upsert(key, vals, buffer)


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
                        'redis': {'host': 'localhost', 'port': 6379},
                        'redis_buffer': {'transaction': True}
                    }

                one can refer to system environment variables via::

                    storage_config={
                        'type': 'redis',
                        'redis': {
                            'host': {'env': 'REDIS_HOSTNAME',
                                     'default':'localhost'},
                            'port': 6379}
                        },
                        'redis_buffer': {'transaction': True}
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
            redis_buffer_param = self._parse_config(self.config.get('redis_buffer', {}))
            self._buffer = RedisBuffer(self._redis.connection_pool,
                                       self._redis.response_callbacks,
                                       transaction=redis_buffer_param.get('transaction', True),
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
