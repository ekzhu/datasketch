import sys
import asyncio
import pickle
import unittest
import random
import string
from itertools import chain, islice

import aioredis
import motor.motor_asyncio
import aiounittest
import numpy as np

from datasketch.experimental.async import AsyncMinHashLSH
from datasketch.minhash import MinHash
from datasketch.weighted_minhash import WeightedMinHashGenerator

STORAGE_CONFIG_REDIS = {'type': 'aioredis', 'redis': {'host': 'localhost', 'port': 6379}}
STORAGE_CONFIG_MONGO = {'type': 'aiomongo', 'mongo': {'host': 'localhost', 'port': 27017, 'db': 'lsh_test'}}
DO_TEST_REDIS = False
DO_TEST_MONGO = False


@unittest.skipIf(sys.version_info < (3, 6), "Skipping TestAsyncMinHashLSH. Supported Python version >= 3.6")
class TestAsyncMinHashLSH(aiounittest.AsyncTestCase):
    """
        For tests Redis should be installed on local machine or set right host.

        For tests Mongo should be installed on local machine or set right host.
    """

    def setUp(self):
        self._storage_config_redis = STORAGE_CONFIG_REDIS
        self._storage_config_mongo = STORAGE_CONFIG_MONGO

    async def tearDownAsync(self):
        if DO_TEST_REDIS:
            dsn = 'redis://{host}:{port}'.format(**self._storage_config_redis['redis'])
            redis = await aioredis.create_redis(dsn, loop=self.get_event_loop())
            await redis.flushdb()
            redis.close()
            await redis.wait_closed()

        if DO_TEST_MONGO:
            dsn = 'mongodb://{host}:{port}'.format(**self._storage_config_mongo['mongo'])
            motor.motor_asyncio.AsyncIOMotorClient(dsn).drop_database(self._storage_config_mongo['mongo']['db'])

    @unittest.skipIf(not DO_TEST_REDIS, "Skipping test_init_redis")
    async def test_init_redis(self):
        async with AsyncMinHashLSH(storage_config=self._storage_config_redis,
                                   threshold=0.8) as lsh:
            self.assertTrue(await lsh.is_empty())
            b1, r1 = lsh.b, lsh.r

        async with AsyncMinHashLSH(storage_config=self._storage_config_redis,
                                   threshold=0.8, weights=(0.2, 0.8)) as lsh:
            b2, r2 = lsh.b, lsh.r
        self.assertTrue(b1 < b2)
        self.assertTrue(r1 > r2)

    @unittest.skipIf(not DO_TEST_REDIS, "Skipping test__H_redis")
    async def test__H_redis(self):
        """
        Check _H output consistent bytes length given
        the same concatenated hash value size
        """
        for l in range(2, 128 + 1, 16):
            m = MinHash()
            m.update("abcdefg".encode("utf8"))
            m.update("1234567".encode("utf8"))
            async with AsyncMinHashLSH(storage_config=self._storage_config_redis,
                                       num_perm=128) as lsh:
                await lsh.insert("m", m)
                sizes = [len(H) for ht in lsh.hashtables for H in
                         await ht.keys()]
                self.assertTrue(all(sizes[0] == s for s in sizes))

    @unittest.skipIf(not DO_TEST_REDIS, "Skipping test_insert_redis")
    async def test_insert_redis(self):
        async with AsyncMinHashLSH(storage_config=self._storage_config_redis,
                                   threshold=0.5, num_perm=16) as lsh:
            m1 = MinHash(16)
            m1.update("a".encode("utf8"))
            m2 = MinHash(16)
            m2.update("b".encode("utf8"))
            await lsh.insert("a", m1)
            await lsh.insert("b", m2)
            for t in lsh.hashtables:
                self.assertTrue(await t.size() >= 1)
                items = []
                for H in await t.keys():
                    items.extend(await t.get(H))
                self.assertTrue(pickle.dumps("a") in items)
                self.assertTrue(pickle.dumps("b") in items)
            self.assertTrue(await lsh.has_key("a"))
            self.assertTrue(await lsh.has_key("b"))
            for i, H in enumerate(await lsh.keys.get(pickle.dumps("a"))):
                res = await lsh.hashtables[i].get(H)
                self.assertTrue(pickle.dumps("a") in res)

            m3 = MinHash(18)
            with self.assertRaises(ValueError):
                await lsh.insert("c", m3)

    @unittest.skipIf(not DO_TEST_REDIS, "Skipping test_query_redis")
    async def test_query_redis(self):
        async with AsyncMinHashLSH(storage_config=self._storage_config_redis,
                                   threshold=0.5, num_perm=16) as lsh:
            m1 = MinHash(16)
            m1.update("a".encode("utf8"))
            m2 = MinHash(16)
            m2.update("b".encode("utf8"))
            await lsh.insert("a", m1)
            await lsh.insert("b", m2)
            result = await lsh.query(m1)
            self.assertTrue("a" in result)
            result = await lsh.query(m2)
            self.assertTrue("b" in result)

            m3 = MinHash(18)
            with self.assertRaises(ValueError):
                await lsh.query(m3)

    @unittest.skipIf(not DO_TEST_REDIS, "Skipping test_remove_redis")
    async def test_remove_redis(self):
        async with AsyncMinHashLSH(storage_config=self._storage_config_redis,
                                   threshold=0.5, num_perm=16) as lsh:
            m1 = MinHash(16)
            m1.update("a".encode("utf8"))
            m2 = MinHash(16)
            m2.update("b".encode("utf8"))
            await lsh.insert("a", m1)
            await lsh.insert("b", m2)

            await lsh.remove("a")
            self.assertTrue(not await lsh.has_key("a"))
            for table in lsh.hashtables:
                for H in await table.keys():
                    self.assertGreater(len(await table.get(H)), 0)
                    self.assertTrue("a" not in await table.get(H))

            with self.assertRaises(ValueError):
                await lsh.remove("c")

    @unittest.skipIf(not DO_TEST_REDIS, "Skipping test_pickle_redis")
    async def test_pickle_redis(self):
        async with AsyncMinHashLSH(storage_config=self._storage_config_redis,
                                   threshold=0.5, num_perm=16) as lsh:
            m1 = MinHash(16)
            m1.update("a".encode("utf8"))
            m2 = MinHash(16)
            m2.update("b".encode("utf8"))
            await lsh.insert("a", m1)
            await lsh.insert("b", m2)
            pickled = pickle.dumps(lsh)

        async with pickle.loads(pickled) as lsh2:
            result = await lsh2.query(m1)
            self.assertTrue("a" in result)
            result = await lsh2.query(m2)
            self.assertTrue("b" in result)

    @unittest.skipIf(not DO_TEST_REDIS, "Skipping test_insertion_session_redis")
    async def test_insertion_session_redis(self):
        async with AsyncMinHashLSH(storage_config=self._storage_config_redis,
                                   threshold=0.5, num_perm=16) as lsh:
            m1 = MinHash(16)
            m1.update("a".encode("utf8"))
            m2 = MinHash(16)
            m2.update("b".encode("utf8"))
            data = [("a", m1), ("b", m2)]
            async with lsh.insertion_session() as session:
                for key, minhash in data:
                    await session.insert(key, minhash)
            for t in lsh.hashtables:
                self.assertTrue(await t.size() >= 1)
                items = []
                for H in await t.keys():
                    items.extend(await t.get(H))
                self.assertTrue(pickle.dumps("a") in items)
                self.assertTrue(pickle.dumps("b") in items)
            self.assertTrue(await lsh.has_key("a"))
            self.assertTrue(await lsh.has_key("b"))
            for i, H in enumerate(await lsh.keys.get(pickle.dumps("a"))):
                res = await lsh.hashtables[i].get(H)
                self.assertTrue(pickle.dumps("a") in res)

    @unittest.skipIf(not DO_TEST_REDIS, "Skipping test_get_counts_redis")
    async def test_get_counts_redis(self):
        async with AsyncMinHashLSH(storage_config=self._storage_config_redis,
                                   threshold=0.5, num_perm=16) as lsh:
            m1 = MinHash(16)
            m1.update("a".encode("utf8"))
            m2 = MinHash(16)
            m2.update("b".encode("utf8"))
            await lsh.insert("a", m1)
            await lsh.insert("b", m2)
            counts = await lsh.get_counts()
            self.assertEqual(len(counts), lsh.b)
            for table in counts:
                self.assertEqual(sum(table.values()), 2)

    @unittest.skipIf(not DO_TEST_MONGO, "Skipping test_init_mongo")
    async def test_init_mongo(self):
        async with AsyncMinHashLSH(storage_config=self._storage_config_mongo,
                                   threshold=0.8) as lsh:
            self.assertTrue(await lsh.is_empty())
            b1, r1 = lsh.b, lsh.r

        async with AsyncMinHashLSH(storage_config=self._storage_config_mongo,
                                   threshold=0.8,
                                   weights=(0.2, 0.8)) as lsh:
            b2, r2 = lsh.b, lsh.r
        self.assertTrue(b1 < b2)
        self.assertTrue(r1 > r2)

    @unittest.skipIf(not DO_TEST_MONGO, "Skipping test__H_mongo")
    async def test__H_mongo(self):
        """
        Check _H output consistent bytes length given
        the same concatenated hash value size
        """
        for l in range(2, 128 + 1, 16):
            m = MinHash()
            m.update("abcdefg".encode("utf8"))
            m.update("1234567".encode("utf8"))
            async with AsyncMinHashLSH(storage_config=self._storage_config_mongo,
                                       num_perm=128) as lsh:
                await lsh.insert("m", m)
                sizes = [len(H) for ht in lsh.hashtables for H in await ht.keys()]
                self.assertTrue(all(sizes[0] == s for s in sizes))

    @unittest.skipIf(not DO_TEST_MONGO, "Skipping test_insert_mongo")
    async def test_insert_mongo(self):
        async with AsyncMinHashLSH(storage_config=self._storage_config_mongo,
                                   threshold=0.5, num_perm=16) as lsh:
            seq = ['aahhb', 'aahh', 'aahhc', 'aac', 'kld', 'bhg', 'kkd', 'yow', 'ppi', 'eer']
            objs = [MinHash(16) for _ in range(len(seq))]
            for e, obj in zip(seq, objs):
                for i in e:
                    obj.update(i.encode('utf-8'))

            data = [(e, m) for e, m in zip(seq, objs)]
            for key, minhash in data:
                await lsh.insert(key, minhash)
            for t in lsh.hashtables:
                self.assertTrue(await t.size() >= 1)
                items = []
                for H in await t.keys():
                    items.extend(await t.get(H))
                self.assertTrue('aahh' in items)
                self.assertTrue('bhg' in items)
            self.assertTrue(await lsh.has_key('aahh'))
            self.assertTrue(await lsh.has_key('bhg'))
            for i, H in enumerate(await lsh.keys.get('aahhb')):
                self.assertTrue('aahhb' in await lsh.hashtables[i].get(H))

            m3 = MinHash(18)
            with self.assertRaises(ValueError):
                await lsh.insert("c", m3)

    @unittest.skipIf(not DO_TEST_MONGO, "Skipping test_query_mongo")
    async def test_query_mongo(self):
        async with AsyncMinHashLSH(storage_config=self._storage_config_mongo,
                                   threshold=0.5, num_perm=16) as lsh:
            m1 = MinHash(16)
            m1.update("a".encode("utf8"))
            m2 = MinHash(16)
            m2.update("b".encode("utf8"))
            await lsh.insert("a", m1)
            await lsh.insert("b", m2)
            result = await lsh.query(m1)
            self.assertTrue("a" in result)
            result = await lsh.query(m2)
            self.assertTrue("b" in result)

            m3 = MinHash(18)
            with self.assertRaises(ValueError):
                await lsh.query(m3)

    @unittest.skipIf(not DO_TEST_MONGO, "Skipping test_remove_mongo")
    async def test_remove_mongo(self):
        async with AsyncMinHashLSH(storage_config=self._storage_config_mongo,
                                   threshold=0.5, num_perm=16) as lsh:
            m1 = MinHash(16)
            m1.update("a".encode("utf8"))
            m2 = MinHash(16)
            m2.update("b".encode("utf8"))
            await lsh.insert("a", m1)
            await lsh.insert("b", m2)

            await lsh.remove("a")
            self.assertTrue(not await lsh.has_key("a"))
            for table in lsh.hashtables:
                for H in await table.keys():
                    self.assertGreater(len(await table.get(H)), 0)
                    self.assertTrue("a" not in await table.get(H))

            with self.assertRaises(ValueError):
                await lsh.remove("c")

    @unittest.skipIf(not DO_TEST_MONGO, "Skipping test_pickle_mongo")
    async def test_pickle_mongo(self):
        async with AsyncMinHashLSH(storage_config=self._storage_config_mongo, threshold=0.5, num_perm=16) as lsh:
            m1 = MinHash(16)
            m1.update("a".encode("utf8"))
            m2 = MinHash(16)
            m2.update("b".encode("utf8"))
            await lsh.insert("a", m1)
            await lsh.insert("b", m2)
            pickled = pickle.dumps(lsh)

        async with pickle.loads(pickled) as lsh2:
            result = await lsh2.query(m1)
            self.assertTrue("a" in result)
            result = await lsh2.query(m2)
            self.assertTrue("b" in result)
            await lsh2.close()

    @unittest.skipIf(not DO_TEST_MONGO, "Skipping test_insertion_session_mongo")
    async def test_insertion_session_mongo(self):
        def chunk(it, size):
            it = iter(it)
            return iter(lambda: tuple(islice(it, size)), ())

        _chunked_str = chunk((random.choice(string.ascii_lowercase) for _ in range(10000)), 4)
        seq = frozenset(chain((''.join(s) for s in _chunked_str),
                          ('aahhb', 'aahh', 'aahhc', 'aac', 'kld', 'bhg', 'kkd', 'yow', 'ppi', 'eer')))
        objs = [MinHash(16) for _ in range(len(seq))]
        for e, obj in zip(seq, objs):
            for i in e:
                obj.update(i.encode('utf-8'))

        data = [(e, m) for e, m in zip(seq, objs)]

        async with AsyncMinHashLSH(storage_config=self._storage_config_mongo,
                                   threshold=0.5, num_perm=16, batch_size=1000) as lsh:
            async with lsh.insertion_session() as session:
                fs = (session.insert(key, minhash, check_duplication=False) for key, minhash in data)
                await asyncio.gather(*fs)

            for t in lsh.hashtables:
                self.assertTrue(await t.size() >= 1)
                items = []
                for H in await t.keys():
                    items.extend(await t.get(H))
                self.assertTrue('aahhb' in items)
                self.assertTrue('kld' in items)
            self.assertTrue(await lsh.has_key('aahhb'))
            self.assertTrue(await lsh.has_key('kld'))
            for i, H in enumerate(await lsh.keys.get('aahh')):
                self.assertTrue('aahh' in await lsh.hashtables[i].get(H))

    @unittest.skipIf(not DO_TEST_MONGO, "Skipping test_get_counts_mongo")
    async def test_get_counts_mongo(self):
        async with AsyncMinHashLSH(storage_config=self._storage_config_mongo,
                                   threshold=0.5, num_perm=16) as lsh:
            m1 = MinHash(16)
            m1.update("a".encode("utf8"))
            m2 = MinHash(16)
            m2.update("b".encode("utf8"))
            await lsh.insert("a", m1)
            await lsh.insert("b", m2)
            counts = await lsh.get_counts()
            self.assertEqual(len(counts), lsh.b)
            for table in counts:
                self.assertEqual(sum(table.values()), 2)


@unittest.skipIf(sys.version_info < (3, 6), "Skipping TestAsyncMinHashLSH. Supported Python version >= 3.6")
class TestWeightedMinHashLSH(aiounittest.AsyncTestCase):
    """For tests Redis should be installed on local machine.
    Don't forget to clean Redis DB=0."""

    def setUp(self):
        super().setUp()
        self._storage_config_redis = STORAGE_CONFIG_REDIS
        self._storage_config_mongo = STORAGE_CONFIG_MONGO

    async def tearDownAsync(self):
        if DO_TEST_REDIS:
            dsn = 'redis://{host}:{port}'.format(**self._storage_config_redis['redis'])
            redis = await aioredis.create_redis(dsn, loop=self.get_event_loop())
            await redis.flushall()

        if DO_TEST_MONGO:
            dsn = 'mongodb://{host}:{port}'.format(**self._storage_config_mongo['mongo'])
            motor.motor_asyncio.AsyncIOMotorClient(dsn).drop_database(self._storage_config_mongo['mongo']['db'])

    @unittest.skipIf(not DO_TEST_REDIS, "Skipping test_init_redis")
    async def test_init_redis(self):
        async with AsyncMinHashLSH(storage_config=self._storage_config_redis,
                                   threshold=0.8) as lsh:
            self.assertTrue(await lsh.is_empty())
            b1, r1 = lsh.b, lsh.r
        async with AsyncMinHashLSH(storage_config=self._storage_config_redis,
                                   threshold=0.8, weights=(0.2, 0.8)) as lsh:
            b2, r2 = lsh.b, lsh.r
        self.assertTrue(b1 < b2)
        self.assertTrue(r1 > r2)

    @unittest.skipIf(not DO_TEST_REDIS, "Skipping test__H_redis")
    async def test__H_redis(self):
        """
        Check _H output consistent bytes length given
        the same concatenated hash value size
        """
        mg = WeightedMinHashGenerator(100, sample_size=128)
        for l in range(2, mg.sample_size + 1, 16):
            m = mg.minhash(np.random.randint(1, 99999999, 100))
            async with AsyncMinHashLSH(storage_config=self._storage_config_redis,
                                       num_perm=128) as lsh:
                await lsh.insert("m", m)
                fs = (ht.keys() for ht in lsh.hashtables)
                hashtables = await asyncio.gather(*fs)
                sizes = [len(H) for H in hashtables]
                self.assertTrue(all(sizes[0] == s for s in sizes))

    @unittest.skipIf(not DO_TEST_REDIS, "Skipping test_insert_redis")
    async def test_insert_redis(self):
        async with AsyncMinHashLSH(storage_config=self._storage_config_redis,
                                   threshold=0.5, num_perm=4) as lsh:
            mg = WeightedMinHashGenerator(10, 4)
            m1 = mg.minhash(np.random.uniform(1, 10, 10))
            m2 = mg.minhash(np.random.uniform(1, 10, 10))
            await lsh.insert("a", m1)
            await lsh.insert("b", m2)
            for t in lsh.hashtables:
                self.assertTrue(await t.size() >= 1)
                items = []
                for H in await t.keys():
                    items.extend(await t.get(H))
                self.assertTrue(pickle.dumps("a") in items)
                self.assertTrue(pickle.dumps("b") in items)
            self.assertTrue(await lsh.has_key("a"))
            self.assertTrue(await lsh.has_key("b"))
            for i, H in enumerate(await lsh.keys.get(pickle.dumps("a"))):
                self.assertTrue(pickle.dumps("a") in await lsh.hashtables[i].get(H))

            mg = WeightedMinHashGenerator(10, 5)
            m3 = mg.minhash(np.random.uniform(1, 10, 10))
            with self.assertRaises(ValueError):
                await lsh.insert("c", m3)

    @unittest.skipIf(not DO_TEST_REDIS, "Skipping test_query_redis")
    async def test_query_redis(self):
        async with AsyncMinHashLSH(storage_config=self._storage_config_redis,
                                   threshold=0.5, num_perm=4) as lsh:
            mg = WeightedMinHashGenerator(10, 4)
            m1 = mg.minhash(np.random.uniform(1, 10, 10))
            m2 = mg.minhash(np.random.uniform(1, 10, 10))
            await lsh.insert("a", m1)
            await lsh.insert("b", m2)
            result = await lsh.query(m1)
            self.assertTrue("a" in result)
            result = await lsh.query(m2)
            self.assertTrue("b" in result)

            mg = WeightedMinHashGenerator(10, 5)
            m3 = mg.minhash(np.random.uniform(1, 10, 10))

            with self.assertRaises(ValueError):
                await lsh.query(m3)

    @unittest.skipIf(not DO_TEST_REDIS, "Skipping test_remove_redis")
    async def test_remove_redis(self):
        async with AsyncMinHashLSH(storage_config=self._storage_config_redis,
                                   threshold=0.5, num_perm=4) as lsh:
            mg = WeightedMinHashGenerator(10, 4)
            m1 = mg.minhash(np.random.uniform(1, 10, 10))
            m2 = mg.minhash(np.random.uniform(1, 10, 10))
            await lsh.insert("a", m1)
            await lsh.insert("b", m2)

            await lsh.remove("a")
            self.assertTrue(not await lsh.has_key("a"))
            for table in lsh.hashtables:
                for H in await table.keys():
                    self.assertGreater(len(await table.get(H)), 0)
                    self.assertTrue("a" not in await table.get(H))

            with self.assertRaises(ValueError):
                await lsh.remove("c")

    @unittest.skipIf(not DO_TEST_REDIS, "Skipping test_pickle_redis")
    async def test_pickle_redis(self):
        async with AsyncMinHashLSH(storage_config=self._storage_config_redis,
                                   threshold=0.5, num_perm=4) as lsh:
            mg = WeightedMinHashGenerator(10, 4)
            m1 = mg.minhash(np.random.uniform(1, 10, 10))
            m2 = mg.minhash(np.random.uniform(1, 10, 10))
            await lsh.insert("a", m1)
            await lsh.insert("b", m2)
            pickled = pickle.dumps(lsh)

        async with pickle.loads(pickled) as lsh2:
            result = await lsh2.query(m1)
            self.assertTrue("a" in result)
            result = await lsh2.query(m2)
            self.assertTrue("b" in result)

    @unittest.skipIf(not DO_TEST_MONGO, "Skipping test_init_mongo")
    async def test_init_mongo(self):
        async with AsyncMinHashLSH(storage_config=self._storage_config_mongo,
                                   threshold=0.8) as lsh:
            self.assertTrue(await lsh.is_empty())
            b1, r1 = lsh.b, lsh.r
        async with AsyncMinHashLSH(storage_config=self._storage_config_mongo,
                                   threshold=0.8, weights=(0.2, 0.8)) as lsh:
            b2, r2 = lsh.b, lsh.r
        self.assertTrue(b1 < b2)
        self.assertTrue(r1 > r2)

    @unittest.skipIf(not DO_TEST_MONGO, "Skipping test__H_mongo")
    async def test__H_mongo(self):
        """
        Check _H output consistent bytes length given
        the same concatenated hash value size
        """
        mg = WeightedMinHashGenerator(100, sample_size=128)
        for l in range(2, mg.sample_size + 1, 16):
            m = mg.minhash(np.random.randint(1, 99999999, 100))
            async with AsyncMinHashLSH(storage_config=self._storage_config_mongo,
                                       num_perm=128) as lsh:
                await lsh.insert("m", m)
                fs = (ht.keys() for ht in lsh.hashtables)
                hashtables = await asyncio.gather(*fs)
                sizes = [len(H) for H in hashtables]
                self.assertTrue(all(sizes[0] == s for s in sizes))

    @unittest.skipIf(not DO_TEST_MONGO, "Skipping test_insert_mongo")
    async def test_insert_mongo(self):
        async with AsyncMinHashLSH(storage_config=self._storage_config_mongo,
                                   threshold=0.5, num_perm=4) as lsh:
            mg = WeightedMinHashGenerator(10, 4)
            m1 = mg.minhash(np.random.uniform(1, 10, 10))
            m2 = mg.minhash(np.random.uniform(1, 10, 10))
            await lsh.insert("a", m1)
            await lsh.insert("b", m2)
            for t in lsh.hashtables:
                self.assertTrue(await t.size() >= 1)
                items = []
                for H in await t.keys():
                    items.extend(await t.get(H))
                self.assertTrue("a" in items)
                self.assertTrue("b" in items)
            self.assertTrue(await lsh.has_key("a"))
            self.assertTrue(await lsh.has_key("b"))
            for i, H in enumerate(await lsh.keys.get("a")):
                self.assertTrue("a" in await lsh.hashtables[i].get(H))

            mg = WeightedMinHashGenerator(10, 5)
            m3 = mg.minhash(np.random.uniform(1, 10, 10))
            with self.assertRaises(ValueError):
                await lsh.insert("c", m3)

    @unittest.skipIf(not DO_TEST_MONGO, "Skipping test_query_mongo")
    async def test_query_mongo(self):
        async with AsyncMinHashLSH(storage_config=self._storage_config_mongo,
                                   threshold=0.5, num_perm=4) as lsh:
            mg = WeightedMinHashGenerator(10, 4)
            m1 = mg.minhash(np.random.uniform(1, 10, 10))
            m2 = mg.minhash(np.random.uniform(1, 10, 10))
            await lsh.insert("a", m1)
            await lsh.insert("b", m2)
            result = await lsh.query(m1)
            self.assertTrue("a" in result)
            result = await lsh.query(m2)
            self.assertTrue("b" in result)

            mg = WeightedMinHashGenerator(10, 5)
            m3 = mg.minhash(np.random.uniform(1, 10, 10))

            with self.assertRaises(ValueError):
                await lsh.query(m3)

    @unittest.skipIf(not DO_TEST_MONGO, "Skipping test_remove_mongo")
    async def test_remove_mongo(self):
        async with AsyncMinHashLSH(storage_config=self._storage_config_mongo,
                                   threshold=0.5, num_perm=4) as lsh:
            mg = WeightedMinHashGenerator(10, 4)
            m1 = mg.minhash(np.random.uniform(1, 10, 10))
            m2 = mg.minhash(np.random.uniform(1, 10, 10))
            await lsh.insert("a", m1)
            await lsh.insert("b", m2)

            await lsh.remove("a")
            self.assertTrue(not await lsh.has_key("a"))
            for table in lsh.hashtables:
                for H in await table.keys():
                    self.assertGreater(len(await table.get(H)), 0)
                    self.assertTrue("a" not in await table.get(H))

            with self.assertRaises(ValueError):
                await lsh.remove("c")

    @unittest.skipIf(not DO_TEST_MONGO, "Skipping test_pickle_mongo")
    async def test_pickle_mongo(self):
        async with AsyncMinHashLSH(storage_config=self._storage_config_mongo,
                                   threshold=0.5, num_perm=4) as lsh:
            mg = WeightedMinHashGenerator(10, 4)
            m1 = mg.minhash(np.random.uniform(1, 10, 10))
            m2 = mg.minhash(np.random.uniform(1, 10, 10))
            await lsh.insert("a", m1)
            await lsh.insert("b", m2)
            pickled = pickle.dumps(lsh)

        async with pickle.loads(pickled) as lsh2:
            result = await lsh2.query(m1)
            self.assertTrue("a" in result)
            result = await lsh2.query(m2)
            self.assertTrue("b" in result)


if __name__ == "__main__":
    unittest.main()
