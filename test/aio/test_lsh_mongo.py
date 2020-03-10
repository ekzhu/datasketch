import sys
import os
import unittest
import asyncio
import pickle
import random
import string
import aiounittest
import numpy as np

from itertools import chain, islice
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient

from datasketch.experimental.aio.lsh import AsyncMinHashLSH
from datasketch.minhash import MinHash
from datasketch.weighted_minhash import WeightedMinHashGenerator

STORAGE_CONFIG_MONGO = {'type': 'aiomongo'}
DO_TEST_MONGO = os.environ.get("DO_TEST_MONGO")

MONGO_URL = os.environ.get("MONGO_UNIT_TEST_URL")
if MONGO_URL:
    STORAGE_CONFIG_MONGO['mongo'] = {'url': MONGO_URL}
else:
    STORAGE_CONFIG_MONGO['mongo'] = {'host': 'localhost', 'port': 27017, 'db': 'lsh_test'}


@unittest.skipIf(sys.version_info < (3, 6), "Skipping TestAsyncMinHashLSH. Supported Python version >= 3.6")
class TestAsyncMinHashLSH(aiounittest.AsyncTestCase):
    """
        For tests Mongo should be installed on local machine or set right host.
    """

    def setUp(self):
        self._storage_config_mongo = STORAGE_CONFIG_MONGO

    def tearDown(self):
        if DO_TEST_MONGO:
            dsn = MONGO_URL or 'mongodb://{host}:{port}'.format(**self._storage_config_mongo['mongo'])
            MongoClient(dsn).drop_database(self._storage_config_mongo['mongo']['db'])

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
                sizes = []
                for ht in lsh.hashtables:
                    keys = await ht.keys()
                    for H in keys:
                        sizes.append(len(H))
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
            m3 = MinHash(16)
            m3.update("b".encode("utf8"))
            fs = (lsh.insert("a", m1, check_duplication=False), lsh.insert("b", m2, check_duplication=False),
                  lsh.insert("b", m3, check_duplication=False))
            await asyncio.gather(*fs)
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
            m3 = MinHash(16)
            m3.update("a".encode("utf8"))
            await lsh.insert("a", m1)
            await lsh.insert("b", m2)
            await lsh.insert("a1", m3)

            await lsh.remove("a")
            self.assertTrue(not await lsh.has_key("a"))
            self.assertTrue(await lsh.has_key('a1'))
            hashtable_correct = False
            for table in lsh.hashtables:
                for H in await table.keys():
                    table_vals = await table.get(H)
                    self.assertGreater(len(table_vals), 0)
                    self.assertTrue("a" not in table_vals)
                    if 'a1' in table_vals:
                        hashtable_correct = True
            self.assertTrue(hashtable_correct, 'Hashtable broken')

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
                                   threshold=0.5, num_perm=16) as lsh:
            async with lsh.insertion_session(batch_size=1000) as session:
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

    @unittest.skipIf(not DO_TEST_MONGO, "Skipping test_insertion_session_mongo")
    async def test_remove_session_mongo(self):
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
        keys_to_remove = ('aahhb', 'aahh', 'aahhc', 'aac', 'kld', 'bhg', 'kkd', 'yow', 'ppi', 'eer')
        keys_left = frozenset(seq) - frozenset(keys_to_remove)

        async with AsyncMinHashLSH(storage_config=self._storage_config_mongo,
                                   threshold=0.5, num_perm=16) as lsh:
            async with lsh.insertion_session(batch_size=1000) as session:
                fs = (session.insert(key, minhash, check_duplication=False) for key, minhash in data)
                await asyncio.gather(*fs)

            async with lsh.delete_session(batch_size=3) as session:
                fs = (session.remove(key) for key in keys_to_remove)
                await asyncio.gather(*fs)

            for t in lsh.hashtables:
                self.assertTrue(await t.size() >= 1)
                items = []
                for H in await t.keys():
                    items.extend(await t.get(H))
                for key in keys_to_remove:
                    self.assertTrue(key not in items, '{0} in items, but should not be'.format(key))
                for key in keys_left:
                    self.assertTrue(key in items, '{0} not in items, but should be'.format(key))

            for key in keys_to_remove:
                self.assertTrue(not (await lsh.has_key(key)), '<{0}> key should not be in LSH index'.format(key))
            for key in keys_left:
                self.assertTrue(await lsh.has_key(key), '<{0}> key should be in LSH index'.format(key))

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

    @unittest.skipIf(not DO_TEST_MONGO, "Skipping test_arbitrary_url")
    async def test_arbitrary_url(self):
        config = {"type": "aiomongo", "mongo": {"url": MONGO_URL or "mongodb://localhost/lsh_test"}}
        async with AsyncMinHashLSH(storage_config=config, threshold=0.5, num_perm=16) as lsh:
            m1 = MinHash(16)
            m1.update(b"a")
            await lsh.insert("a", m1)

        database = AsyncIOMotorClient(config["mongo"]["url"]).get_default_database("lsh_test")
        collection_names = await database.list_collection_names()
        self.assertGreater(len(collection_names), 0)
        await database.client.drop_database(database.name)

    @unittest.skipIf(not DO_TEST_MONGO, "Skipping test_arbitrary_collection")
    async def test_arbitrary_collection(self):
        self._storage_config_mongo["mongo"]["collection_name"] = "unit_test_collection"
        async with AsyncMinHashLSH(storage_config=self._storage_config_mongo, threshold=0.5, num_perm=16) as lsh:
            m1 = MinHash(16)
            m1.update(b"a")
            await lsh.insert("a", m1)

        dsn = MONGO_URL or "mongodb://{host}:{port}/{db}".format(**self._storage_config_mongo["mongo"])
        collection = AsyncIOMotorClient(dsn).get_default_database("lsh_test").get_collection("unit_test_collection")
        count = await collection.count_documents({})

        self.assertGreaterEqual(count, 1)
        del self._storage_config_mongo["mongo"]["collection_name"]


@unittest.skipIf(sys.version_info < (3, 6), "Skipping TestAsyncMinHashLSH. Supported Python version >= 3.6")
class TestWeightedMinHashLSH(aiounittest.AsyncTestCase):
    """For tests Redis should be installed on local machine.
    Don't forget to clean Redis DB=0."""

    def setUp(self):
        super().setUp()
        self._storage_config_mongo = STORAGE_CONFIG_MONGO

    def tearDown(self):
        if DO_TEST_MONGO:
            dsn = 'mongodb://{host}:{port}'.format(**self._storage_config_mongo['mongo'])
            MongoClient(dsn).drop_database(self._storage_config_mongo['mongo']['db'])

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


def test_suite_minhashlsh_aiomongo():
    suite = unittest.TestSuite()
    suite.addTest(TestAsyncMinHashLSH('test_init_mongo'))
    suite.addTest(TestAsyncMinHashLSH('test__H_mongo'))
    suite.addTest(TestAsyncMinHashLSH('test_insert_mongo'))
    suite.addTest(TestAsyncMinHashLSH('test_query_mongo'))
    suite.addTest(TestAsyncMinHashLSH('test_remove_mongo'))
    suite.addTest(TestAsyncMinHashLSH('test_pickle_mongo'))
    suite.addTest(TestAsyncMinHashLSH('test_insertion_session_mongo'))
    suite.addTest(TestAsyncMinHashLSH('test_remove_session_mongo'))
    suite.addTest(TestAsyncMinHashLSH('test_get_counts_mongo'))
    return suite


def test_suite_weightedminhashlsh_aiomongo():
    suite = unittest.TestSuite()
    suite.addTest(TestWeightedMinHashLSH('test_init_mongo'))
    suite.addTest(TestWeightedMinHashLSH('test__H_mongo'))
    suite.addTest(TestWeightedMinHashLSH('test_insert_mongo'))
    suite.addTest(TestWeightedMinHashLSH('test_query_mongo'))
    suite.addTest(TestWeightedMinHashLSH('test_remove_mongo'))
    suite.addTest(TestWeightedMinHashLSH('test_pickle_mongo'))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    if DO_TEST_MONGO:
        runner.run(test_suite_minhashlsh_aiomongo())
        runner.run(test_suite_weightedminhashlsh_aiomongo())

