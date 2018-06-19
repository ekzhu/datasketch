import asyncio
import pickle
import unittest

import aioredis
import aiounittest
import numpy as np

from datasketch.async.lsh import AsyncMinHashLSH
from datasketch.minhash import MinHash
from datasketch.weighted_minhash import WeightedMinHashGenerator

"""For tests Mongo should be installed on local machine or set right host. Don't forget to clean Mongo DB=db_0."""

class AsyncTestCase(aiounittest.AsyncTestCase):
    def get_event_loop(self):
        return asyncio.get_event_loop()

    async def setUpAsync(self):
        pass

    async def tearDownAsync(self):
        pass

    def setUp(self):
        if hasattr(super(), 'setUp'):
            super().setUp()

        loop = self.get_event_loop()
        loop.run_until_complete(self.setUpAsync())

    def tearDown(self):
        if hasattr(super(), 'tearDown'):
            super().tearDown()

        loop = self.get_event_loop()
        loop.run_until_complete(self.tearDownAsync())


class TestAsyncMinHashLSH(AsyncTestCase):
    """For tests Redis should be installed on local machine.
    Don't forget to clean Redis DB=0."""

    def setUp(self):
        super().setUp()
        self._storage_config_redis = {'type': 'aioredis', 'redis': {'host': 'localhost', 'port': 6379}}
        self._storage_config_mongo = {'type': 'aiomongo', 'mongo': {'host': '10.216.20.31', 'port': 27017}}

    async def tearDownAsync(self):
        dsn = 'redis://{host}:{port}'.format(**self._storage_config_redis['redis'])
        redis = await aioredis.create_redis(dsn, loop=self.get_event_loop())
        await redis.flushall()
        redis.close()
        await redis.wait_closed()

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

    async def test_init_mongo(self):
        async with (await create_AsyncMinHashLSH(storage_config=self._storage_config_mongo, threshold=0.8)) as lsh:
            self.assertTrue(await lsh.is_empty())
            b1, r1 = lsh.b, lsh.r

        async with (await create_AsyncMinHashLSH(storage_config=self._storage_config_mongo,
                                                 threshold=0.8, weights=(0.2, 0.8))) as lsh:
            b2, r2 = lsh.b, lsh.r
        self.assertTrue(b1 < b2)
        self.assertTrue(r1 > r2)

    async def test__H_mongo(self):
        """
        Check _H output consistent bytes length given
        the same concatenated hash value size
        """
        for l in range(2, 128 + 1, 16):
                m = MinHash()
                m.update("abcdefg".encode("utf8"))
                m.update("1234567".encode("utf8"))
                async with (await create_AsyncMinHashLSH(storage_config=self._storage_config_mongo, num_perm=128)) as lsh:
                    await lsh.insert("m", m)
                    sizes = [len(H) for ht in lsh.hashtables for H in await ht.keys()]
                    self.assertTrue(all(sizes[0] == s for s in sizes))

    async def test_insert_mongo(self):
        async with (await create_AsyncMinHashLSH(storage_config=self._storage_config_mongo,
                                                 threshold=0.5, num_perm=16)) as lsh:
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
                self.assertTrue("a" in items)
                self.assertTrue("b" in items)
            self.assertTrue(await lsh.has_key("a"))
            self.assertTrue(await lsh.has_key("b"))
            for i, H in enumerate(await lsh.keys.get("a")):
                self.assertTrue("a" in await lsh.hashtables[i].get(H))

            m3 = MinHash(18)
            with self.assertRaises(ValueError):
                await lsh.insert("c", m3)

    async def test_query_mongo(self):
        async with (await create_AsyncMinHashLSH(storage_config=self._storage_config_mongo,
                                                 threshold=0.5, num_perm=16)) as lsh:
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

    async def test_remove_mongo(self):
        async with (await create_AsyncMinHashLSH(storage_config=self._storage_config_mongo,
                                                 threshold=0.5, num_perm=16)) as lsh:
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

    async def test_pickle_mongo(self):
        lsh = await create_AsyncMinHashLSH(storage_config=self._storage_config_mongo, threshold=0.5, num_perm=16)
        m1 = MinHash(16)
        m1.update("a".encode("utf8"))
        m2 = MinHash(16)
        m2.update("b".encode("utf8"))
        await lsh.insert("a", m1)
        await lsh.insert("b", m2)
        lsh2 = pickle.loads(pickle.dumps(lsh))

        result = await lsh2.query(m1)
        self.assertTrue("a" in result)
        result = await lsh2.query(m2)
        self.assertTrue("b" in result)
        await lsh2.close()

    # async def test_insertion_session_mongo(self):
    #     async with (await create_AsyncMinHashLSH(storage_config=self._storage_config_redis,
    #                                              threshold=0.5, num_perm=16)) as lsh:
    #         m1 = MinHash(16)
    #         m1.update("a".encode("utf8"))
    #         m2 = MinHash(16)
    #         m2.update("b".encode("utf8"))
    #         data = [("a", m1), ("b", m2)]
    #         async with lsh.insertion_session() as session:
    #             for key, minhash in data:
    #                 await session.insert(key, minhash)
    #         for t in lsh.hashtables:
    #             self.assertTrue(await t.size() >= 1)
    #             items = []
    #             for H in await t.keys():
    #                 items.extend(await t.get(H))
    #             self.assertTrue(pickle.dumps("a") in items)
    #             self.assertTrue(pickle.dumps("b") in items)
    #         self.assertTrue(await lsh.has_key("a"))
    #         self.assertTrue(await lsh.has_key("b"))
    #         for i, H in enumerate(await lsh.keys.get(pickle.dumps("a"))):
    #             self.assertTrue(pickle.dumps("a") in await lsh.hashtables[i].get(H))

    async def test_get_counts_mongo(self):
        async with (await create_AsyncMinHashLSH(storage_config=self._storage_config_mongo,
                                                 threshold=0.5, num_perm=16)) as lsh:
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


class TestWeightedMinHashLSH(AsyncTestCase):
    """For tests Redis should be installed on local machine.
    Don't forget to clean Redis DB=0."""

    def setUp(self):
        super().setUp()
        self._storage_config_redis = {
            'type': 'aioredis',
            'redis': {'host': 'localhost', 'port': 6379}
        }

    async def tearDownAsync(self):
        dsn = 'redis://{host}:{port}'.format(**self._storage_config_redis['redis'])
        redis = await aioredis.create_redis(dsn, loop=self.get_event_loop())
        await redis.flushall()

    async def test_init(self):
        async with AsyncMinHashLSH(storage_config=self._storage_config_redis,
                                   threshold=0.8) as lsh:
            self.assertTrue(await lsh.is_empty())
            b1, r1 = lsh.b, lsh.r
        async with AsyncMinHashLSH(storage_config=self._storage_config_redis,
                                   threshold=0.8, weights=(0.2, 0.8)) as lsh:
            b2, r2 = lsh.b, lsh.r
        self.assertTrue(b1 < b2)
        self.assertTrue(r1 > r2)

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
                res = await lsh.hashtables[i].get(H)
                self.assertTrue(pickle.dumps("a") in res)

            mg = WeightedMinHashGenerator(10, 5)
            m3 = mg.minhash(np.random.uniform(1, 10, 10))
            with self.assertRaises(ValueError):
                await lsh.insert("c", m3)

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


if __name__ == "__main__":
    unittest.main()
