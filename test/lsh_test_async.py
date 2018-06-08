import unittest
import pickle
import numpy as np
import aiounittest

from datasketch.aiolsh import create_AsyncMinHashLSH
from datasketch.minhash import MinHash
from datasketch.weighted_minhash import WeightedMinHashGenerator

"""For tests Redis should be installed on local machine. Don't forget to clean Redis DB=0."""


class TestAsyncMinHashLSH(aiounittest.AsyncTestCase):
    def setUp(self):
        self._storage_config = {'type': 'aioredis', 'redis': {'host': 'localhost', 'port': 6379}}

    async def test_init(self):
        async with (await create_AsyncMinHashLSH(storage_config=self._storage_config, threshold=0.8)) as lsh:
            self.assertTrue(await lsh.is_empty())
            b1, r1 = lsh.b, lsh.r

        async with (await create_AsyncMinHashLSH(storage_config=self._storage_config,
                                                 threshold=0.8, weights=(0.2, 0.8))) as lsh:
            b2, r2 = lsh.b, lsh.r
        self.assertTrue(b1 < b2)
        self.assertTrue(r1 > r2)

    async def test__H(self):
        """
        Check _H output consistent bytes length given
        the same concatenated hash value size
        """
        for l in range(2, 128 + 1, 16):
                m = MinHash()
                m.update("abcdefg".encode("utf8"))
                m.update("1234567".encode("utf8"))
                async with (await create_AsyncMinHashLSH(storage_config=self._storage_config, num_perm=128)) as lsh:
                    await lsh.insert("m", m)
                    sizes = [len(H) for ht in lsh.hashtables for H in await ht.keys()]
                    self.assertTrue(all(sizes[0] == s for s in sizes))

    async def test_insert(self):
        async with (await create_AsyncMinHashLSH(storage_config=self._storage_config,
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
                self.assertTrue(pickle.dumps("a") in items)
                self.assertTrue(pickle.dumps("b") in items)
            self.assertTrue(await lsh.has_key("a"))
            self.assertTrue(await lsh.has_key("b"))
            for i, H in enumerate(await lsh.keys.get(pickle.dumps("a"))):
                self.assertTrue(pickle.dumps("a") in await lsh.hashtables[i].get(H))

            m3 = MinHash(18)
            with self.assertRaises(ValueError):
                await lsh.insert("c", m3)

    async def test_query(self):
        async with (await create_AsyncMinHashLSH(storage_config=self._storage_config,
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

    async def test_remove(self):
        async with (await create_AsyncMinHashLSH(storage_config=self._storage_config,
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

    async def test_pickle(self):
        lsh = await create_AsyncMinHashLSH(storage_config=self._storage_config, threshold=0.5, num_perm=16)
        m1 = MinHash(16)
        m1.update("a".encode("utf8"))
        m2 = MinHash(16)
        m2.update("b".encode("utf8"))
        await lsh.insert("a", m1)
        await lsh.insert("b", m2)
        lsh2 = pickle.loads(pickle.dumps(lsh))
        # Important! Initialize storage's after pickle.load/s. And don't forget to call class close() method
        await lsh2.init_storages()
        await lsh.close()

        result = await lsh2.query(m1)
        self.assertTrue("a" in result)
        result = await lsh2.query(m2)
        self.assertTrue("b" in result)
        await lsh2.close()

    async def test_insertion_session(self):
        async with (await create_AsyncMinHashLSH(storage_config=self._storage_config,
                                                 threshold=0.5, num_perm=16)) as lsh:
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
                self.assertTrue(pickle.dumps("a") in await lsh.hashtables[i].get(H))

    async def test_get_counts(self):
        async with (await create_AsyncMinHashLSH(storage_config=self._storage_config,
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


class TestWeightedMinHashLSH(aiounittest.AsyncTestCase):
    def setUp(self):
        self._storage_config = {'type': 'aioredis', 'redis': {'host': 'localhost', 'port': 6379}}

    async def test_init(self):
        async with (await create_AsyncMinHashLSH(storage_config=self._storage_config, threshold=0.8)) as lsh:
            self.assertTrue(await lsh.is_empty())
            b1, r1 = lsh.b, lsh.r
        async with (await create_AsyncMinHashLSH(storage_config=self._storage_config,
                                                 threshold=0.8, weights=(0.2, 0.8))) as lsh:
            b2, r2 = lsh.b, lsh.r
        self.assertTrue(b1 < b2)
        self.assertTrue(r1 > r2)

    async def test__H(self):
        """
        Check _H output consistent bytes length given
        the same concatenated hash value size
        """
        mg = WeightedMinHashGenerator(100, sample_size=128)
        for l in range(2, mg.sample_size + 1, 16):
            m = mg.minhash(np.random.randint(1, 99999999, 100))
            async with (await create_AsyncMinHashLSH(storage_config=self._storage_config, num_perm=128)) as lsh:
                await lsh.insert("m", m)
                sizes = [len(H) for ht in lsh.hashtables for H in await ht.keys()]
                self.assertTrue(all(sizes[0] == s for s in sizes))

    async def test_insert(self):
        async with (await create_AsyncMinHashLSH(storage_config=self._storage_config,
                                                 threshold=0.5, num_perm=4)) as lsh:
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

    async def test_query(self):
        async with (await create_AsyncMinHashLSH(storage_config=self._storage_config,
                                                 threshold=0.5, num_perm=4)) as lsh:
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

    async def test_remove(self):
        async with (await create_AsyncMinHashLSH(storage_config=self._storage_config,
                                                 threshold=0.5, num_perm=4)) as lsh:
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

    async def test_pickle(self):
        lsh = await create_AsyncMinHashLSH(storage_config=self._storage_config, threshold=0.5, num_perm=4)
        mg = WeightedMinHashGenerator(10, 4)
        m1 = mg.minhash(np.random.uniform(1, 10, 10))
        m2 = mg.minhash(np.random.uniform(1, 10, 10))
        await lsh.insert("a", m1)
        await lsh.insert("b", m2)
        lsh2 = pickle.loads(pickle.dumps(lsh))
        # Important! Initialize storage's after pickle.load/s. And don't forget to call class close() method
        await lsh2.init_storages()
        await lsh.close()

        result = await lsh2.query(m1)
        self.assertTrue("a" in result)
        result = await lsh2.query(m2)
        self.assertTrue("b" in result)
        await lsh2.close()


if __name__ == "__main__":
    unittest.main()
