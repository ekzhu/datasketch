import unittest
import pickle
import numpy as np
import mockredis
from mock import patch
from datasketch.lsh import MinHashLSH
from datasketch.minhash import MinHash
from datasketch.weighted_minhash import WeightedMinHashGenerator


def fake_redis(**kwargs):
    redis = mockredis.mock_redis_client(**kwargs)
    redis.connection_pool = None
    redis.response_callbacks = None
    return redis


class TestMinHashLSH(unittest.TestCase):

    def test_init(self):
        lsh = MinHashLSH(threshold=0.8)
        self.assertTrue(lsh.is_empty())
        b1, r1 = lsh.b, lsh.r
        lsh = MinHashLSH(threshold=0.8, weights=(0.2,0.8))
        b2, r2 = lsh.b, lsh.r
        self.assertTrue(b1 < b2)
        self.assertTrue(r1 > r2)

    def test__H(self):
        '''
        Check _H output consistent bytes length given
        the same concatenated hash value size
        '''
        for l in range(2, 128+1, 16):
            lsh = MinHashLSH(num_perm=128)
            m = MinHash()
            m.update("abcdefg".encode("utf8"))
            m.update("1234567".encode("utf8"))
            lsh.insert("m", m)
            sizes = [len(H) for ht in lsh.hashtables for H in ht]
            self.assertTrue(all(sizes[0] == s for s in sizes))

    def test_insert(self):
        lsh = MinHashLSH(threshold=0.5, num_perm=16)
        m1 = MinHash(16)
        m1.update("a".encode("utf8"))
        m2 = MinHash(16)
        m2.update("b".encode("utf8"))
        lsh.insert("a", m1)
        lsh.insert("b", m2)
        for t in lsh.hashtables:
            self.assertTrue(len(t) >= 1)
            items = []
            for H in t:
                items.extend(t[H])
            self.assertTrue("a" in items)
            self.assertTrue("b" in items)
        self.assertTrue("a" in lsh)
        self.assertTrue("b" in lsh)
        for i, H in enumerate(lsh.keys["a"]):
            self.assertTrue("a" in lsh.hashtables[i][H])

        m3 = MinHash(18)
        self.assertRaises(ValueError, lsh.insert, "c", m3)

    def test_query(self):
        lsh = MinHashLSH(threshold=0.5, num_perm=16)
        m1 = MinHash(16)
        m1.update("a".encode("utf8"))
        m2 = MinHash(16)
        m2.update("b".encode("utf8"))
        lsh.insert("a", m1)
        lsh.insert("b", m2)
        result = lsh.query(m1)
        self.assertTrue("a" in result)
        result = lsh.query(m2)
        self.assertTrue("b" in result)

        m3 = MinHash(18)
        self.assertRaises(ValueError, lsh.query, m3)

    def test_query_buffer(self):
        lsh = MinHashLSH(threshold=0.5, num_perm=16)
        m1 = MinHash(16)
        m1.update("a".encode("utf8"))
        m2 = MinHash(16)
        m2.update("b".encode("utf8"))
        lsh.insert("a", m1)
        lsh.insert("b", m2)
        lsh.add_to_query_buffer(m1)
        result = lsh.collect_query_buffer()
        self.assertTrue("a" in result)
        lsh.add_to_query_buffer(m2)
        result = lsh.collect_query_buffer()
        self.assertTrue("b" in result)
        m3 = MinHash(18)
        self.assertRaises(ValueError, lsh.add_to_query_buffer, m3)

    def test_remove(self):
        lsh = MinHashLSH(threshold=0.5, num_perm=16)
        m1 = MinHash(16)
        m1.update("a".encode("utf8"))
        m2 = MinHash(16)
        m2.update("b".encode("utf8"))
        lsh.insert("a", m1)
        lsh.insert("b", m2)

        lsh.remove("a")
        self.assertTrue("a" not in lsh.keys)
        for table in lsh.hashtables:
            for H in table:
                self.assertGreater(len(table[H]), 0)
                self.assertTrue("a" not in table[H])

        self.assertRaises(ValueError, lsh.remove, "c")

    def test_pickle(self):
        lsh = MinHashLSH(threshold=0.5, num_perm=16)
        m1 = MinHash(16)
        m1.update("a".encode("utf8"))
        m2 = MinHash(16)
        m2.update("b".encode("utf8"))
        lsh.insert("a", m1)
        lsh.insert("b", m2)
        lsh2 = pickle.loads(pickle.dumps(lsh))
        result = lsh2.query(m1)
        self.assertTrue("a" in result)
        result = lsh2.query(m2)
        self.assertTrue("b" in result)

    def test_insert_redis(self):
        with patch('redis.Redis', fake_redis) as mock_redis:
            lsh = MinHashLSH(threshold=0.5, num_perm=16, storage_config={
                'type': 'redis', 'redis': {'host': 'localhost', 'port': 6379}
            })
            m1 = MinHash(16)
            m1.update("a".encode("utf8"))
            m2 = MinHash(16)
            m2.update("b".encode("utf8"))
            lsh.insert("a", m1)
            lsh.insert("b", m2)
            for t in lsh.hashtables:
                self.assertTrue(len(t) >= 1)
                items = []
                for H in t:
                    items.extend(t[H])
                self.assertTrue(pickle.dumps("a") in items)
                self.assertTrue(pickle.dumps("b") in items)
            self.assertTrue("a" in lsh)
            self.assertTrue("b" in lsh)
            for i, H in enumerate(lsh.keys[pickle.dumps("a")]):
                self.assertTrue(pickle.dumps("a") in lsh.hashtables[i][H])

            m3 = MinHash(18)
            self.assertRaises(ValueError, lsh.insert, "c", m3)

    def test_query_redis(self):
        with patch('redis.Redis', fake_redis) as mock_redis:
            lsh = MinHashLSH(threshold=0.5, num_perm=16, storage_config={
                'type': 'redis', 'redis': {'host': 'localhost', 'port': 6379}
            })
            m1 = MinHash(16)
            m1.update("a".encode("utf8"))
            m2 = MinHash(16)
            m2.update("b".encode("utf8"))
            lsh.insert("a", m1)
            lsh.insert("b", m2)
            result = lsh.query(m1)
            self.assertTrue("a" in result)
            result = lsh.query(m2)
            self.assertTrue("b" in result)

            m3 = MinHash(18)
            self.assertRaises(ValueError, lsh.query, m3)

    def test_query_buffer_redis(self):
        with patch('redis.Redis', fake_redis) as mock_redis:
            lsh = MinHashLSH(threshold=0.5, num_perm=16, storage_config={
                'type': 'redis', 'redis': {'host': 'localhost', 'port': 6379}
            })
            m1 = MinHash(16)
            m1.update("a".encode("utf8"))
            m2 = MinHash(16)
            m2.update("b".encode("utf8"))
            lsh.insert("a", m1)
            lsh.insert("b", m2)
            lsh.query(m1)
            lsh.add_to_query_buffer(m1)
            result = lsh.collect_query_buffer()
            self.assertTrue("a" in result)
            lsh.add_to_query_buffer(m2)
            result = lsh.collect_query_buffer()
            self.assertTrue("b" in result)

            m3 = MinHash(18)
            self.assertRaises(ValueError, lsh.add_to_query_buffer, m3)

    def test_insertion_session(self):
        lsh = MinHashLSH(threshold=0.5, num_perm=16)
        m1 = MinHash(16)
        m1.update("a".encode("utf8"))
        m2 = MinHash(16)
        m2.update("b".encode("utf8"))
        data = [("a", m1), ("b", m2)]
        with lsh.insertion_session() as session:
            for key, minhash in data:
                session.insert(key, minhash)
        for t in lsh.hashtables:
            self.assertTrue(len(t) >= 1)
            items = []
            for H in t:
                items.extend(t[H])
            self.assertTrue("a" in items)
            self.assertTrue("b" in items)
        self.assertTrue("a" in lsh)
        self.assertTrue("b" in lsh)
        for i, H in enumerate(lsh.keys["a"]):
            self.assertTrue("a" in lsh.hashtables[i][H])

    def test_get_counts(self):
        lsh = MinHashLSH(threshold=0.5, num_perm=16)
        m1 = MinHash(16)
        m1.update("a".encode("utf8"))
        m2 = MinHash(16)
        m2.update("b".encode("utf8"))
        lsh.insert("a", m1)
        lsh.insert("b", m2)
        counts = lsh.get_counts()
        self.assertEqual(len(counts), lsh.b)
        for table in counts:
            self.assertEqual(sum(table.values()), 2)


class TestWeightedMinHashLSH(unittest.TestCase):

    def test_init(self):
        lsh = MinHashLSH(threshold=0.8)
        self.assertTrue(lsh.is_empty())
        b1, r1 = lsh.b, lsh.r
        lsh = MinHashLSH(threshold=0.8, weights=(0.2,0.8))
        b2, r2 = lsh.b, lsh.r
        self.assertTrue(b1 < b2)
        self.assertTrue(r1 > r2)

    def test__H(self):
        '''
        Check _H output consistent bytes length given
        the same concatenated hash value size
        '''
        mg = WeightedMinHashGenerator(100, sample_size=128)
        for l in range(2, mg.sample_size+1, 16):
            m = mg.minhash(np.random.randint(1, 99999999, 100))
            lsh = MinHashLSH(num_perm=128)
            lsh.insert("m", m)
            sizes = [len(H) for ht in lsh.hashtables for H in ht]
            self.assertTrue(all(sizes[0] == s for s in sizes))

    def test_insert(self):
        lsh = MinHashLSH(threshold=0.5, num_perm=4)
        mg = WeightedMinHashGenerator(10, 4)
        m1 = mg.minhash(np.random.uniform(1, 10, 10))
        m2 = mg.minhash(np.random.uniform(1, 10, 10))
        lsh.insert("a", m1)
        lsh.insert("b", m2)
        for t in lsh.hashtables:
            self.assertTrue(len(t) >= 1)
            items = []
            for H in t:
                items.extend(t[H])
            self.assertTrue("a" in items)
            self.assertTrue("b" in items)
        self.assertTrue("a" in lsh)
        self.assertTrue("b" in lsh)
        for i, H in enumerate(lsh.keys["a"]):
            self.assertTrue("a" in lsh.hashtables[i][H])

        mg = WeightedMinHashGenerator(10, 5)
        m3 = mg.minhash(np.random.uniform(1, 10, 10))
        self.assertRaises(ValueError, lsh.insert, "c", m3)

    def test_query(self):
        lsh = MinHashLSH(threshold=0.5, num_perm=4)
        mg = WeightedMinHashGenerator(10, 4)
        m1 = mg.minhash(np.random.uniform(1, 10, 10))
        m2 = mg.minhash(np.random.uniform(1, 10, 10))
        lsh.insert("a", m1)
        lsh.insert("b", m2)
        result = lsh.query(m1)
        self.assertTrue("a" in result)
        result = lsh.query(m2)
        self.assertTrue("b" in result)

        mg = WeightedMinHashGenerator(10, 5)
        m3 = mg.minhash(np.random.uniform(1, 10, 10))
        self.assertRaises(ValueError, lsh.query, m3)

    def test_remove(self):
        lsh = MinHashLSH(threshold=0.5, num_perm=4)
        mg = WeightedMinHashGenerator(10, 4)
        m1 = mg.minhash(np.random.uniform(1, 10, 10))
        m2 = mg.minhash(np.random.uniform(1, 10, 10))
        lsh.insert("a", m1)
        lsh.insert("b", m2)

        lsh.remove("a")
        self.assertTrue("a" not in lsh.keys)
        for table in lsh.hashtables:
            for H in table:
                self.assertGreater(len(table[H]), 0)
                self.assertTrue("a" not in table[H])

        self.assertRaises(ValueError, lsh.remove, "c")

    def test_pickle(self):
        lsh = MinHashLSH(threshold=0.5, num_perm=4)
        mg = WeightedMinHashGenerator(10, 4)
        m1 = mg.minhash(np.random.uniform(1, 10, 10))
        m2 = mg.minhash(np.random.uniform(1, 10, 10))
        lsh.insert("a", m1)
        lsh.insert("b", m2)
        lsh2 = pickle.loads(pickle.dumps(lsh))

        result = lsh2.query(m1)
        self.assertTrue("a" in result)
        result = lsh2.query(m2)
        self.assertTrue("b" in result)


if __name__ == "__main__":
    unittest.main()
