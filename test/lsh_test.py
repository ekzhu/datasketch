import unittest
from hashlib import sha1
import pickle
import numpy as np
from datasketch.lsh import MinHashLSH, WeightedMinHashLSH
from datasketch.minhash import MinHash
from datasketch.weighted_minhash import WeightedMinHashGenerator


class TestMinHashLSH(unittest.TestCase):

    def test_init(self):
        lsh = MinHashLSH(threshold=0.8)
        self.assertTrue(lsh.is_empty())
        b1, r1 = lsh.b, lsh.r
        lsh = MinHashLSH(threshold=0.8, weights=(0.2,0.8))
        b2, r2 = lsh.b, lsh.r
        self.assertTrue(b1 < b2)
        self.assertTrue(r1 > r2)

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

    def test_pickle(self):
        lsh = MinHashLSH(threshold=0.5, num_perm=16)
        m1 = MinHash(16)
        m1.update("a".encode("utf8"))
        m2 = MinHash(16)
        m2.update("b".encode("utf8"))
        lsh.insert("a", m1)
        lsh.insert("b", m2)
        lsh2 = pickle.loads(pickle.dumps(lsh))
        result = lsh.query(m1)
        self.assertTrue("a" in result)
        result = lsh.query(m2)
        self.assertTrue("b" in result)


class TestWeightedMinHashLSH(unittest.TestCase):

    def test_init(self):
        lsh = WeightedMinHashLSH(threshold=0.8)
        self.assertTrue(lsh.is_empty())
        b1, r1 = lsh.b, lsh.r
        lsh = WeightedMinHashLSH(threshold=0.8, weights=(0.2,0.8))
        b2, r2 = lsh.b, lsh.r
        self.assertTrue(b1 < b2)
        self.assertTrue(r1 > r2)

    def test_insert(self):
        lsh = WeightedMinHashLSH(threshold=0.5, sample_size=4)
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

        mg = WeightedMinHashGenerator(10, 5)
        m3 = mg.minhash(np.random.uniform(1, 10, 10))
        self.assertRaises(ValueError, lsh.insert, "c", m3)

    def test_query(self):
        lsh = WeightedMinHashLSH(threshold=0.5, sample_size=4)
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

    def test_pickle(self):
        lsh = WeightedMinHashLSH(threshold=0.5, sample_size=4)
        mg = WeightedMinHashGenerator(10, 4)
        m1 = mg.minhash(np.random.uniform(1, 10, 10))
        m2 = mg.minhash(np.random.uniform(1, 10, 10))
        lsh.insert("a", m1)
        lsh.insert("b", m2)
        lsh2 = pickle.loads(pickle.dumps(lsh))
        result = lsh.query(m1)
        self.assertTrue("a" in result)
        result = lsh.query(m2)
        self.assertTrue("b" in result)

if __name__ == "__main__":
    unittest.main()
