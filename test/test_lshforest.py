import unittest
import pickle
import numpy as np
from datasketch.lshforest import MinHashLSHForest
from datasketch.minhash import MinHash
from datasketch import WeightedMinHashGenerator


class TestMinHashLSHForest(unittest.TestCase):

    def _setup(self):
        d = "abcdefghijklmnopqrstuvwxyz"
        forest = MinHashLSHForest()
        for i in range(len(d)-2):
            key = d[i]
            m = MinHash()
            j = i + 3
            for s in d[i:j]:
                m.update(s.encode("utf8"))
            forest.add(key, m)
        forest.index()
        return forest

    def test__H(self):
        '''
        Check _H output consistent bytes length given
        the same concatenated hash value size
        '''
        for l in range(2, 128+1, 16):
            forest = MinHashLSHForest(num_perm=128, l=l)
            m = MinHash()
            m.update("abcdefg".encode("utf8"))
            m.update("1234567".encode("utf8"))
            forest.add("m", m)
            sizes = [len(H) for ht in forest.hashtables for H in ht]
            self.assertTrue(all(sizes[0] == s for s in sizes))

    def test_init(self):
        forest = MinHashLSHForest()
        self.assertTrue(forest.is_empty())

    def test_add_index(self):
        forest = MinHashLSHForest()
        m1 = MinHash()
        m1.update("a".encode("utf8"))
        m2 = MinHash()
        m2.update("b".encode("utf8"))
        forest.add("a", m1)
        forest.add("b", m2)
        self.assertTrue(forest.is_empty())
        for t in forest.hashtables:
            self.assertTrue(len(t) >= 1)
            items = []
            for H in t:
                items.extend(t[H])
            self.assertTrue("a" in items)
            self.assertTrue("b" in items)
        self.assertTrue("a" in forest)
        self.assertTrue("b" in forest)
        for i, H in enumerate(forest.keys["a"]):
            self.assertTrue("a" in forest.hashtables[i][H])
        m3 = MinHash(18)
        self.assertRaises(ValueError, forest.add, "c", m3)
        forest.index()
        self.assertFalse(forest.is_empty())

    def test_query(self):
        m1 = MinHash()
        m1.update("a".encode("utf8"))
        m1.update("b".encode("utf8"))
        m1.update("c".encode("utf8"))
        forest = self._setup()
        result = forest.query(m1, 3)
        self.assertTrue("a" in result)
        self.assertTrue("b" in result)
        self.assertTrue("c" in result)

        m3 = MinHash(18)
        self.assertRaises(ValueError, forest.query, m3, 1)

    def test_pickle(self):
        forest = MinHashLSHForest()
        m1 = MinHash()
        m1.update("a".encode("utf8"))
        m2 = MinHash()
        m2.update("b".encode("utf8"))
        forest.add("a", m1)
        forest.add("b", m2)
        forest.index()
        forest2 = pickle.loads(pickle.dumps(forest))
        result = forest2.query(m1, 1)
        self.assertTrue("a" in result)
        result = forest2.query(m2, 1)
        self.assertTrue("b" in result)


class TestWeightedMinHashLSHForest(unittest.TestCase):

    def test__H(self):
        '''
        Check _H output consistent bytes length given
        the same concatenated hash value size
        '''
        mg = WeightedMinHashGenerator(100, sample_size=128)
        for l in range(2, mg.sample_size+1, 16):
            m = mg.minhash(np.random.randint(1, 99999999, 100))
            forest = MinHashLSHForest(num_perm=128, l=l)
            forest.add("m", m)
            sizes = [len(H) for ht in forest.hashtables for H in ht]
            self.assertTrue(all(sizes[0] == s for s in sizes))

    def test_insert(self):
        forest = MinHashLSHForest()
        mg = WeightedMinHashGenerator(10)
        m1 = mg.minhash(np.random.uniform(1, 10, 10))
        m2 = mg.minhash(np.random.uniform(1, 10, 10))
        forest.add("a", m1)
        forest.add("b", m2)

        self.assertTrue(forest.is_empty())
        for t in forest.hashtables:
            self.assertTrue(len(t) >= 1)
            items = []
            for H in t:
                items.extend(t[H])
            self.assertTrue("a" in items)
            self.assertTrue("b" in items)
        self.assertTrue("a" in forest)
        self.assertTrue("b" in forest)
        for i, H in enumerate(forest.keys["a"]):
            self.assertTrue("a" in forest.hashtables[i][H])

        forest.index()
        self.assertFalse(forest.is_empty())

        mg = WeightedMinHashGenerator(10, 5)
        m3 = mg.minhash(np.random.uniform(1, 10, 10))
        self.assertRaises(ValueError, forest.add, "c", m3)

    def test_query(self):
        forest = MinHashLSHForest()
        mg = WeightedMinHashGenerator(10)
        m1 = mg.minhash(np.random.uniform(1, 10, 10))
        m2 = mg.minhash(np.random.uniform(1, 10, 10))
        forest.add("a", m1)
        forest.add("b", m2)
        forest.index()
        result = forest.query(m1, 2)
        self.assertTrue("a" in result)
        self.assertTrue("b" in result)

        mg = WeightedMinHashGenerator(10, 5)
        m3 = mg.minhash(np.random.uniform(1, 10, 10))
        self.assertRaises(ValueError, forest.query, m3, 1)

    def test_pickle(self):
        forest = MinHashLSHForest()
        mg = WeightedMinHashGenerator(10)
        m1 = mg.minhash(np.random.uniform(1, 10, 10))
        m2 = mg.minhash(np.random.uniform(1, 10, 10))
        forest.add("a", m1)
        forest.add("b", m2)
        forest.index()
        forest2 = pickle.loads(pickle.dumps(forest))
        result = forest2.query(m1, 2)
        self.assertTrue("a" in result)
        self.assertTrue("b" in result)

if __name__ == "__main__":
    unittest.main()
