import pickle
import unittest

import numpy as np

from datasketch import WeightedMinHashGenerator
from datasketch.lshforest import MinHashLSHForest
from datasketch.minhash import MinHash


class TestMinHashLSHForest(unittest.TestCase):
    def _setup(self):
        d = "abcdefghijklmnopqrstuvwxyz"
        data = {}
        forest = MinHashLSHForest()
        for i in range(len(d) - 2):
            key = d[i]
            m = MinHash()
            j = i + 3
            for s in d[i:j]:
                m.update(s.encode("utf8"))
            data[key] = m
            forest.add(key, m)
        self.assertTrue(forest.is_empty())
        for t in forest.hashtables:
            self.assertTrue(len(t) >= 1)
            items = []
            for H in t:
                items.extend(t[H])
            for key in data:
                self.assertTrue(key in items)
        for key in data:
            self.assertTrue(key in forest)
        for key in forest.keys:
            for i, H in enumerate(forest.keys[key]):
                self.assertTrue(key in forest.hashtables[i][H])
        self.assertRaises(ValueError, forest.add, "a", data["a"])
        forest.index()
        self.assertFalse(forest.is_empty())
        self.assertRaises(ValueError, forest.add, "a", data["a"])
        return forest, data

    def test__H(self):
        """Check _H output consistent bytes length given
        the same concatenated hash value size.
        """
        for l in range(2, 128 + 1, 16):
            forest = MinHashLSHForest(num_perm=128, l=l)
            m = MinHash()
            m.update(b"abcdefg")
            m.update(b"1234567")
            forest.add("m", m)
            sizes = [len(H) for ht in forest.hashtables for H in ht]
            self.assertTrue(all(sizes[0] == s for s in sizes))

    def test_init(self):
        forest = MinHashLSHForest()
        self.assertTrue(forest.is_empty())

    def test_query(self):
        forest, data = self._setup()
        for key in data:
            results = forest.query(data[key], 10)
            self.assertIn(key, results)

    def test_get_minhash_hashvalues(self):
        forest, data = self._setup()
        for key in data:
            minhash_ori = data[key]
            hashvalues = forest.get_minhash_hashvalues(key)
            minhash_retrieved = MinHash(seed=minhash_ori.seed, hashvalues=hashvalues, scheme=minhash_ori.scheme)
            retrieved_hashvalues = minhash_retrieved.hashvalues
            self.assertEqual(len(hashvalues), len(retrieved_hashvalues))
            self.assertEqual(minhash_retrieved.jaccard(minhash_ori), 1.0)
            for i in range(len(retrieved_hashvalues)):
                self.assertEqual(hashvalues[i], retrieved_hashvalues[i])

    def test_get_minhash_hashvalues_rejects_corrupt_segments(self):
        forest, data = self._setup()
        key = next(iter(data))
        segments = forest.keys[key]
        # Truncated first segment: no longer a whole number of hash values
        # of a supported width, must not be silently reinterpreted.
        forest.keys[key] = [segments[0][:-3], *segments[1:]]
        with self.assertRaises(ValueError):
            forest.get_minhash_hashvalues(key)
        # First segment intact but a later one truncated: inconsistent sizes.
        forest.keys[key] = [*segments[:-1], segments[-1][: len(segments[-1]) // 2]]
        with self.assertRaises(ValueError):
            forest.get_minhash_hashvalues(key)
        forest.keys[key] = segments

    def test_pickle(self):
        forest, _ = self._setup()
        forest2 = pickle.loads(pickle.dumps(forest))
        self.assertEqual(forest.hashtables, forest2.hashtables)
        self.assertEqual(forest.keys, forest2.keys)
        self.assertEqual(forest.l, forest2.l)
        self.assertEqual(forest.k, forest2.k)
        self.assertEqual(forest.hashranges, forest2.hashranges)


class TestWeightedMinHashLSHForest(unittest.TestCase):
    def _setup(self):
        freqs = np.random.randint(1, 100, (100, 100))
        data = {}
        forest = MinHashLSHForest()
        mg = WeightedMinHashGenerator(100, sample_size=128)
        for i in range(len(freqs)):
            m = mg.minhash(freqs[i])
            forest.add(i, m)
            data[i] = m
        self.assertTrue(forest.is_empty())
        for t in forest.hashtables:
            self.assertTrue(len(t) >= 1)
            items = []
            for H in t:
                items.extend(t[H])
            for key in data:
                self.assertTrue(key in items)
        for key in data:
            self.assertTrue(key in forest)
        for key in forest.keys:
            for i, H in enumerate(forest.keys[key]):
                self.assertTrue(key in forest.hashtables[i][H])

        self.assertRaises(ValueError, forest.add, 5, data[5])

        forest.index()
        self.assertFalse(forest.is_empty())

        self.assertRaises(ValueError, forest.add, 5, data[5])
        return forest, data

    def test__H(self):
        """Check _H output consistent bytes length given
        the same concatenated hash value size.
        """
        mg = WeightedMinHashGenerator(100, sample_size=128)
        for l in range(2, mg.sample_size + 1, 16):
            m = mg.minhash(np.random.randint(1, 99999999, 100))
            forest = MinHashLSHForest(num_perm=128, l=l)
            forest.add("m", m)
            sizes = [len(H) for ht in forest.hashtables for H in ht]
            self.assertTrue(all(sizes[0] == s for s in sizes))

    def test_query(self):
        forest, data = self._setup()
        for key in data:
            results = forest.query(data[key], 10)
            self.assertIn(key, results)

    def test_pickle(self):
        forest, _ = self._setup()
        forest2 = pickle.loads(pickle.dumps(forest))
        self.assertEqual(forest.hashtables, forest2.hashtables)
        self.assertEqual(forest.keys, forest2.keys)
        self.assertEqual(forest.l, forest2.l)
        self.assertEqual(forest.k, forest2.k)
        self.assertEqual(forest.hashranges, forest2.hashranges)


if __name__ == "__main__":
    unittest.main()
