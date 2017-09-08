import unittest
import pickle
import numpy as np
from datasketch.lshensemble import MinHashLSHEnsemble
from datasketch.minhash import MinHash


class TestMinHashLSHEnsemble(unittest.TestCase):

    def test_init(self):
        lsh = MinHashLSHEnsemble(threshold=0.8)
        self.assertTrue(lsh.is_empty())

    def _data(self, count):
        sizes = np.random.randint(1, 100, count)
        for key, size in enumerate(sizes):
            m = MinHash()
            for i in range(size):
                m.update(("%d" % i).encode("utf8"))
            yield (key, m, size) 

    def test_index(self):
        lsh = MinHashLSHEnsemble(threshold=0.8)
        lsh.index(self._data(160))
        self.assertFalse(lsh.is_empty())
        self.assertTrue(41 in lsh)

    def test_query(self):
        lsh = MinHashLSHEnsemble(threshold=0.9)
        data = list(self._data(64))
        lsh.index(data)
        for key, minhash, size in data:
            keys = lsh.query(minhash, size)
            self.assertTrue(key in keys)

    def test_pickle(self):
        lsh = MinHashLSHEnsemble(threshold=0.9)
        data = list(self._data(32))
        lsh.index(data)
        buf = pickle.dumps(lsh)
        lsh2 = pickle.loads(buf)
        for key, minhash, size in data:
            keys1 = lsh.query(minhash, size)
            keys2 = lsh2.query(minhash, size)
            self.assertTrue(set(keys1) == set(keys2))
