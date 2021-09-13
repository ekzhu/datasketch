import unittest
import pickle
import numpy as np
import scipy as sp
import scipy.sparse
from datasketch.weighted_minhash import WeightedMinHashGenerator, WeightedMinHash

class TestWeightedMinHash(unittest.TestCase):

    def test_pickle(self):
        mg = WeightedMinHashGenerator(4, 10, 1)
        m = mg.minhash([1,2,3,4])
        p = pickle.loads(pickle.dumps(m))
        self.assertEqual(p.seed, m.seed)
        self.assertTrue(np.array_equal(p.hashvalues, m.hashvalues))

class TestWeightedMinHashGenerator(unittest.TestCase):
    def test_init(self):
        mg = WeightedMinHashGenerator(2, 4, 1)
        self.assertEqual(len(mg.rs), 4)
        self.assertEqual(len(mg.ln_cs), 4)
        self.assertEqual(len(mg.betas), 4)
        self.assertEqual(mg.seed, 1)
        self.assertEqual(mg.sample_size, 4)

    def test_minhash(self):
        mg = WeightedMinHashGenerator(2, 4, 1)
        m = mg.minhash([1,3])
        self.assertIsInstance(m, WeightedMinHash)
        self.assertEqual(len(m.hashvalues), 4)
        self.assertEqual(len(m), 4)
        self.assertTrue(m.hashvalues.dtype == int)

    def test_minhash_many_dense_onerow(self):
        mg = WeightedMinHashGenerator(2, 4, 1)
        m_list = mg.minhash_many(np.array([1,3]).reshape(1, 2))
        self.assertIsInstance(m_list, list)
        self.assertEqual(len(m_list), 1)
        self.assertIsInstance(m_list[0], WeightedMinHash)
        self.assertEqual(len(m_list[0].hashvalues), 4)
        self.assertEqual(len(m_list[0]), 4)
        self.assertTrue(m_list[0].hashvalues.dtype == int)

    def test_minhash_many_dense_tworows(self):
        mg = WeightedMinHashGenerator(2, 4, 1)
        m_list = mg.minhash_many(np.array([[1,3], [1, 3]]))
        self.assertIsInstance(m_list, list)
        self.assertEqual(len(m_list), 2)
        for m in m_list:
            self.assertIsInstance(m, WeightedMinHash)
            self.assertEqual(len(m.hashvalues), 4)
            self.assertEqual(len(m), 4)
            self.assertTrue(m.hashvalues.dtype == int)

    def test_minhash_many_dense_tworows_with_null(self):
        mg = WeightedMinHashGenerator(2, 4, 1)
        m_list = mg.minhash_many(np.array([[1,3], [0, 0]]))
        self.assertIsInstance(m_list, list)
        self.assertEqual(len(m_list), 2)

        m = m_list[0]
        self.assertIsInstance(m, WeightedMinHash)
        self.assertEqual(len(m.hashvalues), 4)
        self.assertEqual(len(m), 4)
        self.assertTrue(m.hashvalues.dtype == int)

        self.assertIs(m_list[1], None)

    def test_minhash_many_sparse_onerow(self):
        mg = WeightedMinHashGenerator(2, 4, 1)
        X = sp.sparse.csr_matrix(np.array([1,3]).reshape(1, 2))
        m_list = mg.minhash_many(X)
        self.assertIsInstance(m_list, list)
        self.assertEqual(len(m_list), 1)
        self.assertIsInstance(m_list[0], WeightedMinHash)
        self.assertEqual(len(m_list[0].hashvalues), 4)
        self.assertEqual(len(m_list[0]), 4)
        self.assertTrue(m_list[0].hashvalues.dtype == int)

    def test_minhash_many_sparse_tworows(self):
        mg = WeightedMinHashGenerator(2, 4, 1)
        X = sp.sparse.csr_matrix(np.array([[1,3], [1, 3]]))
        m_list = mg.minhash_many(X)
        self.assertIsInstance(m_list, list)
        self.assertEqual(len(m_list), 2)
        for m in m_list:
            self.assertIsInstance(m, WeightedMinHash)
            self.assertEqual(len(m.hashvalues), 4)
            self.assertEqual(len(m), 4)
            self.assertTrue(m.hashvalues.dtype == int)

    def test_minhash_many_sparse_tworows_with_null(self):
        mg = WeightedMinHashGenerator(2, 4, 1)
        X = sp.sparse.csr_matrix(np.array([[1,3], [0, 0]]))
        m_list = mg.minhash_many(X)
        self.assertIsInstance(m_list, list)
        self.assertEqual(len(m_list), 2)

        m = m_list[0]
        self.assertIsInstance(m, WeightedMinHash)
        self.assertEqual(len(m.hashvalues), 4)
        self.assertEqual(len(m), 4)
        self.assertTrue(m.hashvalues.dtype == int)

        self.assertIs(m_list[1], None)

if __name__ == "__main__":
    unittest.main()
