import pickle
import unittest

import numpy as np
import scipy as sp
import scipy.sparse

from datasketch.weighted_minhash import WeightedMinHash, WeightedMinHashGenerator


class TestWeightedMinHash(unittest.TestCase):
    def test_pickle(self):
        mg = WeightedMinHashGenerator(4, 10, 1)
        m = mg.minhash([1, 2, 3, 4])
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
        m = mg.minhash([1, 3])
        self.assertIsInstance(m, WeightedMinHash)
        self.assertEqual(len(m.hashvalues), 4)
        self.assertEqual(len(m), 4)
        self.assertTrue(m.hashvalues.dtype == int)

    def test_minhash_does_not_mutate_float32_input(self):
        mg = WeightedMinHashGenerator(3, 4, 1)
        v = np.array([1, 0, 3], dtype=np.float32)
        expected = v.copy()
        mg.minhash(v)
        np.testing.assert_array_equal(v, expected)

    def test_minhash_many_dense_onerow(self):
        mg = WeightedMinHashGenerator(2, 4, 1)
        m_list = mg.minhash_many(np.array([1, 3]).reshape(1, 2))
        self.assertIsInstance(m_list, list)
        self.assertEqual(len(m_list), 1)
        self.assertIsInstance(m_list[0], WeightedMinHash)
        self.assertEqual(len(m_list[0].hashvalues), 4)
        self.assertEqual(len(m_list[0]), 4)
        self.assertTrue(m_list[0].hashvalues.dtype == int)

    def test_minhash_many_dense_tworows(self):
        mg = WeightedMinHashGenerator(2, 4, 1)
        m_list = mg.minhash_many(np.array([[1, 3], [1, 3]]))
        self.assertIsInstance(m_list, list)
        self.assertEqual(len(m_list), 2)
        for m in m_list:
            self.assertIsInstance(m, WeightedMinHash)
            self.assertEqual(len(m.hashvalues), 4)
            self.assertEqual(len(m), 4)
            self.assertTrue(m.hashvalues.dtype == int)

    def test_minhash_many_dense_tworows_with_null(self):
        mg = WeightedMinHashGenerator(2, 4, 1)
        m_list = mg.minhash_many(np.array([[1, 3], [0, 0]]))
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
        X = sp.sparse.csr_matrix(np.array([1, 3]).reshape(1, 2))
        m_list = mg.minhash_many(X)
        self.assertIsInstance(m_list, list)
        self.assertEqual(len(m_list), 1)
        self.assertIsInstance(m_list[0], WeightedMinHash)
        self.assertEqual(len(m_list[0].hashvalues), 4)
        self.assertEqual(len(m_list[0]), 4)
        self.assertTrue(m_list[0].hashvalues.dtype == int)

    def test_minhash_many_sparse_tworows(self):
        mg = WeightedMinHashGenerator(2, 4, 1)
        X = sp.sparse.csr_matrix(np.array([[1, 3], [1, 3]]))
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
        X = sp.sparse.csr_matrix(np.array([[1, 3], [0, 0]]))
        m_list = mg.minhash_many(X)
        self.assertIsInstance(m_list, list)
        self.assertEqual(len(m_list), 2)

        m = m_list[0]
        self.assertIsInstance(m, WeightedMinHash)
        self.assertEqual(len(m.hashvalues), 4)
        self.assertEqual(len(m), 4)
        self.assertTrue(m.hashvalues.dtype == int)

        self.assertIs(m_list[1], None)

    def test_minhash_many_sparse_explicit_zeros(self):
        mg = WeightedMinHashGenerator(3, 32, 17)
        for empty_row in range(3):
            dense = np.array([[1, 0, 3], [0, 2, 0]], dtype=np.float32)
            dense = np.insert(dense, empty_row, 0, axis=0)
            expected = mg.minhash_many(dense)
            rows, cols = np.indices(dense.shape)
            stored = sp.sparse.coo_matrix((dense.ravel(), (rows.ravel(), cols.ravel())), shape=dense.shape)
            for fmt in ("csr", "csc", "coo"):
                with self.subTest(empty_row=empty_row, fmt=fmt):
                    matrix = stored.asformat(fmt)
                    original = matrix.copy()
                    actual = mg.minhash_many(matrix)
                    self.assertEqual(actual, expected)
                    self.assertIsNone(actual[empty_row])
                    self.assertEqual(matrix.nnz, original.nnz)
                    np.testing.assert_array_equal(matrix.data, original.data)
                    np.testing.assert_array_equal(matrix.toarray(), original.toarray())

    def test_minhash_many_empty_batches(self):
        mg = WeightedMinHashGenerator(3, 32, 17)
        for shape in ((2, 3), (0, 3)):
            dense = np.zeros(shape, dtype=np.float32)
            rows, cols = np.indices(shape)
            stored = sp.sparse.coo_matrix((dense.ravel(), (rows.ravel(), cols.ravel())), shape=shape)
            matrices = [dense, sp.sparse.csr_matrix(dense), stored.tocsr(), stored.tocsc(), stored]
            for matrix in matrices:
                with self.subTest(shape=shape, kind=type(matrix).__name__):
                    self.assertEqual(mg.minhash_many(matrix), [None] * shape[0])


if __name__ == "__main__":
    unittest.main()
