import unittest

import numpy as np

from datasketch.hnsw import HNSW


def l2_distance(x, y):
    return np.linalg.norm(x - y)


class TestHNSW(unittest.TestCase):
    def test_search_l2(self):
        data = np.random.rand(100, 10)
        hnsw = HNSW(
            distance_func=lambda x, y: np.linalg.norm(x - y),
            m=16,
            ef_construction=100,
        )
        for i in range(len(data)):
            hnsw.insert(i, data[i])
            self.assertIn(i, hnsw)
            self.assertTrue(np.array_equal(hnsw[i], data[i]))
        for i in range(len(data)):
            results = hnsw.query(data[i], 10)
            self.assertEqual(len(results), 10)
            for j in range(len(results) - 1):
                self.assertLessEqual(
                    np.linalg.norm(hnsw[results[j][0]] - data[i]),
                    np.linalg.norm(hnsw[results[j + 1][0]] - data[i]),
                )

    def test_search_jaccard(self):
        data = np.random.randint(0, 100, (100, 10))
        jaccard_func = lambda x, y: (
            1.0
            - float(len(np.intersect1d(x, y, assume_unique=False)))
            / float(len(np.union1d(x, y)))
        )
        hnsw = HNSW(distance_func=jaccard_func, m=16, ef_construction=100)
        for i in range(len(data)):
            hnsw.insert(i, data[i])
            self.assertIn(i, hnsw)
            self.assertTrue(np.array_equal(hnsw[i], data[i]))
        for i in range(len(data)):
            results = hnsw.query(data[i], 10)
            self.assertEqual(len(results), 10)
            for j in range(len(results) - 1):
                self.assertLessEqual(
                    jaccard_func(hnsw[results[j][0]], data[i]),
                    jaccard_func(hnsw[results[j + 1][0]], data[i]),
                )

    def test_update_point_l2(self):
        data = np.random.rand(100, 10)
        hnsw = HNSW(
            distance_func=lambda x, y: np.linalg.norm(x - y),
            m=16,
            ef_construction=100,
        )
        for i in range(len(data)):
            hnsw.insert(i, data[i])
        new_data = np.random.rand(10, 10)
        for i in range(len(new_data)):
            hnsw.insert(i, new_data[i])
            self.assertTrue(np.array_equal(hnsw[i], new_data[i]))
        for i in range(len(data)):
            results = hnsw.query(data[i], 10)
            self.assertEqual(len(results), 10)
            for j in range(len(results) - 1):
                self.assertLessEqual(
                    np.linalg.norm(hnsw[results[j][0]] - data[i]),
                    np.linalg.norm(hnsw[results[j + 1][0]] - data[i]),
                )

    def test_update_jaccard(self):
        data = np.random.randint(0, 100, (100, 10))
        jaccard_func = lambda x, y: (
            1.0
            - float(len(np.intersect1d(x, y, assume_unique=False)))
            / float(len(np.union1d(x, y)))
        )
        hnsw = HNSW(distance_func=jaccard_func, m=16, ef_construction=100)
        for i in range(len(data)):
            hnsw.insert(i, data[i])
        new_data = np.random.randint(0, 100, (10, 10))
        for i in range(len(new_data)):
            hnsw.insert(i, new_data[i])
            self.assertTrue(np.array_equal(hnsw[i], new_data[i]))
        for i in range(len(data)):
            results = hnsw.query(data[i], 10)
            self.assertEqual(len(results), 10)
            for j in range(len(results) - 1):
                self.assertLessEqual(
                    jaccard_func(hnsw[results[j][0]], data[i]),
                    jaccard_func(hnsw[results[j + 1][0]], data[i]),
                )

    def test_pickle(self):
        data = np.random.rand(100, 10)
        hnsw = HNSW(
            distance_func=l2_distance,
            m=16,
            ef_construction=100,
        )
        for i in range(len(data)):
            hnsw.insert(i, data[i])

        import pickle

        hnsw2 = pickle.loads(pickle.dumps(hnsw))

        for i in range(len(data)):
            results1 = hnsw.query(data[i], 10)
            results2 = hnsw2.query(data[i], 10)
            self.assertEqual(results1, results2)
