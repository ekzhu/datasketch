import unittest

import numpy as np

from datasketch.hnsw import HNSW


class TestHNSW(unittest.TestCase):
    def test_search_l2(self):
        data = np.random.rand(100, 10)
        hnsw = HNSW(
            distance_func=lambda x, y: np.linalg.norm(x - y),
            m=16,
            ef_construction=100,
        )
        for i in range(len(data)):
            hnsw.add(i, data[i])
        for i in range(len(data)):
            results = hnsw.search(data[i], 10)
            self.assertEqual(len(results), 10)

    def test_search_jaccard(self):
        data = np.random.randint(0, 100, (100, 10))
        jaccard_func = lambda x, y: (
            1.0
            - float(len(np.intersect1d(x, y, assume_unique=False)))
            / float(len(np.union1d(x, y)))
        )
        hnsw = HNSW(distance_func=jaccard_func, m=16, ef_construction=100)
        for i in range(len(data)):
            hnsw.add(i, data[i])
        for i in range(len(data)):
            results = hnsw.search(data[i], 10)
            self.assertEqual(len(results), 10)

    def test_update_point_l2(self):
        data = np.random.rand(100, 10)
        hnsw = HNSW(
            distance_func=lambda x, y: np.linalg.norm(x - y),
            m=16,
            ef_construction=100,
        )
        for i in range(len(data)):
            hnsw.add(i, data[i])
        new_data = np.random.rand(10, 10)
        for i in range(len(new_data)):
            hnsw.add(i, new_data[i])
        for i in range(len(data)):
            results = hnsw.search(data[i], 10)
            self.assertEqual(len(results), 10)

    def test_update_jaccard(self):
        data = np.random.randint(0, 100, (100, 10))
        jaccard_func = lambda x, y: (
            1.0
            - float(len(np.intersect1d(x, y, assume_unique=False)))
            / float(len(np.union1d(x, y)))
        )
        hnsw = HNSW(distance_func=jaccard_func, m=16, ef_construction=100)
        for i in range(len(data)):
            hnsw.add(i, data[i])
        for i in range(len(data)):
            results = hnsw.search(data[i], 10)
            self.assertEqual(len(results), 10)
