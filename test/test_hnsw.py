import unittest

import numpy as np

from datasketch.hnsw import HNSW


class TestHNSW(unittest.TestCase):
    def test_search(self):
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
