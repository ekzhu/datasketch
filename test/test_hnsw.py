import unittest

import numpy as np

from datasketch.hnsw import HNSW


def l2_distance(x, y):
    return np.linalg.norm(x - y)


def jaccard_distance(x, y):
    return 1.0 - float(len(np.intersect1d(x, y, assume_unique=False))) / float(
        len(np.union1d(x, y))
    )


class TestHNSW(unittest.TestCase):
    def _create_random_points(self, n=100, dim=10):
        return np.random.rand(n, dim)

    def _create_index(self, vecs, keys=None):
        hnsw = HNSW(
            distance_func=l2_distance,
            m=16,
            ef_construction=100,
        )
        self._insert_points(hnsw, vecs, keys)
        return hnsw

    def _search_index(self, index, queries, k=10):
        return self._search_index_dist(index, queries, l2_distance, k)

    def _insert_points(self, index, points, keys=None):
        original_length = len(index)

        if keys is None:
            keys = list(range(len(points)))

        for key, point in zip(keys, points):
            # Test insert.
            if np.random.random_sample() < 0.5:
                index.insert(key, point)
            else:
                index[key] = point
            # Test contains.
            self.assertIn(key, index)
            if original_length == 0:
                self.assertNotIn(key + 1, index)
            # Test get.
            self.assertTrue(np.array_equal(index.get(key), point))
            self.assertTrue(np.array_equal(index[key], point))

        if original_length == 0:
            # Test length.
            self.assertEqual(len(index), len(points))

            # Test order.
            for key_indexed, key in zip(index, keys):
                self.assertEqual(key_indexed, key)
            for key_indexed, key in zip(index.keys(), keys):
                self.assertEqual(key_indexed, key)
            for vec_indexed, vec in zip(index.values(), points):
                self.assertTrue(np.array_equal(vec_indexed, vec))
            for (key_indexed, vec_indexed), key, vec in zip(
                index.items(), keys, points
            ):
                self.assertEqual(key_indexed, key)
                self.assertTrue(np.array_equal(vec_indexed, vec))

    def _search_index_dist(self, index, queries, distance_func, k=10):
        for i in range(len(queries)):
            results = index.query(queries[i], 10)
            self.assertEqual(len(results), 10)
            for j in range(len(results) - 1):
                self.assertLessEqual(
                    distance_func(index[results[j][0]], queries[i]),
                    distance_func(index[results[j + 1][0]], queries[i]),
                )

    def test_search(self):
        data = self._create_random_points()
        hnsw = self._create_index(data)
        self._search_index(hnsw, data)

    def test_upsert(self):
        data = self._create_random_points()
        hnsw = self._create_index(data)
        new_data = self._create_random_points(n=10, dim=10)
        self._insert_points(hnsw, new_data)
        self._search_index(hnsw, new_data)

    def test_update(self):
        data = self._create_random_points()
        hnsw = self._create_index(data)
        new_data = self._create_random_points(n=10, dim=10)
        hnsw.update({i: new_data[i] for i in range(len(new_data))})
        self._search_index(hnsw, new_data)

    def test_merge(self):
        data1 = self._create_random_points()
        data2 = self._create_random_points()
        hnsw1 = self._create_index(data1, keys=list(range(len(data1))))
        hnsw2 = self._create_index(
            data2, keys=list(range(len(data1), len(data1) + len(data2)))
        )
        new_index = hnsw1.merge(hnsw2)
        self._search_index(new_index, data1)
        self._search_index(new_index, data2)
        for i in range(len(data1)):
            self.assertIn(i, new_index)
            self.assertTrue(np.array_equal(new_index[i], data1[i]))
        for i in range(len(data2)):
            self.assertIn(i + len(data1), new_index)
            self.assertTrue(np.array_equal(new_index[i + len(data1)], data2[i]))

    def test_pickle(self):
        data = self._create_random_points()
        hnsw = self._create_index(data)
        import pickle

        hnsw2 = pickle.loads(pickle.dumps(hnsw))
        self.assertEqual(hnsw, hnsw2)

    def test_copy(self):
        data = self._create_random_points()
        hnsw = self._create_index(data)
        hnsw2 = hnsw.copy()
        self.assertEqual(hnsw, hnsw2)


class TestHNSWJaccard(TestHNSW):
    def _create_random_points(self, high=50, n=100, dim=10):
        return np.random.randint(0, high, (n, dim))

    def _create_index(self, sets, keys=None):
        hnsw = HNSW(
            distance_func=jaccard_distance,
            m=16,
            ef_construction=100,
        )
        self._insert_points(hnsw, sets, keys)
        return hnsw

    def _search_index(self, index, queries, k=10):
        return super()._search_index_dist(index, queries, jaccard_distance, k)
