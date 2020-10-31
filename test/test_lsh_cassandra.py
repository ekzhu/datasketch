import unittest
import os
import numpy as np
import pickle
from multiprocessing import Pool, TimeoutError
import time
from datasketch.lsh import MinHashLSH
from datasketch.minhash import MinHash
from datasketch.weighted_minhash import WeightedMinHashGenerator

STORAGE_CONFIG_CASSANDRA = {
    'basename': b'test',
    'type': 'cassandra',
    'cassandra': {
        'seeds': ['127.0.0.1'],
        'keyspace': 'lsh_test',
        'replication': {
            'class': 'SimpleStrategy',
            'replication_factor': '1'
        },
        'drop_keyspace': True,
        'drop_tables': True,
    }
}
DO_TEST_CASSANDRA = os.environ.get("DO_TEST_CASSANDRA") == "true"

class TestMinHashLSHCassandra(unittest.TestCase):

    @unittest.skipIf(not DO_TEST_CASSANDRA, "Skipping test_cassandra__init")
    def test_cassandra__init(self):
        lsh = MinHashLSH(threshold=0.8, storage_config=STORAGE_CONFIG_CASSANDRA)
        self.assertTrue(lsh.is_empty())
        b1, r1 = lsh.b, lsh.r
        lsh = MinHashLSH(threshold=0.8, weights=(0.2,0.8))
        b2, r2 = lsh.b, lsh.r
        self.assertTrue(b1 < b2)
        self.assertTrue(r1 > r2)

    @unittest.skipIf(not DO_TEST_CASSANDRA, "Skipping test_cassandra__H")
    def test_cassandra__H(self):
        '''
        Check _H output consistent bytes length given
        the same concatenated hash value size
        '''
        for l in range(2, 128+1, 16):
            lsh = MinHashLSH(num_perm=128, storage_config=STORAGE_CONFIG_CASSANDRA)
            m = MinHash()
            m.update("abcdefg".encode("utf8"))
            m.update("1234567".encode("utf8"))
            lsh.insert("m", m)
            sizes = [len(H) for ht in lsh.hashtables for H in ht]
            self.assertTrue(all(sizes[0] == s for s in sizes))

    @unittest.skipIf(not DO_TEST_CASSANDRA, "Skipping test_cassandra__insert")
    def test_cassandra__insert(self):
        lsh = MinHashLSH(threshold=0.5, num_perm=16, storage_config=STORAGE_CONFIG_CASSANDRA)
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
        self.assertTrue("a" in lsh)
        self.assertTrue("b" in lsh)
        for i, H in enumerate(lsh.keys["a"]):
            self.assertTrue("a" in lsh.hashtables[i][H])

    @unittest.skipIf(not DO_TEST_CASSANDRA, "Skipping test_cassandra__query")
    def test_cassandra__query(self):
        lsh = MinHashLSH(threshold=0.5, num_perm=16, storage_config=STORAGE_CONFIG_CASSANDRA)
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

    @unittest.skipIf(not DO_TEST_CASSANDRA, "Skipping test_cassandra__remove")
    def test_cassandra__remove(self):
        lsh = MinHashLSH(threshold=0.5, num_perm=16, storage_config=STORAGE_CONFIG_CASSANDRA)
        m1 = MinHash(16)
        m1.update("a".encode("utf8"))
        m2 = MinHash(16)
        m2.update("b".encode("utf8"))
        lsh.insert("a", m1)
        lsh.insert("b", m2)

        lsh.remove("a")
        self.assertTrue("a" not in lsh.keys)
        for table in lsh.hashtables:
            for H in table:
                self.assertGreater(len(table[H]), 0)
                self.assertTrue("a" not in table[H])

        self.assertRaises(ValueError, lsh.remove, "c")

    @unittest.skipIf(not DO_TEST_CASSANDRA, "Skipping test_cassandra__get_subset_counts")
    def test_cassandra__get_subset_counts(self):
        m1 = MinHash(16)
        m1.update("a".encode("utf8"))
        m2 = MinHash(16)
        m2.update("b".encode("utf8"))

        lsh_c = MinHashLSH(threshold=0.5, num_perm=16, storage_config=STORAGE_CONFIG_CASSANDRA)
        lsh_c.insert("a", m1)
        lsh_c.insert("b", m2)
        lsh_m = MinHashLSH(threshold=0.5, num_perm=16)
        lsh_m.insert("a", m1)
        lsh_m.insert("b", m2)

        self.assertEquals(lsh_c.get_subset_counts("a"), lsh_m.get_subset_counts("a"))
        self.assertEquals(lsh_c.get_subset_counts("b"), lsh_m.get_subset_counts("b"))

    @unittest.skipIf(not DO_TEST_CASSANDRA, "Skipping test_cassandra__insertion_session")
    def test_cassandra__insertion_session(self):
        lsh = MinHashLSH(threshold=0.5, num_perm=16, storage_config=STORAGE_CONFIG_CASSANDRA)
        m1 = MinHash(16)
        m1.update("a".encode("utf8"))
        m2 = MinHash(16)
        m2.update("b".encode("utf8"))
        data = [("a", m1), ("b", m2)]
        with lsh.insertion_session() as session:
            for key, minhash in data:
                session.insert(key, minhash)
        for t in lsh.hashtables:
            self.assertTrue(len(t) >= 1)
            items = []
            for H in t:
                items.extend(t[H])
            self.assertTrue("a" in items)
            self.assertTrue("b" in items)
        self.assertTrue("a" in lsh)
        self.assertTrue("b" in lsh)
        for i, H in enumerate(lsh.keys["a"]):
            self.assertTrue("a" in lsh.hashtables[i][H])

    @unittest.skipIf(not DO_TEST_CASSANDRA, "Skipping test_cassandra__get_counts")
    def test_cassandra__get_counts(self):
        lsh = MinHashLSH(threshold=0.5, num_perm=16, storage_config=STORAGE_CONFIG_CASSANDRA)
        m1 = MinHash(16)
        m1.update("a".encode("utf8"))
        m2 = MinHash(16)
        m2.update("b".encode("utf8"))
        lsh.insert("a", m1)
        lsh.insert("b", m2)
        counts = lsh.get_counts()
        self.assertEqual(len(counts), lsh.b)
        for table in counts:
            self.assertEqual(sum(table.values()), 2)

    def multiprocess_test(self):
        lsh_file_path = "lsh.pkl"
        with open(lsh_file_path, "rb") as f:
            lsh = pickle.load(f)
        return True

    @unittest.skipIf(not DO_TEST_CASSANDRA, "Skipping test_cassandra__get_counts")
    def test_cassandra__multiprocess(self):
        lsh = MinHashLSH(threshold=0.5, num_perm=16, storage_config=STORAGE_CONFIG_CASSANDRA)
        lsh_file_path = "lsh.pkl"
        with open(lsh_file_path, "wb") as f:
            pickle.dump(lsh, f)

        pool = Pool(processes=4)
        async_results = [pool.apply_async(self.multiprocess_test, ()) for i in range(4)]
        completion_status = [False for _ in async_results]
        countdown = 4
        wait_time = 30
        start_time = time.time()
        while countdown > 0:
            elapsed_time = time.time() - start_time
            self.assertTrue(elapsed_time < wait_time)

            for i, async_result in enumerate(async_results):
                if completion_status[i]:
                    continue
                if async_result.ready():
                    completion_status[i] = True
                    countdown -= 1

class TestWeightedMinHashLSHCassandra(unittest.TestCase):

    @unittest.skipIf(not DO_TEST_CASSANDRA, "Skipping test_cassandra__init")
    def test_cassandra__init(self):
        lsh = MinHashLSH(threshold=0.8, storage_config=STORAGE_CONFIG_CASSANDRA)
        self.assertTrue(lsh.is_empty())
        b1, r1 = lsh.b, lsh.r
        lsh = MinHashLSH(threshold=0.8, weights=(0.2,0.8), storage_config=STORAGE_CONFIG_CASSANDRA)
        b2, r2 = lsh.b, lsh.r
        self.assertTrue(b1 < b2)
        self.assertTrue(r1 > r2)

    @unittest.skipIf(not DO_TEST_CASSANDRA, "Skipping test_cassandra__H")
    def test_cassandra__H(self):
        '''
        Check _H output consistent bytes length given
        the same concatenated hash value size
        '''
        mg = WeightedMinHashGenerator(100, sample_size=128)
        for l in range(2, mg.sample_size+1, 16):
            m = mg.minhash(np.random.randint(1, 99999999, 100))
            lsh = MinHashLSH(num_perm=128, storage_config=STORAGE_CONFIG_CASSANDRA)
            lsh.insert("m", m)
            sizes = [len(H) for ht in lsh.hashtables for H in ht]
            self.assertTrue(all(sizes[0] == s for s in sizes))

    @unittest.skipIf(not DO_TEST_CASSANDRA, "Skipping test_cassandra__insert")
    def test_cassandra__insert(self):
        lsh = MinHashLSH(threshold=0.5, num_perm=4, storage_config=STORAGE_CONFIG_CASSANDRA)
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
        self.assertTrue("a" in lsh)
        self.assertTrue("b" in lsh)
        for i, H in enumerate(lsh.keys["a"]):
            self.assertTrue("a" in lsh.hashtables[i][H])

        mg = WeightedMinHashGenerator(10, 5)
        m3 = mg.minhash(np.random.uniform(1, 10, 10))
        self.assertRaises(ValueError, lsh.insert, "c", m3)

    @unittest.skipIf(not DO_TEST_CASSANDRA, "Skipping test_cassandra__query")
    def test_cassandra__query(self):
        lsh = MinHashLSH(threshold=0.5, num_perm=4, storage_config=STORAGE_CONFIG_CASSANDRA)
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

    @unittest.skipIf(not DO_TEST_CASSANDRA, "Skipping test_cassandra__remove")
    def test_cassandra__remove(self):
        lsh = MinHashLSH(threshold=0.5, num_perm=4, storage_config=STORAGE_CONFIG_CASSANDRA)
        mg = WeightedMinHashGenerator(10, 4)
        m1 = mg.minhash(np.random.uniform(1, 10, 10))
        m2 = mg.minhash(np.random.uniform(1, 10, 10))
        lsh.insert("a", m1)
        lsh.insert("b", m2)

        lsh.remove("a")
        self.assertTrue("a" not in lsh.keys)
        for table in lsh.hashtables:
            for H in table:
                self.assertGreater(len(table[H]), 0)
                self.assertTrue("a" not in table[H])

        self.assertRaises(ValueError, lsh.remove, "c")


if __name__ == "__main__":
    unittest.main()
