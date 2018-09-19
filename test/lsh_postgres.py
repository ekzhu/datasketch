import unittest
import pickle
import numpy as np
from datasketch.lsh import MinHashLSH
from datasketch.minhash import MinHash
import psycopg2

class TestPostgres(unittest.TestCase):
    def setUp(self):
        with psycopg2.connect(dbname="bpmatching") as conn:
            with conn.cursor() as cur:
                cur.execute("TRUNCATE LSH_BUCKETS_UNORDERED;")
                cur.execute("TRUNCATE LSH_BUCKETS_ORDERED;")


    def test_insert(self):
        lsh = MinHashLSH(
            threshold=0.5,
            num_perm=16,
            storage_config={
                'type': 'postgres',
                'basename': b'bp_matching_lsh',
                'postgres': {"dbname": "bpmatching"}
            })
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
            self.assertTrue(pickle.dumps("a") in items)
            self.assertTrue(pickle.dumps("b") in items)
        self.assertTrue("a" in lsh)
        self.assertTrue("b" in lsh)
        for i, H in enumerate(lsh.keys[pickle.dumps("a")]):
            self.assertTrue(pickle.dumps("a") in lsh.hashtables[i][H])

        m3 = MinHash(18)
        self.assertRaises(ValueError, lsh.insert, "c", m3)

    def test_query(self):
        lsh = MinHashLSH(
            threshold=0.5,
            num_perm=16,
            storage_config={
                'type': 'postgres',
                'basename': b'bp_matching_lsh',
                'postgres': {"dbname": "bpmatching"}
            })
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

if __name__ == "__main__":
    unittest.main()
