import unittest
import pickle
from datasketch.lsh import MinHashLSH
from datasketch.minhash import MinHash
from pymemcache.client.base import Client

class TestMemcache(unittest.TestCase):
    def setUp(self):
        client = Client(("localhost", 11211))
        client.flush_all()
        client.close()

    def test_insert(self):
        lsh = MinHashLSH(
            threshold=0.5,
            num_perm=16,
            storage_config={
                'type': 'memcached',
                'basename': b'bp_matching_lsh',
                'memcached': {"host": "localhost",
                              "port" : 11211}
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
            self.assertTrue("a" in items)
            self.assertTrue("b" in items)
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
                'type': 'memcached',
                'basename': b'bp_matching_lsh',
                'memcached': {"host": "localhost",
                              "port": 11211}
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
