import unittest
from hashlib import sha1
import pickle
from datasketch.lsh import LSH
from datasketch.minhash import MinHash

class TestLSH(unittest.TestCase):
    
    def test_init(self):
        lsh = LSH(threshold=0.8)
        self.assertTrue(lsh.is_empty())
        b1, r1 = lsh.b, lsh.r
        lsh = LSH(threshold=0.8, weights=(0.2,0.8))
        b2, r2 = lsh.b, lsh.r
        self.assertTrue(b1 < b2)
        self.assertTrue(r1 > r2)

    def test_insert(self):
        lsh = LSH(threshold=0.5, num_perm=16)
        m1 = MinHash(16)
        m1.digest(sha1("a".encode("utf8")))
        m2 = MinHash(16)
        m2.digest(sha1("b".encode("utf8")))
        lsh.insert("a", m1)
        lsh.insert("b", m2)
        for t in lsh.hashtables:
            self.assertTrue(len(t) >= 1)
            items = []
            for H in t:
                items.extend(t[H])
            self.assertTrue("a" in items)
            self.assertTrue("b" in items)

    def test_query(self):
        lsh = LSH(threshold=0.5, num_perm=16)
        m1 = MinHash(16)
        m1.digest(sha1("a".encode("utf8")))
        m2 = MinHash(16)
        m2.digest(sha1("b".encode("utf8")))
        lsh.insert("a", m1)
        lsh.insert("b", m2)
        result = lsh.query(m1)
        self.assertTrue("a" in result)
        result = lsh.query(m2)
        self.assertTrue("b" in result)

    def test_pickle(self):
        lsh = LSH(threshold=0.5, num_perm=16)
        m1 = MinHash(16)
        m1.digest(sha1("a".encode("utf8")))
        m2 = MinHash(16)
        m2.digest(sha1("b".encode("utf8")))
        lsh.insert("a", m1)
        lsh.insert("b", m2)
        lsh2 = pickle.loads(pickle.dumps(lsh))
        result = lsh.query(m1)
        self.assertTrue("a" in result)
        result = lsh.query(m2)
        self.assertTrue("b" in result)

