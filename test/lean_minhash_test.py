import unittest
import struct
import pickle
import numpy as np
from datasketch import MinHash
from datasketch import LeanMinHash
from test.utils import fake_hash_func

class TestLeanMinHash(unittest.TestCase):

    def test_init(self):
        m1 = MinHash(4, 1, hashfunc=fake_hash_func)
        m2 = MinHash(4, 1, hashfunc=fake_hash_func)
        lm1 = LeanMinHash(m1)
        lm2 = LeanMinHash(m2)
        self.assertTrue(np.array_equal(lm1.hashvalues, lm2.hashvalues))
        self.assertTrue(np.array_equal(lm1.seed, lm2.seed))

        lm1 = LeanMinHash(seed=m1.seed, hashvalues=m1.hashvalues)
        lm2 = LeanMinHash(seed=m2.seed, hashvalues=m2.hashvalues)
        self.assertTrue(np.array_equal(lm1.hashvalues, lm2.hashvalues))
        self.assertTrue(np.array_equal(lm1.seed, lm2.seed))

    def test_is_empty(self):
        m = MinHash()
        lm = LeanMinHash(m)
        self.assertTrue(lm.is_empty())

    def test_update(self):
        m1 = MinHash(4, 1, hashfunc=fake_hash_func)
        try:
            lm1 = LeanMinHash(m1)
            lm1.update(12)
        except TypeError:
            pass
        else:
            raise Exception

    def test_jaccard(self):
        m1 = MinHash(4, 1, hashfunc=fake_hash_func)
        m2 = MinHash(4, 1, hashfunc=fake_hash_func)
        lm1 = LeanMinHash(m1)
        lm2 = LeanMinHash(m2)
        self.assertTrue(lm1.jaccard(lm2) == 1.0)
        m2.update(12)
        lm2 = LeanMinHash(m2)
        self.assertTrue(lm1.jaccard(lm2) == 0.0)
        m1.update(13)
        lm1 = LeanMinHash(m1)
        self.assertTrue(lm1.jaccard(lm2) < 1.0)

    def test_merge(self):
        m1 = MinHash(4, 1, hashfunc=fake_hash_func)
        m2 = MinHash(4, 1, hashfunc=fake_hash_func)
        m2.update(12)
        lm1 = LeanMinHash(m1)
        lm2 = LeanMinHash(m2)
        lm1.merge(lm2)
        self.assertTrue(lm1.jaccard(lm2) == 1.0)

    def test_union(self):
        m1 = MinHash(4, 1, hashfunc=fake_hash_func)
        m2 = MinHash(4, 1, hashfunc=fake_hash_func)
        m2.update(12)
        lm1 = LeanMinHash(m1)
        lm2 = LeanMinHash(m2)
        u = LeanMinHash.union(lm1, lm2)
        self.assertTrue(u.jaccard(lm2) == 1.0)

    def test_bytesize(self):
        m1 = MinHash(4, 1, hashfunc=fake_hash_func)
        lm1 = LeanMinHash(m1)
        self.assertTrue(lm1.bytesize() == (4*4)+4+8)

    def test_serialize(self):
        m1 = MinHash(2, 1, hashfunc=fake_hash_func)
        lm1 = LeanMinHash(m1)
        buf = bytearray(lm1.bytesize())
        # Only test for syntax
        lm1.serialize(buf)

        m2 = MinHash(2, 1, hashfunc=fake_hash_func)
        lm2 = LeanMinHash(m2)
        size = lm1.bytesize()
        buf = bytearray(size*2)
        lm1.serialize(buf)
        lm2.serialize(buf[size:])

    def test_deserialize(self):
        m1 = MinHash(10, 1, hashfunc=fake_hash_func)
        m1.update(123)
        lm1 = LeanMinHash(m1)
        buf = bytearray(lm1.bytesize())
        lm1.serialize(buf)

        # Test if we get back the exact same LeanMinHash objects after
        # deserializing from bytes
        lm1d = LeanMinHash.deserialize(buf)
        self.assertEqual(lm1d.seed, lm1.seed)
        self.assertEqual(len(lm1d.hashvalues), len(lm1.hashvalues))
        self.assertTrue(all(hvd == hv for hv, hvd in zip(lm1.hashvalues,
                lm1d.hashvalues)))

    def test_deserialize_byteorder(self):
        for byteorder in "@=<>!":
            m1 = MinHash(10, 1, hashfunc=fake_hash_func)
            m1.update(123)
            lm1 = LeanMinHash(m1)
            buf = bytearray(lm1.bytesize(byteorder))
            lm1.serialize(buf, byteorder)

            # Test if we get back the exact same LeanMinHash objects after
            # deserializing from bytes
            lm1d = LeanMinHash.deserialize(buf, byteorder)
            self.assertEqual(lm1d.seed, lm1.seed)
            self.assertEqual(len(lm1d.hashvalues), len(lm1.hashvalues))
            self.assertTrue(all(hvd == hv for hv, hvd in zip(lm1.hashvalues,
                    lm1d.hashvalues)))

    def test_pickle(self):
        m = MinHash(4, 1, hashfunc=fake_hash_func)
        m.update(123)
        m.update(45)
        lm = LeanMinHash(m)

        p = pickle.loads(pickle.dumps(lm))
        self.assertEqual(p.seed, lm.seed)
        self.assertTrue(np.array_equal(p.hashvalues, lm.hashvalues))

    def test_eq(self):
        m1 = MinHash(4, 1, hashfunc=fake_hash_func)
        m2 = MinHash(4, 1, hashfunc=fake_hash_func)
        m3 = MinHash(4, 2, hashfunc=fake_hash_func)
        m4 = MinHash(8, 1, hashfunc=fake_hash_func)
        m5 = MinHash(4, 1, hashfunc=fake_hash_func)
        m1.update(11)
        m2.update(12)
        m3.update(11)
        m4.update(11)
        m5.update(11)
        lm1 = LeanMinHash(m1)
        lm2 = LeanMinHash(m2)
        lm3 = LeanMinHash(m3)
        lm4 = LeanMinHash(m4)
        lm5 = LeanMinHash(m5)
        self.assertNotEqual(lm1, lm2)
        self.assertNotEqual(lm1, lm3)
        self.assertNotEqual(lm1, lm4)
        self.assertEqual(lm1, lm5)

        m1.update(12)
        m2.update(11)
        lm1 = LeanMinHash(m1)
        lm2 = LeanMinHash(m2)
        self.assertEqual(lm1, lm2)

    def test_count(self):
        m = MinHash(hashfunc=fake_hash_func)
        m.update(11)
        m.update(123)
        m.update(92)
        m.update(98)
        m.update(123218)
        m.update(32)
        lm = LeanMinHash(m)
        c = lm.count()
        self.assertGreaterEqual(c, 0)

    def test_hash(self):
        m = MinHash(hashfunc=fake_hash_func)
        m.update(11)
        m.update(123)
        m.update(92)
        m.update(98)
        m.update(123218)
        m.update(32)
        lm1 = LeanMinHash(m)
        lm2 = LeanMinHash(m)
        self.assertEqual(hash(lm1), hash(lm2))
        m.update(444)
        lm3 = LeanMinHash(m)
        self.assertNotEqual(hash(lm1), hash(lm3))
        d = dict()
        d[lm1] = True
        self.assertTrue(d[lm2])


if __name__ == "__main__":
    unittest.main()

