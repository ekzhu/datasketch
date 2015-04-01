import unittest
import struct
import pickle
from hashlib import sha1
from datasketch import minhash

class TestMinHash(unittest.TestCase):

    def test_init(self):
        m1 = minhash.MinHash(1, 4)
        m2 = minhash.MinHash(1, 4)
        for i in range(4):
            self.assertEqual(m1.hashvalues[i], m2.hashvalues[i])
            self.assertEqual((m1.permutations[i]), (m2.permutations[i]))

    def test_digest(self):
        m1 = minhash.MinHash(1, 4)
        m2 = minhash.MinHash(1, 4)
        m1.digest(sha1(bytes(12)))
        for i in range(4):
            self.assertTrue(m1.hashvalues[i] < m2.hashvalues[i])

    def test_jaccard(self):
        m1 = minhash.MinHash(1, 4)
        m2 = minhash.MinHash(1, 4)
        self.assertTrue(minhash.jaccard([m1, m2]) == 1.0)
        m2.digest(sha1(bytes(12)))
        self.assertTrue(minhash.jaccard([m1, m2]) == 0.0)
        m1.digest(sha1(bytes(13)))
        self.assertTrue(minhash.jaccard([m1, m2]) < 1.0)

    def test_merge(self):
        m1 = minhash.MinHash(1, 4)
        m2 = minhash.MinHash(1, 4)
        m2.digest(sha1(bytes(12)))
        m1.merge(m2)
        self.assertTrue(minhash.jaccard([m1, m2]) == 1.0)

    def test_bytesize(self):
        m1 = minhash.MinHash(1, 4)
        self.assertTrue(m1.bytesize() == (4*4)+4+8)

    def test_serialize(self):
        m1 = minhash.MinHash(1, 2)
        buf = bytearray(m1.bytesize())
        # Only test for syntax
        m1.serialize(buf)

    def test_deserialize(self):
        m1 = minhash.MinHash(1, 10)
        m1.digest(sha1(bytes(123)))
        buf = bytearray(m1.bytesize())
        m1.serialize(buf)

        # Test if we get back the exact same MinHash objects after
        # deserializing from bytes
        m1d = minhash.MinHash.deserialize(buf)
        self.assertEqual(m1d.seed, m1.seed)
        self.assertEqual(len(m1d.hashvalues), len(m1.hashvalues))
        self.assertTrue(all(hvd == hv for hv, hvd in zip(m1.hashvalues,
                m1d.hashvalues)))

        # Test if the permutation functions are the same
        m1.digest(sha1(bytes(34)))
        m1d.digest(sha1(bytes(34)))
        self.assertTrue(all(hvd == hv for hv, hvd in zip(m1.hashvalues,
                m1d.hashvalues)))

    def test_pickle(self):
        m = minhash.MinHash(1, 4)
        m.digest(sha1(bytes(123)))
        m.digest(sha1(bytes(45)))
        p = pickle.loads(pickle.dumps(m))
        self.assertEqual(p.seed, m.seed)
        self.assertEqual(p.hashvalues, m.hashvalues)
        self.assertEqual(p.permutations, m.permutations)

if __name__ == "__main__":
    unittest.main()
