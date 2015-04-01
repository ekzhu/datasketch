import unittest
import struct
from hashlib import sha1
from datasketch import minhash

class TestMinHash(unittest.TestCase):

    def test_init(self):
        m1 = minhash.MinHash(1, 4)
        m2 = minhash.MinHash(1, 4)
        for i in range(4):
            self.assertEqual(m1.hashvalues[i], m2.hashvalues[i])
            self.assertEqual((m1.permutations[i])(i), (m2.permutations[i])(i))

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
        m2 = minhash.MinHash(1, 2)
        buf = bytearray(m1.bytesize()*2)
        m1.serialize(buf, 0)
        self.assertTrue(all(0 == i for i in buf[m1.bytesize():]))
        m2.serialize(buf, m1.bytesize())
        self.assertTrue(any(0 != i for i in buf[m1.bytesize():]))

    def test_deserialize(self):
        m1 = minhash.MinHash(1, 10)
        m2 = minhash.MinHash(1, 10)
        m1.digest(sha1(bytes(123)))
        m2.digest(sha1(bytes(45)))
        buf = bytearray(m1.bytesize()*2)
        m1.serialize(buf, 0)
        m2.serialize(buf, m1.bytesize())

        # Test if we get back the exact same MinHash objects after
        # deserializing from bytes
        m1d = minhash.MinHash.deserialize(buf, 0)
        m2d = minhash.MinHash.deserialize(buf, m1.bytesize())
        self.assertEqual(m1d.seed, m1.seed)
        self.assertEqual(m2d.seed, m2.seed)
        self.assertEqual(len(m1d.hashvalues), len(m1.hashvalues))
        self.assertEqual(len(m2d.hashvalues), len(m2.hashvalues))
        self.assertTrue(all(hvd == hv for hv, hvd in zip(m1.hashvalues,
                m1d.hashvalues)))
        self.assertTrue(all(hvd == hv for hv, hvd in zip(m2.hashvalues,
                m2d.hashvalues)))

        # Test if the permutation functions are the same
        m1.digest(sha1(bytes(34)))
        m1d.digest(sha1(bytes(34)))
        self.assertTrue(all(hvd == hv for hv, hvd in zip(m1.hashvalues,
                m1d.hashvalues)))


if __name__ == "__main__":
    unittest.main()
