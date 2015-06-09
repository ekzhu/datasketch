import unittest
import struct
import pickle
from hashlib import sha1
from datasketch import minhash

class FakeHash(object):
    '''
    Implmenets the hexdigest required by HyperLogLog.
    '''

    def __init__(self, h):
        '''
        Initialize with an integer
        '''
        self.h = h

    def digest(self):
        '''
        Return the bytes representation of the integer
        '''
        return struct.pack('<Q', self.h)


class TestMinHash(unittest.TestCase):

    def test_init(self):
        m1 = minhash.MinHash(4, 1)
        m2 = minhash.MinHash(4, 1)
        for i in range(4):
            self.assertEqual(m1.hashvalues[i], m2.hashvalues[i])
            self.assertEqual((m1.permutations[i]), (m2.permutations[i]))

    def test_digest(self):
        m1 = minhash.MinHash(4, 1)
        m2 = minhash.MinHash(4, 1)
        m1.digest(FakeHash(12))
        for i in range(4):
            self.assertTrue(m1.hashvalues[i] < m2.hashvalues[i])

    def test_jaccard(self):
        m1 = minhash.MinHash(4, 1)
        m2 = minhash.MinHash(4, 1)
        self.assertTrue(minhash.jaccard(m1, m2) == 1.0)
        m2.digest(FakeHash(12))
        self.assertTrue(minhash.jaccard(m1, m2) == 0.0)
        m1.digest(FakeHash(13))
        self.assertTrue(minhash.jaccard(m1, m2) < 1.0)

    def test_merge(self):
        m1 = minhash.MinHash(4, 1)
        m2 = minhash.MinHash(4, 1)
        m2.digest(FakeHash(12))
        m1.merge(m2)
        self.assertTrue(minhash.jaccard(m1, m2) == 1.0)

    def test_union(self):
        m1 = minhash.MinHash(4, 1)
        m2 = minhash.MinHash(4, 1)
        m2.digest(FakeHash(12))
        u = minhash.MinHash.union(m1, m2)
        self.assertTrue(minhash.jaccard(u, m2) == 1.0)

    def test_bytesize(self):
        m1 = minhash.MinHash(4, 1)
        self.assertTrue(m1.bytesize() == (4*4)+4+8)

    def test_serialize(self):
        m1 = minhash.MinHash(2, 1)
        buf = bytearray(m1.bytesize())
        # Only test for syntax
        m1.serialize(buf)

    def test_deserialize(self):
        m1 = minhash.MinHash(10, 1)
        m1.digest(FakeHash(123))
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
        m1.digest(FakeHash(34))
        m1d.digest(FakeHash(34))
        self.assertTrue(all(hvd == hv for hv, hvd in zip(m1.hashvalues,
                m1d.hashvalues)))

    def test_pickle(self):
        m = minhash.MinHash(4, 1)
        m.digest(FakeHash(123))
        m.digest(FakeHash(45))
        p = pickle.loads(pickle.dumps(m))
        self.assertEqual(p.seed, m.seed)
        self.assertEqual(p.hashvalues, m.hashvalues)
        self.assertEqual(p.permutations, m.permutations)

    def test_eq(self):
        m1 = minhash.MinHash(4, 1)
        m2 = minhash.MinHash(4, 1)
        m3 = minhash.MinHash(4, 2)
        m4 = minhash.MinHash(8, 1)
        m5 = minhash.MinHash(4, 1)
        m1.digest(FakeHash(11))
        m2.digest(FakeHash(12))
        m3.digest(FakeHash(11))
        m4.digest(FakeHash(11))
        m5.digest(FakeHash(11))
        self.assertNotEqual(m1, m2)
        self.assertNotEqual(m1, m3)
        self.assertNotEqual(m1, m4)
        self.assertEqual(m1, m5)
        
        m1.digest(FakeHash(12))
        m2.digest(FakeHash(11))
        self.assertEqual(m1, m2)

    def test_count(self):
        m = minhash.MinHash()
        m.digest(FakeHash(11))
        m.digest(FakeHash(123))
        m.digest(FakeHash(92))
        m.digest(FakeHash(98))
        m.digest(FakeHash(123218))
        m.digest(FakeHash(32))
        c = m.count()
        print c

if __name__ == "__main__":
    unittest.main()
