import unittest
import struct
import pickle
from hashlib import sha1
import numpy as np
from datasketch import minhash
from datasketch.b_bit_minhash import bBitMinHash

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
        m1 = minhash.MinHash(4, 1, hashobj=FakeHash)
        m2 = minhash.MinHash(4, 1, hashobj=FakeHash)
        self.assertTrue(np.array_equal(m1.hashvalues, m2.hashvalues))
        self.assertTrue(np.array_equal(m1.permutations, m2.permutations))

    def test_is_empty(self):
        m = minhash.MinHash()
        self.assertTrue(m.is_empty())

    def test_update(self):
        m1 = minhash.MinHash(4, 1, hashobj=FakeHash)
        m2 = minhash.MinHash(4, 1, hashobj=FakeHash)
        m1.update(12)
        for i in range(4):
            self.assertTrue(m1.hashvalues[i] < m2.hashvalues[i])

    def test_jaccard(self):
        m1 = minhash.MinHash(4, 1, hashobj=FakeHash)
        m2 = minhash.MinHash(4, 1, hashobj=FakeHash)
        self.assertTrue(m1.jaccard(m2) == 1.0)
        m2.update(12)
        self.assertTrue(m1.jaccard(m2) == 0.0)
        m1.update(13)
        self.assertTrue(m1.jaccard(m2) < 1.0)

    def test_merge(self):
        m1 = minhash.MinHash(4, 1, hashobj=FakeHash)
        m2 = minhash.MinHash(4, 1, hashobj=FakeHash)
        m2.update(12)
        m1.merge(m2)
        self.assertTrue(m1.jaccard(m2) == 1.0)

    def test_union(self):
        m1 = minhash.MinHash(4, 1, hashobj=FakeHash)
        m2 = minhash.MinHash(4, 1, hashobj=FakeHash)
        m2.update(12)
        u = minhash.MinHash.union(m1, m2)
        self.assertTrue(u.jaccard(m2) == 1.0)

    def test_pickle(self):
        m = minhash.MinHash(4, 1, hashobj=FakeHash)
        m.update(123)
        m.update(45)
        p = pickle.loads(pickle.dumps(m))
        self.assertEqual(p.seed, m.seed)
        self.assertTrue(np.array_equal(p.hashvalues, m.hashvalues))
        self.assertTrue(np.array_equal(p.permutations, m.permutations))

    def test_eq(self):
        m1 = minhash.MinHash(4, 1, hashobj=FakeHash)
        m2 = minhash.MinHash(4, 1, hashobj=FakeHash)
        m3 = minhash.MinHash(4, 2, hashobj=FakeHash)
        m4 = minhash.MinHash(8, 1, hashobj=FakeHash)
        m5 = minhash.MinHash(4, 1, hashobj=FakeHash)
        m1.update(11)
        m2.update(12)
        m3.update(11)
        m4.update(11)
        m5.update(11)
        self.assertNotEqual(m1, m2)
        self.assertNotEqual(m1, m3)
        self.assertNotEqual(m1, m4)
        self.assertEqual(m1, m5)

        m1.update(12)
        m2.update(11)
        self.assertEqual(m1, m2)

    def test_count(self):
        m = minhash.MinHash(hashobj=FakeHash)
        m.update(11)
        m.update(123)
        m.update(92)
        m.update(98)
        m.update(123218)
        m.update(32)
        c = m.count()
        self.assertGreaterEqual(c, 0)

    def test_byte_tokens(self):
        m = minhash.MinHash(4, 1)
        m.update(b'Hello')
        self.assertListEqual(
            m.hashvalues.tolist(),
            [4159347281, 889859675, 2216179651, 3113718761],
        )

class TestbBitMinHash(unittest.TestCase):

    def setUp(self):
        self.m = minhash.MinHash(hashobj=FakeHash)
        self.m.update(11)
        self.m.update(123)
        self.m.update(92)
        self.m.update(98)
        self.m.update(123218)
        self.m.update(32)

    def test_init(self):
        bm = bBitMinHash(self.m, 1)
        bm = bBitMinHash(self.m, 2)
        bm = bBitMinHash(self.m, 3)
        bm = bBitMinHash(self.m, 4)
        bm = bBitMinHash(self.m, 5)
        bm = bBitMinHash(self.m, 8)
        bm = bBitMinHash(self.m, 12)
        bm = bBitMinHash(self.m, 16)
        bm = bBitMinHash(self.m, 27)
        bm = bBitMinHash(self.m, 32)

    def test_eq(self):
        m1 = minhash.MinHash(4, 1, hashobj=FakeHash)
        m2 = minhash.MinHash(4, 1, hashobj=FakeHash)
        m3 = minhash.MinHash(4, 2, hashobj=FakeHash)
        m4 = minhash.MinHash(8, 1, hashobj=FakeHash)
        m5 = minhash.MinHash(4, 1, hashobj=FakeHash)
        m1.update(11)
        m2.update(12)
        m3.update(11)
        m4.update(11)
        m5.update(11)
        m1 = bBitMinHash(m1)
        m2 = bBitMinHash(m2)
        m3 = bBitMinHash(m3)
        m4 = bBitMinHash(m4)
        m5 = bBitMinHash(m5)
        self.assertNotEqual(m1, m2)
        self.assertNotEqual(m1, m3)
        self.assertNotEqual(m1, m4)
        self.assertEqual(m1, m5)

    def test_jaccard(self):
        m1 = minhash.MinHash(4, 1, hashobj=FakeHash)
        m2 = minhash.MinHash(4, 1, hashobj=FakeHash)
        bm1 = bBitMinHash(m1)
        bm2 = bBitMinHash(m2)
        self.assertTrue(bm1.jaccard(bm2) == 1.0)

        m2.update(12)
        bm2 = bBitMinHash(m2)
        self.assertTrue(bm1.jaccard(bm2) < 1.0)

        m1.update(13)
        bm1 = bBitMinHash(m1)
        self.assertTrue(bm1.jaccard(bm2) < 1.0)

    def test_bytesize(self):
        s = bBitMinHash(self.m).bytesize()
        self.assertGreaterEqual(s, 8*2+4+1+self.m.hashvalues.size/64)

    def test_pickle(self):
        for num_perm in [1 << i for i in range(4, 10)]:
            m = minhash.MinHash(num_perm=num_perm, hashobj=FakeHash)
            m.update(11)
            m.update(123)
            m.update(92)
            m.update(98)
            m.update(123218)
            m.update(32)
            for b in [1, 2, 3, 9, 27, 32]:
                bm = bBitMinHash(m, b)
                bm2 = pickle.loads(pickle.dumps(bm))
                self.assertEqual(bm, bm2)


if __name__ == "__main__":
    unittest.main()
