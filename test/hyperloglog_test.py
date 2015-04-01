import unittest
import struct
import pickle
from hashlib import sha1
from datasketch.hyperloglog import HyperLogLog


class FakeHash(object):
    '''
    Implmenets the hexdigest required by HyperLogLog.
    '''

    def __init__(self, h):
        '''
        Initialize with an integer
        '''
        self.h = h

    def hexdigest(self):
        '''
        Return the hex string of the integer
        '''
        return hex(self.h)[2:]


class TestMinHash(unittest.TestCase):

    def test_init(self):
        h = HyperLogLog(4)
        self.assertEqual(h.m, 1 << 4)
        self.assertEqual(len(h.reg), h.m)
        self.assertTrue(all(0 == i for i in h.reg))

    def test_digest(self):
        h = HyperLogLog(4)
        h.digest(FakeHash(0b0001111))
        self.assertEqual(h.reg[0b1111], 64 - 4 + 1)
        h.digest(FakeHash(0xfffffffffffffff1))
        self.assertEqual(h.reg[1], 1)
        h.digest(FakeHash(0xfffffff5))
        self.assertEqual(h.reg[5], 33)

    def test_merge(self):
        h1 = HyperLogLog(4)
        h2 = HyperLogLog(4)
        h1.digest(FakeHash(0b0001111))
        h2.digest(FakeHash(0xfffffffffffffff1))
        h1.merge(h2)
        self.assertEqual(h1.reg[0b1111], 64 - 4 + 1)
        self.assertEqual(h1.reg[1], 1)

    def test_count(self):
        h = HyperLogLog(4)
        h.digest(FakeHash(0b0001111))
        h.digest(FakeHash(0xfffffffffffffff1))
        h.digest(FakeHash(0xfffffff5))
        # We can't really verify the correctness here, just to make sure
        # no syntax error
        # See benchmarks for the accuracy of the cardinality estimation.
        h.count()

    def test_serialize(self):
        h = HyperLogLog(4)
        buf = bytearray(h.bytesize())
        h.serialize(buf)
        self.assertEqual(h.p, struct.unpack_from('B', buf, 0)[0])

    def test_deserialize(self):
        h = HyperLogLog(4)
        h.digest(FakeHash(123))
        h.digest(FakeHash(33))
        h.digest(FakeHash(12))
        h.digest(FakeHash(0xfffffffffffffff1))
        buf = bytearray(h.bytesize())
        h.serialize(buf)
        hd = HyperLogLog.deserialize(buf)
        self.assertEqual(hd.p, h.p)
        self.assertEqual(hd.m, h.m)
        self.assertTrue(all(i == j for i, j in zip(h.reg, hd.reg)))

    def test_pickle(self):
        h = HyperLogLog(4)
        h.digest(FakeHash(123))
        h.digest(FakeHash(33))
        h.digest(FakeHash(12))
        h.digest(FakeHash(0xfffffffffffffff1))
        p = pickle.loads(pickle.dumps(h))
        self.assertEqual(p.m, h.m)
        self.assertEqual(p.p, h.p)
        self.assertEqual(p.reg, h.reg)


if __name__ == "__main__":
    unittest.main()
