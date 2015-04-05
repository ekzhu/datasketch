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


class TestHyperLogLog(unittest.TestCase):

    def test_init(self):
        h = HyperLogLog(4)
        self.assertEqual(h.m, 1 << 4)
        self.assertEqual(len(h.reg), h.m)
        self.assertTrue(all(0 == i for i in h.reg))

    def test_init_from_reg(self):
        reg = [1 for _ in range(1 << 4)]
        h = HyperLogLog(reg=reg)
        self.assertEqual(h.p, 4)
        h2 = HyperLogLog(p=4)
        self.assertEqual(h.p, h2.p)

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

    def test_union_count(self):
        h1 = HyperLogLog(4)
        h1.digest(FakeHash(0b0001111))
        h1.digest(FakeHash(0xfffffffffffffff1))
        h1.digest(FakeHash(0xfffffff5))
        h2 = HyperLogLog(4)
        self.assertEqual(h1.count(), h1.union_count(h2))

        h2.digest(FakeHash(0b0001111))
        h2.digest(FakeHash(0xfffffffffffffff1))
        h2.digest(FakeHash(0xfffffff5))
        self.assertEqual(h1.count(), h1.union_count(h2))

        h2.digest(FakeHash(0xfffffff6))
        self.assertNotEqual(h1.count(), h1.union_count(h2))

    def test_intersection_count(self):
        h1 = HyperLogLog(4)
        h1.digest(FakeHash(0b0001111))
        h1.digest(FakeHash(0xfffffffffffffff1))
        h1.digest(FakeHash(0xfffffff5))
        h2 = HyperLogLog(4)
        self.assertEqual(h1.intersection_count(h2), 0)

        h2.digest(FakeHash(0b0001111))
        h2.digest(FakeHash(0xfffffffffffffff1))
        h2.digest(FakeHash(0xfffffff5))
        self.assertEqual(int(h1.intersection_count(h2)), 3)

    def test_jaccard(self):
        h1 = HyperLogLog(4)
        h1.digest(FakeHash(0b0001111))
        h1.digest(FakeHash(0xfffffffffffffff1))
        h1.digest(FakeHash(0xfffffff5))
        h2 = HyperLogLog(4)
        self.assertEqual(h1.jaccard(h2), 0)

        h2.digest(FakeHash(0b0001111))
        h2.digest(FakeHash(0xfffffffffffffff1))
        h2.digest(FakeHash(0xfffffff5))
        self.assertEqual(int(h1.jaccard(h2)), 1)

        h2.digest(FakeHash(0xfffffff6))
        self.assertNotEqual(h1.jaccard(h2), 1)

    def test_inclusion(self):
        h1 = HyperLogLog(4)
        h1.digest(FakeHash(0b0001111))
        h1.digest(FakeHash(0xfffffffffffffff1))
        h1.digest(FakeHash(0xfffffff5))
        h2 = HyperLogLog(4)
        self.assertEqual(h1.inclusion(h2), 0)

        h2.digest(FakeHash(0b0001111))
        h2.digest(FakeHash(0xfffffffffffffff1))
        h2.digest(FakeHash(0xfffffff5))
        self.assertEqual(int(h1.inclusion(h2)), 1)

        h2.digest(FakeHash(0xfffffff6))
        self.assertEqual(int(h1.inclusion(h2)), 1)

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
