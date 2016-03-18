import unittest
import struct
import pickle
from hashlib import sha1
from mock import patch
import numpy as np
from datasketch.hyperloglog import HyperLogLog, HyperLogLogPlusPlus


class FakeHash(object):
    '''
    Implmenets the hexupdate required by HyperLogLog.
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


class TestHyperLogLog(unittest.TestCase):

    _class = HyperLogLog

    def test_init(self):
        h = self._class(4, hashobj=FakeHash)
        self.assertEqual(h.m, 1 << 4)
        self.assertEqual(len(h.reg), h.m)
        self.assertTrue(all(0 == i for i in h.reg))

    def test_init_from_reg(self):
        reg = np.array([1 for _ in range(1 << 4)], dtype=np.int8)
        h = self._class(reg=reg, hashobj=FakeHash)
        self.assertEqual(h.p, 4)
        h2 = self._class(p=4, hashobj=FakeHash)
        self.assertEqual(h.p, h2.p)

    def test_is_empty(self):
        h = self._class()
        self.assertTrue(h.is_empty())

    def test_update(self):
        h = self._class(4, hashobj=FakeHash)
        h.update(0b00011111)
        self.assertEqual(h.reg[0b1111], self._class._hash_range_bit - 4)
        h.update(0xfffffff1)
        self.assertEqual(h.reg[1], 1)
        h.update(0x000000f5)
        self.assertEqual(h.reg[5], self._class._hash_range_bit - 4 - 3)

    def test_merge(self):
        h1 = self._class(4, hashobj=FakeHash)
        h2 = self._class(4, hashobj=FakeHash)
        h1.update(0b00011111)
        h2.update(0xfffffff1)
        h1.merge(h2)
        self.assertEqual(h1.reg[0b1111], self._class._hash_range_bit - 4)
        self.assertEqual(h1.reg[1], 1)

    def test_count(self):
        h = self._class(4, hashobj=FakeHash)
        h.update(0b00011111)
        h.update(0xfffffff1)
        h.update(0xfffffff5)
        # We can't really verify the correctness here, just to make sure
        # no syntax error
        # See benchmarks for the accuracy of the cardinality estimation.
        h.count()

    def test_serialize(self):
        h = self._class(4, hashobj=FakeHash)
        buf = bytearray(h.bytesize())
        h.serialize(buf)
        self.assertEqual(h.p, struct.unpack_from('B', bytes(buf), 0)[0])

    def test_deserialize(self):
        h = self._class(4, hashobj=FakeHash)
        h.update(123)
        h.update(33)
        h.update(12)
        h.update(0xfffffff1)
        buf = bytearray(h.bytesize())
        h.serialize(buf)
        hd = self._class.deserialize(buf)
        self.assertEqual(hd.p, h.p)
        self.assertEqual(hd.m, h.m)
        self.assertTrue(all(i == j for i, j in zip(h.reg, hd.reg)))

    def test_pickle(self):
        h = self._class(4, hashobj=FakeHash)
        h.update(123)
        h.update(33)
        h.update(12)
        h.update(0xffffff1)
        p = pickle.loads(pickle.dumps(h))
        self.assertEqual(p.m, h.m)
        self.assertEqual(p.p, h.p)
        self.assertTrue(np.array_equal(p.reg, h.reg))

    def test_union(self):
        h1 = self._class(4, hashobj=FakeHash)
        h2 = self._class(4, hashobj=FakeHash)
        h3 = self._class(4, hashobj=FakeHash)
        h1.update(0b00011111)
        h2.update(0xfffffff1)
        h3.update(0x000000f5)
        h = self._class.union(h1, h2, h3)
        self.assertEqual(h.reg[0b1111], self._class._hash_range_bit - 4)
        self.assertEqual(h.reg[1], 1)
        self.assertEqual(h.reg[5], self._class._hash_range_bit - 4 - 3)

    def test_eq(self):
        h1 = self._class(4, hashobj=FakeHash)
        h2 = self._class(4, hashobj=FakeHash)
        h3 = self._class(4, hashobj=FakeHash)
        h4 = self._class(8, hashobj=FakeHash)
        h1.update(0b00011111)
        h2.update(0xfffffff1)
        h3.update(0b00011111)
        h4.update(0b00011111)
        self.assertNotEqual(h1, h2)
        self.assertEqual(h1, h3)
        self.assertNotEqual(h1, h4)


class TestHyperLogLogSpecific(unittest.TestCase):

    def test_hyperloglog_large_card_est(self):
        reg = np.array([27 for i in range(1 << 4)], dtype=np.int8)
        with patch.object(HyperLogLog, '_largerange_correction') as mock_method:
            mock_method.return_value = 0
            h = HyperLogLog(reg=reg)
            h.count()
        self.assertTrue(mock_method.called)

    def test_hyperloglog_small_card_est(self):
        reg = np.array([1 for i in range(1 << 4)], dtype=np.int8)
        with patch.object(HyperLogLog, '_linearcounting') as mock_method:
            mock_method.return_value = 0
            h = HyperLogLog(reg=reg)
            h.count()
        self.assertTrue(mock_method.called)


class TestHyperLogLogPlusPlus(TestHyperLogLog):

    _class = HyperLogLogPlusPlus

    def test_update(self):
        h = self._class(4, hashobj=FakeHash)
        h.update(0b00011111)
        self.assertEqual(h.reg[0b1111], self._class._hash_range_bit - 4)
        h.update(0xfffffffffffffff1)
        self.assertEqual(h.reg[1], 1)
        h.update(0x000000f5)
        self.assertEqual(h.reg[5], self._class._hash_range_bit - 4 - 3)

    def test_merge(self):
        h1 = self._class(4, hashobj=FakeHash)
        h2 = self._class(4, hashobj=FakeHash)
        h1.update(0b00011111)
        h2.update(0xfffffffffffffff1)
        h1.merge(h2)
        self.assertEqual(h1.reg[0b1111], self._class._hash_range_bit - 4)
        self.assertEqual(h1.reg[1], 1)

    def test_union(self):
        h1 = self._class(4, hashobj=FakeHash)
        h2 = self._class(4, hashobj=FakeHash)
        h3 = self._class(4, hashobj=FakeHash)
        h1.update(0b00011111)
        h2.update(0xfffffffffffffff1)
        h3.update(0x000000f5)
        h = self._class.union(h1, h2, h3)
        self.assertEqual(h.reg[0b1111], self._class._hash_range_bit - 4)
        self.assertEqual(h.reg[1], 1)
        self.assertEqual(h.reg[5], self._class._hash_range_bit - 4 - 3)


if __name__ == "__main__":
    unittest.main()
