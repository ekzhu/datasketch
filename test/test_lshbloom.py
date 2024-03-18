import unittest
from mock import patch
from datasketch.lsh_bloom import BitArray, BandedBitArray, MinHashLSHBloom
from datasketch.minhash import MinHash
from datasketch.hashfunc import sha1_hash16
import numpy as np

class TestBitArray(unittest.TestCase):
	def test_init(self):
		b = BitArray(32)
		self.assertEqual(len(b), 0)
		self.assertEqual(b.size(), 32)
		self.assertTrue(not b.array.any())
		
	def test_insert(self):
		b = BitArray(32)
		b.insert(2)
		b.insert(3)
		b.insert(31)
		self.assertTrue(b.array.any())
		self.assertEqual(b.array.count(1), 3)
		self.assertEqual(len(b), 3)

	def test_query(self):
		b = BitArray(32)
		b.insert(2)
		b.insert(3)
		b.insert(31)
		self.assertTrue(b.query(2))
		self.assertTrue(b.query(3))
		self.assertTrue(b.query(31))
		self.assertFalse(b.query(0))
		
class TestBandedBitArray(unittest.TestCase):
	def test_init(self):
		r = 16
		sz = 32
		b = BandedBitArray(array_size=sz, num_arrays=r)
		self.assertEqual(len(b), r)
		
	def test_insert(self):
		r = 3
		sz = 32
		b = BandedBitArray(array_size=sz, num_arrays=r)
		b.insert([2,3,31])
		self.assertRaises(RuntimeError, b.insert, [2,2])

	def test_query(self):
		r = 3
		sz = 32
		b = BandedBitArray(array_size=sz, num_arrays=r)
		b.insert([2,3,31])
		self.assertTrue(b.query([2,3,31]))
		self.assertFalse(b.query([2,3,30]))
		self.assertRaises(RuntimeError, b.query, [2,2])


class TestMinHashLSHBloom(unittest.TestCase):

	def test_init(self):
		lsh = MinHashLSHBloom(threshold=0.8)
		b1, r1 = lsh.b, lsh.r
		lsh = MinHashLSHBloom(threshold=0.8, weights=(0.2,0.8))
		b2, r2 = lsh.b, lsh.r
		self.assertTrue(b1 < b2)
		self.assertTrue(r1 > r2)
		self.assertTrue(len(lsh.hashtables) == lsh.b)
		

	def test_insert(self):
		lsh = MinHashLSHBloom(threshold=0.5, num_perm=16)
		m1 = MinHash(16)
		m1.update("a".encode("utf8"))
		m2 = MinHash(16)
		m2.update("b".encode("utf8"))
		lsh.insert(m1)
		lsh.insert(m2)
		for t in lsh.hashtables:
			self.assertTrue(len(t) == lsh.r)

		m3 = MinHash(18)
		self.assertRaises(ValueError, lsh.insert, m3)

	def test_query(self):
		lsh = MinHashLSHBloom(threshold=0.5, num_perm=16)
		m1 = MinHash(16)
		m1.update("a".encode("utf8"))
		m2 = MinHash(16)
		m2.update("b".encode("utf8"))
		lsh.insert(m1)
		lsh.insert(m2)
		result = lsh.query(m1)
		self.assertTrue(result)
		result = lsh.query(m2)
		self.assertTrue(result)

		m3 = MinHash(18)
		self.assertRaises(ValueError, lsh.query, m3)

	def test_16bit(self):
		lsh = MinHashLSHBloom(threshold=0.5, num_perm=16, hashrange=2**16)
		mersenne_prime = np.uint64((1 << 31) - 1)
		max_hash = np.uint64((1 << 16) - 1)
		hash_range = 1 << 16
		m1 = MinHash(16, hashfunc=sha1_hash16, mersenne_prime=mersenne_prime, hash_range=hash_range, max_hash=max_hash)
		m1.update("a".encode("utf8"))
		m2 = MinHash(16, hashfunc=sha1_hash16, mersenne_prime=mersenne_prime, hash_range=hash_range, max_hash=max_hash)
		m2.update("b".encode("utf8"))
		lsh.insert(m1)
		lsh.insert(m2)
		result = lsh.query(m1)
		self.assertTrue(result)
		result = lsh.query(m2)
		self.assertTrue(result)


if __name__ == "__main__":
	unittest.main()
