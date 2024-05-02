import unittest
import pickle
from datasketch.lsh_bloom import BloomTable, MinHashLSHBloom
from datasketch.minhash import MinHash
import numpy as np
import os

class TestBloomTable(unittest.TestCase):
	def test_insert(self):
		r = 3
		sz = 32
		x = np.array([2,3,31], dtype=np.uint32)
		b = BloomTable(10, 0.01, num_arrays=r)
		b.insert(x)
		self.assertRaises(RuntimeError, b.insert, np.array([2,2], dtype=np.uint32))

	def test_query(self):
		r = 3
		sz = 32
		x = np.array([2,3,31], dtype=np.uint32)
		b = BloomTable(10, 0.01, num_arrays=r)
		b.insert(x)
		self.assertTrue(b.query(x))
		self.assertFalse(b.query(np.array([2,3,30], dtype=np.uint32)))
		self.assertRaises(RuntimeError, b.query, [2,2])

	def test_sync_in_memory(self):
		fname = "/tmp/bloomfilter-mem.bf"
		if os.path.exists(fname):
			os.remove(fname)
		r = 3
		sz = 32
		x = np.array([2,3,31], dtype=np.uint32)
		y = np.array([12,10,29], dtype=np.uint32)
		z = np.array([27,30,8], dtype=np.uint32)
		items = [x,y,z]
		b = BloomTable(10, 0.01, num_arrays=r, fname=fname, use_mmap=False)
		for item in items:
			b.insert(item)
		for item in items:
			self.assertTrue(b.query(item))
		b.sync()

		del b

		b_ = BloomTable(10, 0.01, num_arrays=r, fname=fname, use_mmap=False)
		for item in items:
			self.assertTrue(b_.query(item))

	def test_sync_mmap(self):
		fname = "/tmp/bloomfilter.bf"
		if os.path.exists(fname):
			os.remove(fname)
		r = 3
		sz = 32
		x = np.array([2,3,31], dtype=np.uint32)
		y = np.array([12,10,29], dtype=np.uint32)
		z = np.array([27,30,8], dtype=np.uint32)
		items = [x,y,z]
		b = BloomTable(10, 0.01, num_arrays=r, fname=fname, use_mmap=True)
		for item in items:
			b.insert(item)
		for item in items:
			self.assertTrue(b.query(item))
		b.sync()

		del b

		b_ = BloomTable(10, 0.01, num_arrays=r, fname=fname, use_mmap=True)
		for item in items:
			self.assertTrue(b_.query(item))


class TestMinHashLSHBloom(unittest.TestCase):

	def test_init(self):
		lsh = MinHashLSHBloom(threshold=0.8, num_bits=16, n=10, fp=0.01)
		b1, r1 = lsh.b, lsh.r
		lsh = MinHashLSHBloom(threshold=0.8, weights=(0.2,0.8), num_bits=16, n=10, fp=0.01)
		b2, r2 = lsh.b, lsh.r
		self.assertTrue(b1 < b2)
		self.assertTrue(r1 > r2)
		self.assertTrue(len(lsh.hashtables) == lsh.b)
		

	def test_insert(self):
		lsh = MinHashLSHBloom(threshold=0.5, num_perm=16, num_bits=16, n=10, fp=0.01)
		m1 = MinHash(16)
		m1.update("a".encode("utf8"))
		m2 = MinHash(16)
		m2.update("b".encode("utf8"))
		lsh.insert(m1)
		lsh.insert(m2)

		m3 = MinHash(18)
		self.assertRaises(ValueError, lsh.insert, m3)

	def test_query(self):
		lsh = MinHashLSHBloom(threshold=0.5, num_perm=16, num_bits=16, n=10, fp=0.01)
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

	def test_8bit(self):
		lsh = MinHashLSHBloom(threshold=0.5, num_perm=16, num_bits=8, n=10, fp=0.01)
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

	# def test_pickle(self):
	# 	lsh = MinHashLSHBloom(threshold=0.5, num_perm=16, num_bits=16, n=10, fp=0.01)
	# 	m1 = MinHash(16)
	# 	m1.update("a".encode("utf8"))
	# 	m2 = MinHash(16)
	# 	m2.update("b".encode("utf8"))
	# 	lsh.insert(m1)
	# 	lsh.insert(m2)
	# 	lsh2 = pickle.loads(pickle.dumps(lsh))

	# 	result = lsh2.query(m1)
	# 	self.assertTrue(result)
	# 	result = lsh2.query(m2)
	# 	self.assertTrue(result)


if __name__ == "__main__":
	unittest.main()
