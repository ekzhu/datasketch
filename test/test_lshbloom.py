import unittest
from mock import patch
from datasketch.lsh_bloom import BitArray, MinHashLSHBloom
from datasketch.minhash import MinHash

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
		
"""
class TestMinHashLSHBloom(unittest.TestCase):

	def test_init(self):
		lsh = MinHashLSHBloom(threshold=0.8)
		b1, r1 = lsh.b, lsh.r
		lsh = MinHashLSHBloom(threshold=0.8, weights=(0.2,0.8))
		b2, r2 = lsh.b, lsh.r
		self.assertTrue(b1 < b2)
		self.assertTrue(r1 > r2)
		

	def test_insert(self):
		lsh = MinHashLSHBloom(threshold=0.5, num_perm=16)
		m1 = MinHash(16)
		m1.update("a".encode("utf8"))
		m2 = MinHash(16)
		m2.update("b".encode("utf8"))
		lsh.insert("a", m1)
		lsh.insert("b", m2)
		for t in lsh.hashtables:
			self.assertTrue(len(t) >= 1)
			items = []
			for H in t:
				items.extend(t[H])
			self.assertTrue("a" in items)
			self.assertTrue("b" in items)
		self.assertTrue("a" in lsh)
		self.assertTrue("b" in lsh)
		for i, H in enumerate(lsh.keys["a"]):
			self.assertTrue("a" in lsh.hashtables[i][H])

		m3 = MinHash(18)
		self.assertRaises(ValueError, lsh.insert, "c", m3)

	def test_query(self):
		lsh = MinHashLSHBloom(threshold=0.5, num_perm=16)
		m1 = MinHash(16)
		m1.update("a".encode("utf8"))
		m2 = MinHash(16)
		m2.update("b".encode("utf8"))
		lsh.insert("a", m1)
		lsh.insert("b", m2)
		result = lsh.query(m1)
		self.assertTrue("a" in result)
		result = lsh.query(m2)
		self.assertTrue("b" in result)

		m3 = MinHash(18)
		self.assertRaises(ValueError, lsh.query, m3)


		with patch('redis.Redis', fake_redis) as mock_redis:
			lsh = MinHashLSHBloom(threshold=0.5, num_perm=16, storage_config={
				'type': 'redis', 'redis': {'host': 'localhost', 'port': 6379}
			})
			m1 = MinHash(16)
			m1.update("a".encode("utf8"))
			m2 = MinHash(16)
			m2.update("b".encode("utf8"))
			lsh.insert("a", m1)
			lsh.insert("b", m2)
			result = lsh.query(m1)
			self.assertTrue("a" in result)
			result = lsh.query(m2)
			self.assertTrue("b" in result)

			m3 = MinHash(18)
			self.assertRaises(ValueError, lsh.query, m3)
"""

if __name__ == "__main__":
	unittest.main()
