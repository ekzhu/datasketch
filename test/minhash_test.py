import unittest
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


if __name__ == "__main__":
    unittest.main()
