import unittest
import numpy as np
from datasketch import minhash, bulk
from test.utils import fake_hash_func


class TestBulk(unittest.TestCase):
    
    def test_compute_minhashes(self):
        kwargs = dict(num_perm=4, seed=1, hashfunc=fake_hash_func)
        b = [[n*4 for n in range(4)]]*2
        m1 = minhash.MinHash(**kwargs)
        m1.update_batch(b[0])
        m2, m3 = bulk.compute_minhashes(b, **kwargs)
        self.assertTrue(np.array_equal(m1.hashvalues, m2.hashvalues))
        self.assertTrue(np.array_equal(m1.hashvalues, m3.hashvalues))

if __name__ == "__main__":
    unittest.main()
