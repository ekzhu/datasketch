import unittest
import pickle
import numpy as np
from datasketch.weighted_minhash import WeightedMinHashGenerator, WeightedMinHash

class TestWeightedMinHash(unittest.TestCase):

    def test_pickle(self):
        mg = WeightedMinHashGenerator(4, 10, 1)
        m = mg.minhash([1,2,3,4])
        p = pickle.loads(pickle.dumps(m))
        self.assertEqual(p.seed, m.seed)
        self.assertTrue(np.array_equal(p.hashvalues, m.hashvalues))

class TestWeightedMinHashGenerator(unittest.TestCase):

    def test_init(self):
        mg = WeightedMinHashGenerator(2, 4, 1)
        self.assertEqual(len(mg.rs), 4)
        self.assertEqual(len(mg.ln_cs), 4)
        self.assertEqual(len(mg.betas), 4)
        self.assertEqual(mg.seed, 1)
        self.assertEqual(mg.sample_size, 4)

    def test_minhash(self):
        mg = WeightedMinHashGenerator(2, 4, 1)
        m = mg.minhash([1,3])
        self.assertIsInstance(m, WeightedMinHash)
        self.assertEqual(len(m.hashvalues), 4)
        self.assertEqual(len(m), 4)
        self.assertTrue(m.hashvalues.dtype == np.int)

if __name__ == "__main__":
    unittest.main()
