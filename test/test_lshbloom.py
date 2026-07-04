import os
import unittest
from glob import glob

import numpy as np

from datasketch.lsh_bloom import BloomTable, MinHashLSHBloom
from datasketch.minhash import MinHash

# pybloomfilter is an optional dependency (the `bloom` extra); the classes
# above import fine without it but raise on use. Skip rather than error when
# it is absent, as the GPU and storage-backend tests do for theirs -- but the
# `bloom` extra IS installed in CI (see .github/workflows/test.yml), so a
# missing pybloomfilter under CI means a broken setup, not an optional-dep
# absence: fail loudly there instead of silently dropping coverage.
try:
    import pybloomfilter  # noqa: F401

    BLOOM_AVAILABLE = True
except ImportError:
    if os.environ.get("CI"):
        raise
    BLOOM_AVAILABLE = False

_SKIP_REASON = "pybloomfilter not installed (pip install datasketch[bloom])"


@unittest.skipUnless(BLOOM_AVAILABLE, _SKIP_REASON)
class TestBloomTable(unittest.TestCase):
    def test_insert(self):
        r = 3
        x = np.array([2, 3, 31], dtype=np.uint32)
        b = BloomTable(10, 0.01, band_size=r)
        b.insert(x)
        self.assertRaises(RuntimeError, b.insert, np.array([2, 2], dtype=np.uint32))

    def test_query(self):
        r = 3
        x = np.array([2, 3, 31], dtype=np.uint32)
        b = BloomTable(10, 0.01, band_size=r)
        b.insert(x)
        self.assertTrue(b.query(x))
        self.assertFalse(b.query(np.array([2, 3, 30], dtype=np.uint32)))
        self.assertRaises(RuntimeError, b.query, [2, 2])

    def test_save(self):
        fname = "/tmp/bloomfilter.bf"  # noqa: S108
        if os.path.exists(fname):
            os.remove(fname)
        r = 3
        x = np.array([2, 3, 31], dtype=np.uint32)
        y = np.array([12, 10, 29], dtype=np.uint32)
        z = np.array([27, 30, 8], dtype=np.uint32)
        items = [x, y, z]
        b = BloomTable(10, 0.01, band_size=r, fname=fname)
        for item in items:
            b.insert(item)
        for item in items:
            self.assertTrue(b.query(item))
        b.sync()

        del b

        b_ = BloomTable(10, 0.01, band_size=r, fname=fname)
        for item in items:
            self.assertTrue(b_.query(item))


@unittest.skipUnless(BLOOM_AVAILABLE, _SKIP_REASON)
class TestMinHashLSHBloom(unittest.TestCase):
    def test_init(self):
        lsh = MinHashLSHBloom(threshold=0.8, n=10, fp=0.01)
        b1, r1 = lsh.b, lsh.r
        lsh = MinHashLSHBloom(threshold=0.8, weights=(0.2, 0.8), n=10, fp=0.01)
        b2, r2 = lsh.b, lsh.r
        self.assertTrue(b1 < b2)
        self.assertTrue(r1 > r2)
        self.assertTrue(len(lsh.hashtables) == lsh.b)

    def test_insert(self):
        lsh = MinHashLSHBloom(threshold=0.5, num_perm=16, n=10, fp=0.01)
        m1 = MinHash(16)
        m1.update(b"a")
        m2 = MinHash(16)
        m2.update(b"b")
        lsh.insert(m1)
        lsh.insert(m2)

        m3 = MinHash(18)
        self.assertRaises(ValueError, lsh.insert, m3)

    def test_query(self):
        lsh = MinHashLSHBloom(threshold=0.5, num_perm=16, n=10, fp=0.01)
        m1 = MinHash(16)
        m1.update(b"a")
        m2 = MinHash(16)
        m2.update(b"b")
        lsh.insert(m1)
        lsh.insert(m2)
        result = lsh.query(m1)
        self.assertTrue(result)
        result = lsh.query(m2)
        self.assertTrue(result)

        m3 = MinHash(18)
        self.assertRaises(ValueError, lsh.query, m3)

    def test_scheme_mismatch(self):
        lsh = MinHashLSHBloom(threshold=0.5, num_perm=16, n=10, fp=0.01)
        m1 = MinHash(16)
        m1.update(b"a")
        lsh.insert(m1)
        m2 = MinHash(16, scheme="legacy")
        m2.update(b"a")
        self.assertRaises(ValueError, lsh.insert, m2)
        self.assertRaises(ValueError, lsh.query, m2)

    def test_save(self):
        save_path = "./test_save/"
        for item in glob(f"{save_path}/*.bf"):
            os.remove(item)

        lsh = MinHashLSHBloom(threshold=0.5, num_perm=16, n=10, fp=0.01, save_dir=save_path)
        m1 = MinHash(16)
        m1.update(b"a")
        m2 = MinHash(16)
        m2.update(b"b")
        lsh.insert(m1)
        lsh.insert(m2)
        lsh.sync()

        lsh2 = MinHashLSHBloom(threshold=0.5, num_perm=16, n=10, fp=0.01, save_dir=save_path)
        result = lsh2.query(m1)
        self.assertTrue(result)
        result = lsh2.query(m2)
        self.assertTrue(result)

    def test_save_in_memory(self):
        with self.assertWarns(RuntimeWarning):
            lsh = MinHashLSHBloom(threshold=0.5, num_perm=16, n=10, fp=0.01, save_dir=None)

        m1 = MinHash(16)
        m1.update(b"a")
        m2 = MinHash(16)
        m2.update(b"b")
        lsh.insert(m1)
        lsh.insert(m2)

        with self.assertWarns(RuntimeWarning):
            lsh.sync()


if __name__ == "__main__":
    unittest.main()
