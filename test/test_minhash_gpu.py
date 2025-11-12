import unittest

import numpy as np
import pytest

from datasketch.minhash import MinHash

cupy = pytest.importorskip("cupy")


def _make_data(n: int):
    return [f"token-{i}".encode("utf-8") for i in range(n)]


class TestMinHashGPU(unittest.TestCase):
    def test_update_batch_gpu_matches_cpu(self):
        data = _make_data(1000)

        m_cpu = MinHash(num_perm=256, seed=7)
        m_cpu.update_batch(data)

        m_gpu = MinHash(num_perm=256, seed=7)
        m_gpu.enable_gpu()
        m_gpu.update_batch(data)

        # Exact equality of hashvalues
        self.assertTrue(np.array_equal(m_cpu.hashvalues, m_gpu.hashvalues))

    def test_mixed_update_and_update_batch_gpu_matches_cpu(self):
        data1 = _make_data(500)
        data2 = _make_data(700)

        m_cpu = MinHash(num_perm=128, seed=7)
        m_cpu.update_batch(data1)
        m_cpu.update_batch(data2)

        m_gpu = MinHash(num_perm=128, seed=7)
        m_gpu.update_batch(data1)
        m_gpu.enable_gpu()
        m_gpu.update_batch(data2)

        self.assertTrue(np.array_equal(m_cpu.hashvalues, m_gpu.hashvalues))
