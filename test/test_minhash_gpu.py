import contextlib
import pickle
import unittest

import numpy as np

from datasketch import MinHash

# Robust availability check
try:
    import cupy as cp

    try:
        GPU_AVAILABLE = cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        GPU_AVAILABLE = False
except Exception:
    GPU_AVAILABLE = False


def _make_data(n: int):
    return [f"token-{i}".encode("utf-8") for i in range(n)]


class TestMinHashGPU(unittest.TestCase):
    @unittest.skipUnless(GPU_AVAILABLE, "CuPy/CUDA not available")
    def test_update_batch_gpu_matches_cpu(self):
        data = _make_data(1000)

        m_cpu = MinHash(num_perm=256, seed=7)
        m_cpu.update_batch(data)

        m_gpu = MinHash(num_perm=256, seed=7, use_gpu=True)
        m_gpu.update_batch(data)

        self.assertTrue(np.array_equal(m_cpu.hashvalues, m_gpu.hashvalues))

    @unittest.skipUnless(GPU_AVAILABLE, "CuPy/CUDA not available")
    def test_enable_gpu_mid_workflow_matches_cpu(self):
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

    def test_pickle_gpu_enabled_reverts_to_cpu(self):
        m = MinHash(num_perm=128, seed=7)
        # Try enabling GPU if present; ignore RuntimeError if no device
        with contextlib.suppress(RuntimeError):
            m.enable_gpu()
        m2 = pickle.loads(pickle.dumps(m))
        self.assertFalse(getattr(m2, "_use_gpu", False))

    def test_enable_gpu_raises_when_no_device(self):
        if GPU_AVAILABLE:
            self.skipTest("GPU available; cannot force negative path here.")
        m = MinHash(num_perm=64, seed=1)
        with self.assertRaises(RuntimeError):
            m.enable_gpu()

    @unittest.skipUnless(GPU_AVAILABLE, "CuPy/CUDA not available")
    def test_per_call_override(self):
        data = _make_data(300)
        m = MinHash(num_perm=64, seed=3)  # default CPU
        # force GPU per call
        m.update_batch(data, use_gpu=True)
        # then force CPU per call (should still work and be deterministic across backends)
        m2 = MinHash(num_perm=64, seed=3, use_gpu=True)
        m2.update_batch(data, use_gpu=False)
        self.assertTrue(np.array_equal(m.hashvalues, m2.hashvalues))
