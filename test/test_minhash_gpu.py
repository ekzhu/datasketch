import pickle
import unittest

import numpy as np

from datasketch import MinHash

# Robust GPU availability check
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

        m_cpu = MinHash(num_perm=256, seed=7, gpu_mode="disable")
        m_cpu.update_batch(data)

        # Force GPU path
        m_gpu = MinHash(num_perm=256, seed=7, gpu_mode="always")
        m_gpu.update_batch(data)

        self.assertTrue(np.array_equal(m_cpu.hashvalues, m_gpu.hashvalues))

    @unittest.skipUnless(GPU_AVAILABLE, "CuPy/CUDA not available")
    def test_detect_mode_matches_cpu(self):
        """Auto-detect should produce identical results as pure CPU."""
        data1 = _make_data(500)
        data2 = _make_data(700)

        m_cpu = MinHash(num_perm=128, seed=7, gpu_mode="disable")
        m_cpu.update_batch(data1)
        m_cpu.update_batch(data2)

        m_auto = MinHash(num_perm=128, seed=7, gpu_mode="detect")
        m_auto.update_batch(data1)
        m_auto.update_batch(data2)

        self.assertTrue(np.array_equal(m_cpu.hashvalues, m_auto.hashvalues))

    def test_pickle_roundtrip_is_portable(self):
        """Pickle should drop device state so round-tripped objects are portable.
        After unpickling, update_batch should still work and populate caches
        only if GPU is available and mode permits it.
        """
        m = MinHash(num_perm=128, seed=7, gpu_mode="detect")
        m2 = pickle.loads(pickle.dumps(m))

        # Should be able to update on any machine
        m2.update_batch(_make_data(64))

        # GPU caches presence should reflect availability & mode
        if "GPU_AVAILABLE" in globals() and GPU_AVAILABLE and m2._gpu_mode in ("detect", "always"):
            self.assertIsNotNone(m2._a_gpu)
            self.assertIsNotNone(m2._b_gpu)
        else:
            self.assertIsNone(m2._a_gpu)
            self.assertIsNone(m2._b_gpu)

    def test_always_mode_raises_when_no_device(self):
        """If GPU is unavailable, 'always' must raise at call-time."""
        if GPU_AVAILABLE:
            self.skipTest("GPU available; cannot force negative path.")
        m = MinHash(num_perm=64, seed=1, gpu_mode="always")
        with self.assertRaises(RuntimeError):
            m.update_batch(_make_data(32))
