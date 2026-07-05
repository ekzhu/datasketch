import pickle
import types
import unittest

import numpy as np

import datasketch.minhash as _mh
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


# The GPU path branches on the permutation scheme (legacy modular
# arithmetic vs affine multiply-shift), so parity must hold per scheme.
SCHEMES = ["affine32", "affine64", "legacy"]


class TestMinHashGPU(unittest.TestCase):
    @unittest.skipUnless(GPU_AVAILABLE, "CuPy/CUDA not available")
    def test_update_batch_gpu_matches_cpu(self):
        data = _make_data(1000)

        for scheme in SCHEMES:
            with self.subTest(scheme=scheme):
                m_cpu = MinHash(
                    num_perm=256, seed=7, gpu_mode="disable", scheme=scheme)
                m_cpu.update_batch(data)

                # Force GPU path
                m_gpu = MinHash(
                    num_perm=256, seed=7, gpu_mode="always", scheme=scheme)
                m_gpu.update_batch(data)

                self.assertEqual(
                    m_cpu.hashvalues.dtype, m_gpu.hashvalues.dtype)
                self.assertTrue(
                    np.array_equal(m_cpu.hashvalues, m_gpu.hashvalues))

    @unittest.skipUnless(GPU_AVAILABLE, "CuPy/CUDA not available")
    def test_detect_mode_matches_cpu(self):
        """Auto-detect should produce identical results as pure CPU."""
        data1 = _make_data(500)
        data2 = _make_data(700)

        for scheme in SCHEMES:
            with self.subTest(scheme=scheme):
                m_cpu = MinHash(
                    num_perm=128, seed=7, gpu_mode="disable", scheme=scheme)
                m_cpu.update_batch(data1)
                m_cpu.update_batch(data2)

                m_auto = MinHash(
                    num_perm=128, seed=7, gpu_mode="detect", scheme=scheme)
                m_auto.update_batch(data1)
                m_auto.update_batch(data2)

                self.assertTrue(
                    np.array_equal(m_cpu.hashvalues, m_auto.hashvalues))

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


class TestGPUPathOnCPU(unittest.TestCase):
    """Exercise the GPU branch of update_batch without a CUDA device by
    injecting a numpy-backed CuPy stand-in. This runs the same code the GPU
    path runs (per scheme) and asserts it matches the CPU path -- unlike the
    hardware-gated tests above, it always runs.
    """

    @staticmethod
    def _shim():
        # Only the cupy surface update_batch / _ensure_gpu_caches touch.
        return types.SimpleNamespace(
            uint32=np.uint32, uint64=np.uint64,
            asarray=np.asarray, minimum=np.minimum, min=np.min,
            bitwise_and=np.bitwise_and, asnumpy=np.asarray,
        )

    def _run_with_shim(self, cp_module, gpu_cache):
        saved = (_mh.cp, _mh._GPU_OK_CACHE)
        _mh.cp, _mh._GPU_OK_CACHE = cp_module, gpu_cache
        try:
            data = _make_data(500)
            results = {}
            for scheme in SCHEMES:
                m = MinHash(num_perm=128, seed=3, scheme=scheme, gpu_mode="always")
                m.update_batch(data)
                results[scheme] = m.hashvalues
            return results
        finally:
            _mh.cp, _mh._GPU_OK_CACHE = saved

    def test_gpu_branch_matches_cpu_all_schemes(self):
        gpu = self._run_with_shim(self._shim(), True)
        data = _make_data(500)
        for scheme in SCHEMES:
            m_cpu = MinHash(num_perm=128, seed=3, scheme=scheme, gpu_mode="disable")
            m_cpu.update_batch(data)
            self.assertEqual(m_cpu.hashvalues.dtype, gpu[scheme].dtype, scheme)
            self.assertTrue(np.array_equal(m_cpu.hashvalues, gpu[scheme]), scheme)

    def test_shim_catches_wrong_cache_dtype(self):
        # Mutation guard: forcing uint64 caches for affine32 changes the
        # pre-truncation minimum ordering, so the result must differ from CPU.
        bad = self._shim()
        bad.uint32 = np.uint64  # affine32 caches wrongly built as uint64
        gpu = self._run_with_shim(bad, True)
        m_cpu = MinHash(num_perm=128, seed=3, scheme="affine32", gpu_mode="disable")
        m_cpu.update_batch(_make_data(500))
        self.assertFalse(np.array_equal(m_cpu.hashvalues, gpu["affine32"]))
