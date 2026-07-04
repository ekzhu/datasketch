"""Tests for the MinHash permutation schemes introduced in version 2.0.0.

Covers the "affine32"/"affine64" schemes (fmix pre-mix + odd-a affine
permutations mod 2^w), frozen serialization formats, statistical behavior,
and bit-exact backward compatibility of the "legacy" scheme against fixtures
generated with datasketch 1.10.0.
"""

import base64
import pickle
import struct
import unittest

import numpy as np

from datasketch import (
    LeanMinHash,
    MinHash,
    MinHashLSH,
    MinHashLSHEnsemble,
    MinHashLSHForest,
    WeightedMinHashGenerator,
)
from datasketch.b_bit_minhash import bBitMinHash
from datasketch.hashfunc import sha1_hash32, sha1_hash64
from datasketch.minhash import _fmix
from test.utils import fake_hash_func

AFFINE_SCHEMES = ("affine32", "affine64")
ALL_SCHEMES = ("affine32", "affine64", "legacy")

# --------------------------------------------------------------------------
# Frozen format vectors. These freeze the affine schemes' permutation
# generation and hash computation: any change to these values is a breaking
# format change that invalidates persisted sketches.
# --------------------------------------------------------------------------
FROZEN_PERMUTATIONS = {
    # scheme -> (a, b) for seed=1, num_perm=4
    "affine32": (
        [3582191691, 4270784983, 1892572953, 3715639441],
        [491263, 550290313, 1298508491, 4290846341],
    ),
    "affine64": (
        [15385396165118722519, 8128538942144784529, 4219918138050323, 11154103013428313355],
        [2707168392203228096, 1703346441743126657, 3435894450290564679, 6374470260350024188],
    ),
}
FROZEN_HELLO_HASHVALUES = {
    # scheme -> hashvalues of MinHash(4, 1, scheme=scheme).update(b"Hello")
    "affine32": [3231258861, 1388533999, 831950533, 1845606703],
    "affine64": [17263807126536662982, 6203611050164820875, 4141308390362884389, 4915270424612038026],
}

# --------------------------------------------------------------------------
# Legacy fixtures generated with datasketch 1.10.0 (before the scheme
# parameter existed). They guarantee that the legacy scheme keeps producing
# bit-identical values and that old serialized payloads still load.
# --------------------------------------------------------------------------
LEGACY_TOKENS_A = [b"datasketch", b"minhash", b"affine", b"legacy", b"fixture"]
LEGACY_TOKENS_B = [b"datasketch", b"minhash", b"affine", b"probabilistic", b"structure"]
# MinHash(num_perm=8, seed=42) updated with LEGACY_TOKENS_A / LEGACY_TOKENS_B.
LEGACY_HASHVALUES_A = [615829395, 403905604, 2148529625, 66448688, 1847255163, 1126317356, 318181934, 818505764]
LEGACY_HASHVALUES_B = [615829395, 403905604, 2148529625, 23145198, 1767462284, 1126317356, 1395086151, 818505764]
LEGACY_JACCARD_AB = 0.625
# The permutation parameters of a legacy MinHash with num_perm=4, seed=1.
LEGACY_PERM_A = [775169054918279404, 2109959069025162, 401325382989534145, 1130051441076870728]
LEGACY_PERM_B = [1758426461858698312, 965365488286768773, 1703346441743126657, 1762784241922636284]
# pickle.dumps of the LEGACY_TOKENS_A MinHash under 1.10.0
LEGACY_MINHASH_PICKLE = base64.b64decode(
    "".join(
        [
            "gASVHwIAAAAAAACMEmRhdGFza2V0Y2gubWluaGFzaJSMB01pbkhhc2iUk5QpgZR9lCiMBHNlZWSUSyqMCG51bV9wZXJtlEsIjAho",
            "YXNoZnVuY5SME2RhdGFza2V0Y2guaGFzaGZ1bmOUjAtzaGExX2hhc2gzMpSTlIwKaGFzaHZhbHVlc5SMFm51bXB5Ll9jb3JlLm11",
            "bHRpYXJyYXmUjAxfcmVjb25zdHJ1Y3SUk5SMBW51bXB5lIwHbmRhcnJheZSTlEsAhZRDAWKUh5RSlChLAUsIhZRoD4wFZHR5cGWU",
            "k5SMAnU4lImIh5RSlChLA4wBPJROTk5K/////0r/////SwB0lGKJQ0CTz7QkAAAAAEQcExgAAAAA2fUPgAAAAAAw7fUDAAAAAHvg",
            "Gm4AAAAALD0iQwAAAAAuEvcSAAAAACRoyTAAAAAAlHSUYowMcGVybXV0YXRpb25zlGgOaBFLAIWUaBOHlFKUKEsBSwJLCIaUaBuI",
            "Q4C0PerLZtzhHw6V9S5cA2ITSNSZx2r0YxsULMuYvK5BGXoYInJm1vAH1vSXGdJB7wfLVZF1SpHeDnTtbVVXuL0daOuSJGOs4hmC",
            "oKCmlz5EFTX9cA6VBkUFV+nUuAEDTBie/UTw6/UaFYEJMwAl21sWvJMB/r8UjA6gRBOeFJnzDpR0lGKMCV9ncHVfbW9kZZSMB2Rp",
            "c2FibGWUjAZfYV9ncHWUTowGX2JfZ3B1lE51Yi4=",
        ]
    )
)
# LeanMinHash(minhash_a).serialize(buf, "<") under 1.10.0
LEGACY_LEAN_SERIALIZED_LE = bytes.fromhex(
    "2a000000000000000800000093cfb424441c1318d9f50f8030edf5037be01a6e2c3d22432e12f7122468c930"
)
# pickle.dumps(LeanMinHash(minhash_a)) under 1.10.0
LEGACY_LEAN_PICKLE = base64.b64decode(
    "".join(
        [
            "gASVewAAAAAAAACMF2RhdGFza2V0Y2gubGVhbl9taW5oYXNolIwLTGVhbk1pbkhhc2iUk5QpgZSMCGJ1aWx0aW5zlIwJYnl0ZWFy",
            "cmF5lJOUQywqAAAAAAAAAAgAAACTz7QkRBwTGNn1D4Aw7fUDe+Aabiw9IkMuEvcSJGjJMJSFlFKUYi4=",
        ]
    )
)
# pickle.dumps(bBitMinHash(minhash_a, b=4)) under 1.10.0
LEGACY_BBIT_PICKLE = base64.b64decode(
    "".join(
        [
            "gASVbQAAAAAAAACMGGRhdGFza2V0Y2guYl9iaXRfbWluaGFzaJSMC2JCaXRNaW5IYXNolJOUKYGUjAhidWlsdGluc5SMCWJ5dGVh",
            "cnJheZSTlEMdKgAAAAAAAAAEAAAAAAAAAAAIAAAAAAAAAOS8kDSUhZRSlGIu",
        ]
    )
)


class TestSchemeBasics(unittest.TestCase):
    def test_invalid_scheme_raises(self):
        with self.assertRaises(ValueError):
            MinHash(4, 1, scheme="affine128")

    def test_default_scheme_is_affine32(self):
        m = MinHash(4, 1)
        self.assertEqual(m.scheme, "affine32")
        self.assertEqual(m.hashvalues.dtype, np.uint32)

    def test_default_hashfunc_matches_scheme_width(self):
        self.assertIs(MinHash(4, 1).hashfunc, sha1_hash32)
        self.assertIs(MinHash(4, 1, scheme="legacy").hashfunc, sha1_hash32)
        self.assertIs(MinHash(4, 1, scheme="affine64").hashfunc, sha1_hash64)
        self.assertIs(MinHash(4, 1, scheme="affine64", hashfunc=fake_hash_func).hashfunc, fake_hash_func)

    def test_dtype_and_empty_sentinel(self):
        expected = {
            "affine32": (np.uint32, (1 << 32) - 1),
            "affine64": (np.uint64, (1 << 64) - 1),
            "legacy": (np.uint64, (1 << 32) - 1),
        }
        for scheme, (dtype, sentinel) in expected.items():
            m = MinHash(4, 1, scheme=scheme)
            self.assertEqual(m.hashvalues.dtype, dtype, scheme)
            self.assertTrue((m.hashvalues == sentinel).all(), scheme)
            self.assertTrue(m.is_empty(), scheme)
            m.update(b"x")
            self.assertFalse(m.is_empty(), scheme)
            m.clear()
            self.assertTrue(m.is_empty(), scheme)

    def test_update_matches_update_batch(self):
        for scheme in ALL_SCHEMES:
            m1 = MinHash(16, 1, scheme=scheme)
            for t in (b"a", b"b", b"c"):
                m1.update(t)
            m2 = MinHash(16, 1, scheme=scheme)
            m2.update_batch([b"a", b"b", b"c"])
            self.assertTrue(np.array_equal(m1.hashvalues, m2.hashvalues), scheme)

    def test_jaccard_identical_and_disjoint(self):
        for scheme in ALL_SCHEMES:
            m1 = MinHash(32, 1, scheme=scheme, hashfunc=fake_hash_func)
            m2 = MinHash(32, 1, scheme=scheme, hashfunc=fake_hash_func)
            self.assertEqual(m1.jaccard(m2), 1.0, scheme)
            m2.update(12)
            self.assertEqual(m1.jaccard(m2), 0.0, scheme)

    def test_count(self):
        for scheme in ALL_SCHEMES:
            m = MinHash(scheme=scheme)
            m.update_batch([b"a", b"b", b"c", b"d", b"e", b"f"])
            self.assertGreaterEqual(m.count(), 0, scheme)

    def test_copy_merge_union_preserve_scheme(self):
        for scheme in ALL_SCHEMES:
            m1 = MinHash(16, 1, scheme=scheme)
            m1.update(b"a")
            m2 = MinHash(16, 1, scheme=scheme)
            m2.update(b"b")
            c = m1.copy()
            self.assertEqual(c.scheme, scheme)
            self.assertEqual(c, m1)
            u = MinHash.union(m1, m2)
            self.assertEqual(u.scheme, scheme)
            m1.merge(m2)
            self.assertEqual(u, m1)

    def test_pickle_roundtrip(self):
        for scheme in ALL_SCHEMES:
            m = MinHash(16, 1, scheme=scheme)
            m.update(b"a")
            p = pickle.loads(pickle.dumps(m))
            self.assertEqual(p.scheme, scheme)
            self.assertEqual(p, m)
            self.assertTrue(np.array_equal(np.asarray(p.permutations), np.asarray(m.permutations)), scheme)

    def test_bulk_and_generator_pass_scheme(self):
        data = [[b"a", b"b"], [b"c", b"d"]]
        for scheme in ALL_SCHEMES:
            for m in MinHash.bulk(data, num_perm=8, scheme=scheme):
                self.assertEqual(m.scheme, scheme)

    def test_eq_differs_across_schemes(self):
        m32 = MinHash(4, 1, scheme="affine32")
        m64 = MinHash(4, 1, scheme="affine64")
        ml = MinHash(4, 1, scheme="legacy")
        self.assertNotEqual(m32, m64)
        self.assertNotEqual(m32, ml)
        self.assertNotEqual(m64, ml)

    def test_cross_scheme_operations_refused(self):
        m32 = MinHash(8, 1, scheme="affine32")
        ml = MinHash(8, 1, scheme="legacy")
        with self.assertRaises(ValueError):
            m32.jaccard(ml)
        with self.assertRaises(ValueError):
            m32.merge(ml)
        with self.assertRaises(ValueError):
            MinHash.union(m32, ml)

    def test_hashfunc_out_of_range_raises(self):
        m = MinHash(4, 1, scheme="affine32", hashfunc=lambda x: 1 << 40)
        with self.assertRaises(ValueError):
            m.update(b"x")
        with self.assertRaises(ValueError):
            m.update_batch([b"x"])

    def test_affine64_uses_full_range(self):
        m = MinHash(64, 1, scheme="affine64")
        m.update_batch([b"a", b"b", b"c"])
        self.assertTrue((m.hashvalues >= (1 << 32)).any())

    def test_permutation_validation(self):
        # Even `a` breaks bijectivity and must be rejected.
        with self.assertRaises(ValueError):
            MinHash(num_perm=2, seed=1, scheme="affine32", permutations=([3, 2], [0, 1]))
        # Parameters beyond the scheme width must be rejected, not wrapped.
        with self.assertRaises(ValueError):
            MinHash(
                num_perm=2,
                seed=1,
                scheme="affine32",
                permutations=(np.array([1 << 40, 3], dtype=np.uint64), np.array([0, 1], dtype=np.uint64)),
            )
        # Valid parameters round-trip through the constructor (e.g. copy()).
        m = MinHash(num_perm=8, seed=1, scheme="affine64")
        m2 = MinHash(num_perm=8, seed=1, scheme="affine64", permutations=m.permutations)
        self.assertTrue(np.array_equal(np.asarray(m.permutations), np.asarray(m2.permutations)))

    def test_hashvalues_out_of_range_rejected(self):
        with self.assertRaises(ValueError):
            MinHash(seed=1, scheme="affine32", hashvalues=np.array([1 << 40, 1], dtype=np.uint64))

    def test_hashvalues_of_python_ints_convert_exactly(self):
        # dtype inference on mixed-magnitude Python ints would produce
        # float64 and silently round 64-bit values.
        values = [1, (1 << 64) - 59]
        m = MinHash(seed=1, scheme="affine64", hashvalues=values, permutations=([1, 3], [0, 1]))
        self.assertEqual(m.hashvalues.tolist(), values)


class TestExplicitSchemeRequirement(unittest.TestCase):
    """Constructing from raw hash values or permutations requires an explicit
    scheme: the values carry no trace of the scheme that produced them, so a
    default would silently mislabel pre-2.0.0 state and defeat the
    cross-scheme guards.
    """

    def test_minhash_raw_hashvalues_require_scheme(self):
        m = MinHash(8, 1, scheme="legacy")
        with self.assertRaises(ValueError) as ctx:
            MinHash(seed=m.seed, hashvalues=m.hashvalues)
        self.assertIn("legacy", str(ctx.exception))
        rebuilt = MinHash(seed=m.seed, hashvalues=m.hashvalues, scheme="legacy")
        self.assertEqual(rebuilt.jaccard(m), 1.0)

    def test_minhash_raw_permutations_require_scheme(self):
        m = MinHash(8, 1)
        with self.assertRaises(ValueError):
            MinHash(num_perm=8, seed=1, permutations=m.permutations)

    def test_lean_minhash_raw_hashvalues_require_scheme(self):
        m = MinHash(8, 1)
        with self.assertRaises(ValueError) as ctx:
            LeanMinHash(seed=m.seed, hashvalues=m.hashvalues)
        self.assertIn("legacy", str(ctx.exception))

    def test_lean_minhash_scheme_conflicting_with_minhash_rejected(self):
        m = MinHash(8, 1, scheme="affine32")
        with self.assertRaises(ValueError):
            LeanMinHash(m, scheme="legacy")
        self.assertEqual(LeanMinHash(m, scheme="affine32").scheme, "affine32")

    def test_num_perm_must_be_positive(self):
        for scheme in ALL_SCHEMES:
            with self.assertRaises(ValueError):
                MinHash(0, 1, scheme=scheme)
        with self.assertRaises(ValueError):
            MinHash(seed=1, hashvalues=[], scheme="affine32")

    def test_empty_affine_lean_minhash_rejected(self):
        # An empty affine sketch would serialize with a hash value count of
        # zero, indistinguishable from the legacy format on deserialization.
        with self.assertRaises(ValueError):
            LeanMinHash(seed=1, hashvalues=[], scheme="affine32")
        # Empty legacy payloads (possible under 1.x) still round-trip.
        lm = LeanMinHash(seed=1, hashvalues=[], scheme="legacy")
        buf = bytearray(lm.bytesize("<"))
        lm.serialize(buf, "<")
        self.assertEqual(LeanMinHash.deserialize(buf, "<").scheme, "legacy")

    def test_negative_hashvalues_rejected(self):
        with self.assertRaises(ValueError):
            MinHash(seed=1, hashvalues=np.array([-1, 2], dtype=np.int64), scheme="affine32")
        with self.assertRaises(ValueError):
            MinHash(seed=1, hashvalues=[-1, 2], scheme="affine64")


class TestLSHSchemeGuards(unittest.TestCase):
    """LSH indexes must refuse MinHash of a scheme other than the one they
    were built with: mixed schemes silently return wrong results otherwise.
    """

    def _mh(self, scheme, seed=1, num_perm=128):
        m = MinHash(num_perm, seed, scheme=scheme)
        m.update_batch([b"a", b"b", b"c"])
        return m

    def test_lsh_insert_and_query_mixed_scheme_raise(self):
        lsh = MinHashLSH(threshold=0.5, num_perm=128)
        lsh.insert("k32", self._mh("affine32"))
        with self.assertRaises(ValueError):
            lsh.insert("kleg", self._mh("legacy"))
        with self.assertRaises(ValueError):
            lsh.query(self._mh("affine64"))
        with self.assertRaises(ValueError):
            lsh.add_to_query_buffer(self._mh("affine64"))
        self.assertIn("k32", lsh.query(self._mh("affine32")))

    def test_lsh_insertion_session_guarded(self):
        lsh = MinHashLSH(threshold=0.5, num_perm=128)
        with lsh.insertion_session() as session:
            session.insert("k", self._mh("affine32"))
            with self.assertRaises(ValueError):
                session.insert("k2", self._mh("legacy"))

    def test_lsh_merge_mixed_scheme_raises(self):
        a = MinHashLSH(threshold=0.5, num_perm=128)
        a.insert("x", self._mh("affine32"))
        b = MinHashLSH(threshold=0.5, num_perm=128)
        b.insert("y", self._mh("legacy"))
        with self.assertRaises(ValueError):
            a.merge(b)
        # Merging propagates the learned scheme to an empty index.
        c = MinHashLSH(threshold=0.5, num_perm=128)
        c.merge(a)
        with self.assertRaises(ValueError):
            c.insert("z", self._mh("legacy"))

    def test_lsh_pickle_preserves_learned_scheme(self):
        lsh = MinHashLSH(threshold=0.5, num_perm=128)
        lsh.insert("k", self._mh("affine32"))
        restored = pickle.loads(pickle.dumps(lsh))
        with self.assertRaises(ValueError):
            restored.query(self._mh("legacy"))
        self.assertIn("k", restored.query(self._mh("affine32")))

    def test_lshforest_mixed_scheme_raises(self):
        forest = MinHashLSHForest(num_perm=128)
        forest.add("k", self._mh("affine32"))
        with self.assertRaises(ValueError):
            forest.add("k2", self._mh("legacy"))
        forest.index()
        with self.assertRaises(ValueError):
            forest.query(self._mh("legacy"), 1)
        self.assertIn("k", forest.query(self._mh("affine32"), 1))

    def test_lshensemble_mixed_scheme_raises(self):
        ensemble = MinHashLSHEnsemble(threshold=0.8, num_perm=128, num_part=2)
        ensemble.index([("k", self._mh("affine32"), 3)])
        self.assertIn("k", list(ensemble.query(self._mh("affine32"), 3)))
        with self.assertRaises(ValueError):
            list(ensemble.query(self._mh("legacy"), 3))

    def test_weighted_minhash_exempt(self):
        # WeightedMinHash has no permutation scheme and must keep working.
        gen = WeightedMinHashGenerator(10, sample_size=128, seed=1)
        v = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        lsh = MinHashLSH(threshold=0.5, num_perm=128)
        lsh.insert("w1", gen.minhash(v))
        self.assertIn("w1", lsh.query(gen.minhash(v)))


class TestFrozenFormat(unittest.TestCase):
    """Freeze the affine formats: a change here invalidates persisted sketches."""

    def test_permutations_frozen(self):
        for scheme, (a, b) in FROZEN_PERMUTATIONS.items():
            m = MinHash(num_perm=4, seed=1, scheme=scheme)
            self.assertEqual(m.permutations[0].tolist(), a, scheme)
            self.assertEqual(m.permutations[1].tolist(), b, scheme)

    def test_hashvalues_frozen(self):
        for scheme, expected in FROZEN_HELLO_HASHVALUES.items():
            m = MinHash(4, 1, scheme=scheme)
            m.update(b"Hello")
            self.assertEqual(m.hashvalues.tolist(), expected, scheme)

    def test_permutations_deterministic_per_seed(self):
        for scheme in AFFINE_SCHEMES:
            p1 = np.asarray(MinHash(16, 7, scheme=scheme).permutations)
            p2 = np.asarray(MinHash(16, 7, scheme=scheme).permutations)
            p3 = np.asarray(MinHash(16, 8, scheme=scheme).permutations)
            self.assertTrue(np.array_equal(p1, p2), scheme)
            self.assertFalse(np.array_equal(p1, p3), scheme)


class TestBijectivity(unittest.TestCase):
    def test_a_parameters_are_odd(self):
        for scheme in AFFINE_SCHEMES:
            for seed in range(8):
                a = MinHash(num_perm=256, seed=seed, scheme=scheme).permutations[0]
                self.assertTrue((a % 2 == 1).all(), scheme)

    def test_fmix_is_injective_on_sample(self):
        rng = np.random.RandomState(5)
        x32 = np.unique(rng.randint(0, 1 << 32, size=200000, dtype=np.uint32))
        y32 = _fmix(x32, 32)
        self.assertEqual(y32.dtype, np.uint32)
        self.assertEqual(len(np.unique(y32)), len(x32))
        x64 = np.unique(rng.randint(0, 1 << 63, size=200000, dtype=np.uint64) * np.uint64(2))
        y64 = _fmix(x64, 64)
        self.assertEqual(y64.dtype, np.uint64)
        self.assertEqual(len(np.unique(y64)), len(x64))

    def test_fmix_fixes_zero(self):
        # fmix consists of xor-shift and multiplication steps, all of which
        # map zero to zero; a nonzero result would indicate broken constants.
        self.assertEqual(int(_fmix(np.array([0], dtype=np.uint32), 32)[0]), 0)
        self.assertEqual(int(_fmix(np.array([0], dtype=np.uint64), 64)[0]), 0)

    def test_permutation_is_injective_on_sample(self):
        rng = np.random.RandomState(6)
        for scheme, dtype, width in (("affine32", np.uint32, 32), ("affine64", np.uint64, 64)):
            m = MinHash(num_perm=2, seed=3, scheme=scheme)
            a, b = m.permutations[0][0], m.permutations[1][0]
            x = np.unique(rng.randint(0, 1 << min(width, 62), size=100000, dtype=dtype))
            phv = a * x + b
            self.assertEqual(len(np.unique(phv)), len(x), scheme)


class TestStatisticalBehavior(unittest.TestCase):
    """Seeded (deterministic) ports of the issue #212 bias experiments."""

    def _structured_bias(self, scheme, trials=12, n=8000, num_perm=128):
        # Sequential integers hashed with the identity function: true J = 1/3.
        # Without the fmix pre-mix the affine schemes would be biased by
        # about -0.05 here; the legacy scheme is biased regardless.
        errors = []
        for seed in range(trials):
            ma = MinHash(num_perm, seed=seed, hashfunc=fake_hash_func, scheme=scheme)
            mb = MinHash(num_perm, seed=seed, hashfunc=fake_hash_func, scheme=scheme)
            ma.update_batch(range(0, 2 * n))
            mb.update_batch(range(n, 3 * n))
            errors.append(ma.jaccard(mb) - 1.0 / 3.0)
        return sum(errors) / len(errors)

    def test_affine_unbiased_on_structured_input(self):
        for scheme in AFFINE_SCHEMES:
            self.assertLess(abs(self._structured_bias(scheme)), 0.03, scheme)

    def test_legacy_biased_on_structured_input(self):
        # Guards the test harness itself: the legacy scheme must reproduce
        # the known bias on this input (measured -0.061 for these seeds).
        self.assertLess(self._structured_bias("legacy"), -0.04)

    def _structured_count_error(self, scheme, trials=12, n=5000, num_perm=256):
        # Relative error of count() on sequential integers under an identity
        # hash -- the structured-input class the fmix pre-mix was built for.
        errors = []
        for seed in range(trials):
            m = MinHash(num_perm, seed=seed, hashfunc=fake_hash_func, scheme=scheme)
            m.update_batch(range(n))
            errors.append(m.count() / n - 1.0)
        return sum(errors) / len(errors)

    def test_affine_count_accurate_on_structured_input(self):
        # affine's fmix keeps cardinality estimates accurate (measured
        # about +0.013 to +0.027 for these seeds) well within tolerance.
        for scheme in AFFINE_SCHEMES:
            self.assertLess(abs(self._structured_count_error(scheme)), 0.10, scheme)

    def test_legacy_count_collapses_on_structured_input(self):
        # The legacy permutation collapses on structured input, so count()
        # is far off (measured about -0.40 for these seeds).
        self.assertLess(self._structured_count_error("legacy"), -0.15)

    def test_affine_accuracy_on_random_sets(self):
        rng = np.random.RandomState(99)
        for scheme in AFFINE_SCHEMES:
            errors = []
            for t in range(6):
                u = list(dict.fromkeys(rng.randint(0, 1 << 31, size=9000).tolist()))
                common, a_only, b_only = u[:2500], u[2500:5000], u[5000:7500]
                sa, sb = common + a_only, common + b_only
                true_j = len(set(sa) & set(sb)) / len(set(sa) | set(sb))
                ma = MinHash(128, seed=t, hashfunc=fake_hash_func, scheme=scheme)
                mb = MinHash(128, seed=t, hashfunc=fake_hash_func, scheme=scheme)
                ma.update_batch(sa)
                mb.update_batch(sb)
                errors.append(ma.jaccard(mb) - true_j)
            self.assertLess(abs(sum(errors) / len(errors)), 0.05, scheme)


class TestLegacyCompatibility(unittest.TestCase):
    """Bit-exact compatibility with sketches created by datasketch 1.10.0."""

    def _legacy_minhash(self, tokens):
        m = MinHash(num_perm=8, seed=42, scheme="legacy")
        for t in tokens:
            m.update(t)
        return m

    def test_legacy_update_bit_identical(self):
        self.assertEqual(self._legacy_minhash(LEGACY_TOKENS_A).hashvalues.tolist(), LEGACY_HASHVALUES_A)
        self.assertEqual(self._legacy_minhash(LEGACY_TOKENS_B).hashvalues.tolist(), LEGACY_HASHVALUES_B)

    def test_legacy_permutations_bit_identical(self):
        m = MinHash(num_perm=4, seed=1, scheme="legacy")
        self.assertEqual(m.permutations[0].tolist(), LEGACY_PERM_A)
        self.assertEqual(m.permutations[1].tolist(), LEGACY_PERM_B)

    def test_old_minhash_pickle_loads_as_legacy(self):
        old = pickle.loads(LEGACY_MINHASH_PICKLE)
        self.assertEqual(old.scheme, "legacy")
        self.assertEqual(old.hashvalues.tolist(), LEGACY_HASHVALUES_A)
        self.assertAlmostEqual(old.jaccard(self._legacy_minhash(LEGACY_TOKENS_B)), LEGACY_JACCARD_AB)
        # Still updatable with bit-identical semantics.
        old.update(b"more")
        m = self._legacy_minhash([*LEGACY_TOKENS_A, b"more"])
        self.assertEqual(old.hashvalues.tolist(), m.hashvalues.tolist())

    def test_minhash_unpickles_without_scheme_as_legacy(self):
        m = MinHash(4, 1, hashfunc=fake_hash_func, scheme="legacy")
        state = m.__getstate__()
        state.pop("scheme")
        restored = MinHash.__new__(MinHash)
        restored.__setstate__(state)
        self.assertEqual(restored.scheme, "legacy")

    def test_old_lean_bytes_deserialize_and_reserialize_identically(self):
        lm = LeanMinHash.deserialize(LEGACY_LEAN_SERIALIZED_LE, "<")
        self.assertEqual(lm.scheme, "legacy")
        self.assertEqual(lm.seed, 42)
        self.assertEqual(lm.hashvalues.tolist(), LEGACY_HASHVALUES_A)
        buf = bytearray(lm.bytesize("<"))
        lm.serialize(buf, "<")
        self.assertEqual(bytes(buf), LEGACY_LEAN_SERIALIZED_LE)

    def test_old_lean_pickle_loads_and_state_unchanged(self):
        lm = pickle.loads(LEGACY_LEAN_PICKLE)
        self.assertEqual(lm.scheme, "legacy")
        self.assertEqual(lm.hashvalues.tolist(), LEGACY_HASHVALUES_A)
        # A legacy LeanMinHash pickled by this version keeps the old byte
        # format, so it remains readable by versions before 2.0.0.
        self.assertEqual(pickle.loads(pickle.dumps(lm)), lm)
        self.assertEqual(bytes(lm.__getstate__()), bytes(pickle.loads(LEGACY_LEAN_PICKLE).__getstate__()))

    def test_old_bbit_pickle_loads_as_legacy(self):
        bb = pickle.loads(LEGACY_BBIT_PICKLE)
        self.assertEqual(bb.scheme, "legacy")
        self.assertEqual(bb.seed, 42)
        self.assertEqual(bb.b, 4)
        rebuilt = bBitMinHash(self._legacy_minhash(LEGACY_TOKENS_A), b=4)
        self.assertEqual(bb, rebuilt)
        self.assertEqual(bytes(bb.__getstate__()), bytes(rebuilt.__getstate__()))


class TestLeanMinHashSchemes(unittest.TestCase):
    def _lean(self, scheme, num_perm=16):
        m = MinHash(num_perm, 1, scheme=scheme)
        m.update_batch([b"a", b"b", b"c"])
        return LeanMinHash(m)

    def test_scheme_taken_from_minhash(self):
        for scheme in ALL_SCHEMES:
            self.assertEqual(self._lean(scheme).scheme, scheme)

    def test_init_from_seed_and_hashvalues(self):
        for scheme in ALL_SCHEMES:
            m = MinHash(16, 1, scheme=scheme)
            m.update(b"a")
            lm = LeanMinHash(seed=m.seed, hashvalues=m.hashvalues, scheme=scheme)
            self.assertEqual(lm.scheme, scheme)
            self.assertEqual(lm.jaccard(m), 1.0)

    def test_serialize_roundtrip_all_byteorders(self):
        for scheme in ALL_SCHEMES:
            lm = self._lean(scheme)
            for byteorder in "@=<>!":
                buf = bytearray(lm.bytesize(byteorder))
                lm.serialize(buf, byteorder)
                lm2 = LeanMinHash.deserialize(buf, byteorder)
                self.assertEqual(lm2, lm, (scheme, byteorder))
                self.assertEqual(lm2.scheme, scheme, (scheme, byteorder))

    def test_affine_payload_layout(self):
        # seed:int64, negated num_perm:int32, scheme code:uint8, values.
        for scheme, code, value_fmt in (("affine32", 1, "I"), ("affine64", 2, "Q")):
            lm = self._lean(scheme, num_perm=4)
            self.assertEqual(lm.bytesize("<"), struct.calcsize("<qiB4%s" % value_fmt), scheme)
            buf = bytearray(lm.bytesize("<"))
            lm.serialize(buf, "<")
            seed, num_perm_tag, scheme_code = struct.unpack_from("<qiB", buf, 0)
            self.assertEqual(seed, 1)
            self.assertEqual(num_perm_tag, -4)
            self.assertEqual(scheme_code, code)

    def test_unknown_scheme_code_raises(self):
        buf = bytearray(struct.calcsize("<qiB2I"))
        struct.pack_into("<qiB2I", buf, 0, 1, -2, 99, 5, 6)
        with self.assertRaises(ValueError):
            LeanMinHash.deserialize(buf, "<")

    def test_pickle_roundtrip(self):
        for scheme in ALL_SCHEMES:
            lm = self._lean(scheme)
            lm2 = pickle.loads(pickle.dumps(lm))
            self.assertEqual(lm2, lm)
            self.assertEqual(lm2.scheme, scheme)

    def test_copy(self):
        for scheme in ALL_SCHEMES:
            lm = self._lean(scheme)
            c = lm.copy()
            self.assertEqual(c, lm)
            self.assertEqual(c.scheme, scheme)

    def test_union_scheme_guard(self):
        lm32a, lm32b = self._lean("affine32"), self._lean("affine32")
        u = LeanMinHash.union(lm32a, lm32b)
        self.assertEqual(u.scheme, "affine32")
        with self.assertRaises(ValueError):
            LeanMinHash.union(lm32a, self._lean("legacy"))

    def test_cross_scheme_jaccard_refused(self):
        with self.assertRaises(ValueError):
            self._lean("affine32").jaccard(self._lean("legacy"))


class TestBBitSchemes(unittest.TestCase):
    def test_scheme_mismatch_raises(self):
        m32 = MinHash(16, 1, scheme="affine32")
        ml = MinHash(16, 1, scheme="legacy")
        with self.assertRaises(ValueError):
            bBitMinHash(m32).jaccard(bBitMinHash(ml))

    def test_pickle_roundtrip_per_scheme(self):
        for scheme in ALL_SCHEMES:
            m = MinHash(64, 1, scheme=scheme)
            m.update_batch([b"a", b"b", b"c"])
            for b in (1, 4, 32):
                bb = bBitMinHash(m, b)
                bb2 = pickle.loads(pickle.dumps(bb))
                self.assertEqual(bb, bb2, (scheme, b))
                self.assertEqual(bb2.scheme, scheme, (scheme, b))

    def test_estimation_accuracy_per_scheme(self):
        # b-bit keeps the *low* bits of each min hash value, which are the
        # softest bits of a multiply-shift (affine) output, so verify the
        # estimate stays accurate under every scheme. Two sets of size 1000
        # sharing 500 elements have true Jaccard 500 / 1500 = 1/3.
        true_j = 1.0 / 3.0
        a = [b"w%d" % i for i in range(0, 1000)]
        b_set = [b"w%d" % i for i in range(500, 1500)]
        for scheme in ALL_SCHEMES:
            ma = MinHash(num_perm=512, seed=7, scheme=scheme)
            mb = MinHash(num_perm=512, seed=7, scheme=scheme)
            ma.update_batch(a)
            mb.update_batch(b_set)
            for nbits in (1, 4, 8, 16, 32):
                est = bBitMinHash(ma, nbits).jaccard(bBitMinHash(mb, nbits))
                self.assertLess(abs(est - true_j), 0.1, (scheme, nbits))
            # Keeping all 32 bits reproduces the full MinHash estimate.
            est32 = bBitMinHash(ma, 32).jaccard(bBitMinHash(mb, 32))
            self.assertLess(abs(est32 - ma.jaccard(mb)), 0.02, scheme)

    def test_setstate_unknown_scheme_code_raises(self):
        # The pickle path (__setstate__) must reject a corrupt scheme byte,
        # like deserialize() does. The byte sits right after the fixed header:
        # "qi" (seed, negated size) for LeanMinHash, "<qBdi" (seed, b, r,
        # negated size) for bBitMinHash. Valid codes are 1/2; 99 is unknown.
        m = MinHash(16, 1, scheme="affine32")
        m.update_batch([b"a", b"b", b"c"])

        lean_buf = bytearray(LeanMinHash(m).__getstate__())
        lean_buf[struct.calcsize("qi")] = 99
        with self.assertRaises(ValueError):
            LeanMinHash.__new__(LeanMinHash).__setstate__(lean_buf)

        bbit_buf = bytearray(bBitMinHash(m, 4).__getstate__())
        bbit_buf[struct.calcsize("<qBdi")] = 99
        with self.assertRaises(ValueError):
            bBitMinHash.__new__(bBitMinHash).__setstate__(bbit_buf)


class TestLSHIntegration(unittest.TestCase):
    def _tokens(self, words):
        return [w.encode("utf8") for w in words]

    def test_minhash_lsh(self):
        base = ["minhash", "is", "a", "probabilistic", "data", "structure", "for", "estimating", "similarity"]
        near = [*base[:-1], "resemblance"]
        far = ["completely", "different", "words", "with", "no", "overlap", "at", "all", "in", "this", "set"]
        for scheme in ALL_SCHEMES:
            mq, mn, mf = (MinHash(128, scheme=scheme) for _ in range(3))
            mq.update_batch(self._tokens(base))
            mn.update_batch(self._tokens(near))
            mf.update_batch(self._tokens(far))
            lsh = MinHashLSH(threshold=0.5, num_perm=128)
            lsh.insert("near", mn)
            lsh.insert("far", mf)
            results = lsh.query(mq)
            self.assertIn("near", results, scheme)
            self.assertNotIn("far", results, scheme)

    def test_lshforest_roundtrip_hashvalues(self):
        tokens = self._tokens(["minhash", "lsh", "forest", "top", "k", "query", "jaccard", "similarity"])
        for scheme in ALL_SCHEMES:
            m = MinHash(128, scheme=scheme)
            m.update_batch(tokens)
            forest = MinHashLSHForest(num_perm=128)
            forest.add("key", m)
            forest.index()
            self.assertIn("key", forest.query(m, 1))
            hashvalues = forest.get_minhash_hashvalues("key")
            self.assertEqual(hashvalues.dtype, m.hashvalues.dtype, scheme)
            self.assertTrue(np.array_equal(hashvalues, m.hashvalues), scheme)


if __name__ == "__main__":
    unittest.main()
