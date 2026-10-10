import io
import pickle
import unittest
from unittest.mock import AsyncMock, Mock, patch

import mockredis
import numpy as np

from datasketch.aio import AsyncMinHashLSH
from datasketch.key_serialization import _RestrictedUnpickler, dumps_key, loads_key
from datasketch.lsh import MinHashLSH
from datasketch.minhash import MinHash
from datasketch.weighted_minhash import WeightedMinHashGenerator

backend_payload_executed = False


def execute_backend_payload():
    global backend_payload_executed
    backend_payload_executed = True
    return "compromised"


class UnsafeKey:
    def __reduce__(self):
        return execute_backend_payload, ()


def fake_redis(**kwargs):
    redis = mockredis.mock_redis_client(**kwargs)
    redis.connection_pool = None
    redis.response_callbacks = None
    return redis


def first_band_hash(data):
    return data


def second_band_hash(data):
    return data[::-1]


class TestMinHashLSH(unittest.TestCase):
    def test_init(self):
        lsh = MinHashLSH(threshold=0.8)
        self.assertTrue(lsh.is_empty())
        b1, r1 = lsh.b, lsh.r
        lsh = MinHashLSH(threshold=0.8, weights=(0.2, 0.8))
        b2, r2 = lsh.b, lsh.r
        self.assertTrue(b1 < b2)
        self.assertTrue(r1 > r2)

    def test__H(self):
        """Check _H output consistent bytes length given
        the same concatenated hash value size.
        """
        for _l in range(2, 128 + 1, 16):
            lsh = MinHashLSH(num_perm=128)
            m = MinHash()
            m.update(b"abcdefg")
            m.update(b"1234567")
            lsh.insert("m", m)
            sizes = [len(H) for ht in lsh.hashtables for H in ht]
            self.assertTrue(all(sizes[0] == s for s in sizes))

    def test_unpacking(self):
        for b in range(2, 1024 + 1):
            lsh = MinHashLSH(num_perm=b * 4, params=(b, 4))
            m = MinHash(num_perm=b * 4)
            m.update(b"abcdefg")
            m.update(b"1234567")
            lsh.insert("m", m)
            sizes = [len(H) for ht in lsh.hashtables for H in ht]
            self.assertTrue(all(sizes[0] == s for s in sizes))

    def test_insert(self):
        lsh = MinHashLSH(threshold=0.5, num_perm=16)
        m1 = MinHash(16)
        m1.update(b"a")
        m2 = MinHash(16)
        m2.update(b"b")
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
        lsh = MinHashLSH(threshold=0.5, num_perm=16)
        m1 = MinHash(16)
        m1.update(b"a")
        m2 = MinHash(16)
        m2.update(b"b")
        lsh.insert("a", m1)
        lsh.insert("b", m2)
        result = lsh.query(m1)
        self.assertTrue("a" in result)
        result = lsh.query(m2)
        self.assertTrue("b" in result)

        m3 = MinHash(18)
        self.assertRaises(ValueError, lsh.query, m3)

    def test_prepickle_round_trips_primitive_builtin_keys(self):
        lsh = MinHashLSH(threshold=0.5, num_perm=16, prepickle=True)
        minhash = MinHash(16)
        minhash.update(b"value")
        keys = ["string", 42, b"bytes", ("tuple", 3), frozenset({"set", 4}), None]
        for key in keys:
            lsh.insert(key, minhash)

        self.assertEqual(set(lsh.query(minhash)), set(keys))

    def test_prepickle_rejects_custom_key_types_before_insert(self):
        global backend_payload_executed
        backend_payload_executed = False
        lsh = MinHashLSH(threshold=0.5, num_perm=16, prepickle=True)
        minhash = MinHash(16)
        minhash.update(b"value")

        with self.assertRaisesRegex(TypeError, "primitive built-in types"):
            lsh.insert(UnsafeKey(), minhash)

        self.assertFalse(backend_payload_executed)
        self.assertEqual(len(lsh.keys), 0)

    def test_query_rejects_executable_pickle_from_backend(self):
        global backend_payload_executed
        backend_payload_executed = False
        lsh = MinHashLSH(threshold=0.5, num_perm=16, prepickle=True)
        minhash = MinHash(16)
        minhash.update(b"value")
        lsh.insert("safe", minhash)

        start, end = lsh.hashranges[0]
        band_hash = lsh._H(minhash.hashvalues[start:end])
        lsh.hashtables[0].insert(band_hash, pickle.dumps(UnsafeKey()))

        with self.assertRaisesRegex(pickle.UnpicklingError, "forbidden"):
            lsh.query(minhash)
        self.assertFalse(backend_payload_executed)

    def test_restricted_key_loader_defensive_errors(self):
        unpickler = _RestrictedUnpickler(io.BytesIO(b""))
        with self.assertRaisesRegex(pickle.UnpicklingError, "builtins.str"):
            unpickler.find_class("builtins", "str")
        with self.assertRaisesRegex(pickle.UnpicklingError, "persistent IDs"):
            unpickler.persistent_load("backend-reference")

        with self.assertRaisesRegex(pickle.UnpicklingError, "invalid serialized"):
            loads_key(b"not a pickle")
        with self.assertRaisesRegex(pickle.UnpicklingError, "not hashable"):
            loads_key(pickle.dumps(["list"]))
        with self.assertRaisesRegex(TypeError, "must be hashable"):
            dumps_key(["list"])

    def test_prepickle_rejects_structurally_unhashable_tuple(self):
        key = ([],)
        with self.assertRaisesRegex(TypeError, "must be hashable"):
            dumps_key(key)
        with self.assertRaisesRegex(pickle.UnpicklingError, "not hashable"):
            loads_key(pickle.dumps(key))

    def test_get_subset_counts_with_prepickled_key(self):
        lsh = MinHashLSH(threshold=0.5, num_perm=16, prepickle=True)
        minhash = MinHash(16)
        minhash.update(b"value")
        lsh.insert("safe", minhash)

        self.assertEqual(lsh.get_subset_counts("safe"), lsh.get_counts())

    def test_query_buffer(self):
        lsh = MinHashLSH(threshold=0.5, num_perm=16)
        m1 = MinHash(16)
        m1.update(b"a")
        m2 = MinHash(16)
        m2.update(b"b")
        lsh.insert("a", m1)
        lsh.insert("b", m2)
        lsh.add_to_query_buffer(m1)
        result = lsh.collect_query_buffer()
        self.assertTrue("a" in result)
        lsh.add_to_query_buffer(m2)
        result = lsh.collect_query_buffer()
        self.assertTrue("b" in result)
        m3 = MinHash(18)
        self.assertRaises(ValueError, lsh.add_to_query_buffer, m3)

    def test_query_buffer_matches_query_candidates(self):
        lsh = MinHashLSH(threshold=0.5, num_perm=32)
        docs = []
        for tokens in ([b"a", b"b", b"c"], [b"a", b"b", b"d"], [b"x", b"y", b"z"]):
            minhash = MinHash(num_perm=32)
            for token in tokens:
                minhash.update(token)
            docs.append(minhash)
        for key, minhash in enumerate(docs):
            lsh.insert(key, minhash)

        lsh.add_to_query_buffer(docs[0])
        buffered_result = set(lsh.collect_query_buffer())
        direct_result = set(lsh.query(docs[0]))

        self.assertEqual(buffered_result, direct_result)
        self.assertEqual(buffered_result, {0, 1})

    def test_remove(self):
        lsh = MinHashLSH(threshold=0.5, num_perm=16)
        m1 = MinHash(16)
        m1.update(b"a")
        m2 = MinHash(16)
        m2.update(b"b")
        lsh.insert("a", m1)
        lsh.insert("b", m2)

        lsh.remove("a")
        self.assertTrue("a" not in lsh.keys)
        for table in lsh.hashtables:
            for H in table:
                self.assertGreater(len(table[H]), 0)
                self.assertTrue("a" not in table[H])

        self.assertRaises(ValueError, lsh.remove, "c")

    def test_pickle(self):
        lsh = MinHashLSH(threshold=0.5, num_perm=16)
        m1 = MinHash(16)
        m1.update(b"a")
        m2 = MinHash(16)
        m2.update(b"b")
        lsh.insert("a", m1)
        lsh.insert("b", m2)
        lsh2 = pickle.loads(pickle.dumps(lsh))
        result = lsh2.query(m1)
        self.assertTrue("a" in result)
        result = lsh2.query(m2)
        self.assertTrue("b" in result)

    def test_insert_redis(self):
        with patch("redis.Redis", fake_redis):
            lsh = MinHashLSH(
                threshold=0.5,
                num_perm=16,
                storage_config={"type": "redis", "redis": {"host": "localhost", "port": 6379}},
            )
            m1 = MinHash(16)
            m1.update(b"a")
            m2 = MinHash(16)
            m2.update(b"b")
            lsh.insert("a", m1)
            lsh.insert("b", m2)
            for t in lsh.hashtables:
                self.assertTrue(len(t) >= 1)
                items = []
                for H in t:
                    items.extend(t[H])
                self.assertTrue(pickle.dumps("a") in items)
                self.assertTrue(pickle.dumps("b") in items)
            self.assertTrue("a" in lsh)
            self.assertTrue("b" in lsh)
            for i, H in enumerate(lsh.keys[pickle.dumps("a")]):
                self.assertTrue(pickle.dumps("a") in lsh.hashtables[i][H])

            m3 = MinHash(18)
            self.assertRaises(ValueError, lsh.insert, "c", m3)

    def test_query_redis(self):
        with patch("redis.Redis", fake_redis):
            lsh = MinHashLSH(
                threshold=0.5,
                num_perm=16,
                storage_config={"type": "redis", "redis": {"host": "localhost", "port": 6379}},
            )
            m1 = MinHash(16)
            m1.update(b"a")
            m2 = MinHash(16)
            m2.update(b"b")
            lsh.insert("a", m1)
            lsh.insert("b", m2)
            result = lsh.query(m1)
            self.assertTrue("a" in result)
            result = lsh.query(m2)
            self.assertTrue("b" in result)

            m3 = MinHash(18)
            self.assertRaises(ValueError, lsh.query, m3)

    def test_query_buffer_redis(self):
        with patch("redis.Redis", fake_redis):
            lsh = MinHashLSH(
                threshold=0.5,
                num_perm=16,
                storage_config={"type": "redis", "redis": {"host": "localhost", "port": 6379}},
            )
            m1 = MinHash(16)
            m1.update(b"a")
            m2 = MinHash(16)
            m2.update(b"b")
            lsh.insert("a", m1)
            lsh.insert("b", m2)
            lsh.query(m1)
            lsh.add_to_query_buffer(m1)
            result = lsh.collect_query_buffer()
            self.assertTrue("a" in result)
            lsh.add_to_query_buffer(m2)
            result = lsh.collect_query_buffer()
            self.assertTrue("b" in result)

            m3 = MinHash(18)
            self.assertRaises(ValueError, lsh.add_to_query_buffer, m3)

    def test_insertion_session(self):
        lsh = MinHashLSH(threshold=0.5, num_perm=16)
        m1 = MinHash(16)
        m1.update(b"a")
        m2 = MinHash(16)
        m2.update(b"b")
        data = [("a", m1), ("b", m2)]
        with lsh.insertion_session() as session:
            for key, minhash in data:
                session.insert(key, minhash)
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

    def test_deletion_session(self):
        lsh = MinHashLSH(threshold=0.5, num_perm=16)
        m1 = MinHash(16)
        m1.update(b"a")
        m2 = MinHash(16)
        m2.update(b"b")
        m3 = MinHash(16)
        m3.update(b"c")
        lsh.insert("a", m1)
        lsh.insert("b", m2)
        lsh.insert("c", m3)

        keys_to_delete = ["a", "b"]
        with lsh.deletion_session() as session:
            for key in keys_to_delete:
                session.remove(key)

        # Verify deletions
        self.assertTrue("a" not in lsh.keys)
        self.assertTrue("b" not in lsh.keys)
        self.assertTrue("c" in lsh.keys)

        for table in lsh.hashtables:
            for H in table:
                self.assertTrue("a" not in table[H])
                self.assertTrue("b" not in table[H])

    def test_get_counts(self):
        lsh = MinHashLSH(threshold=0.5, num_perm=16)
        m1 = MinHash(16)
        m1.update(b"a")
        m2 = MinHash(16)
        m2.update(b"b")
        lsh.insert("a", m1)
        lsh.insert("b", m2)
        counts = lsh.get_counts()
        self.assertEqual(len(counts), lsh.b)
        for table in counts:
            self.assertEqual(sum(table.values()), 2)

    def test_merge(self):
        lsh1 = MinHashLSH(threshold=0.5, num_perm=16)
        m1 = MinHash(16)
        m1.update(b"a")
        m2 = MinHash(16)
        m2.update(b"b")
        lsh1.insert("a", m1)
        lsh1.insert("b", m2)

        lsh2 = MinHashLSH(threshold=0.5, num_perm=16)
        m3 = MinHash(16)
        m3.update(b"c")
        m4 = MinHash(16)
        m4.update(b"d")
        lsh2.insert("c", m1)
        lsh2.insert("d", m2)

        lsh1.merge(lsh2)
        for t in lsh1.hashtables:
            self.assertTrue(len(t) >= 1)
            items = []
            for H in t:
                items.extend(t[H])
            self.assertTrue("c" in items)
            self.assertTrue("d" in items)
        self.assertTrue("a" in lsh1)
        self.assertTrue("b" in lsh1)
        self.assertTrue("c" in lsh1)
        self.assertTrue("d" in lsh1)
        for i, H in enumerate(lsh1.keys["c"]):
            self.assertTrue("c" in lsh1.hashtables[i][H])

        self.assertTrue(lsh1.merge, lsh2)
        self.assertRaises(ValueError, lsh1.merge, lsh2, check_overlap=True)

        m5 = MinHash(16)
        m5.update(b"e")
        lsh3 = MinHashLSH(threshold=0.5, num_perm=16)
        lsh3.insert("a", m5)

        self.assertRaises(ValueError, lsh1.merge, lsh3, check_overlap=True)

        lsh1.merge(lsh3)

        m6 = MinHash(16)
        m6.update(b"e")
        lsh4 = MinHashLSH(threshold=0.5, num_perm=16)
        lsh4.insert("a", m6)

        lsh1.merge(lsh4, check_overlap=False)

    def test_merge_rejects_different_band_hash_functions(self):
        minhash = MinHash(16)
        minhash.update(b"value")
        destination = MinHashLSH(threshold=0.5, num_perm=16, hashfunc=first_band_hash)
        source = MinHashLSH(threshold=0.5, num_perm=16, hashfunc=second_band_hash)
        source.insert("source", minhash)

        with self.assertRaisesRegex(ValueError, "different initialization parameters"):
            destination.merge(source)
        self.assertNotIn("source", destination)

    def test_merge_rejects_different_key_serialization(self):
        minhash = MinHash(16)
        minhash.update(b"value")
        destination = MinHashLSH(threshold=0.5, num_perm=16, prepickle=False)
        source = MinHashLSH(threshold=0.5, num_perm=16, prepickle=True)
        source.insert("source", minhash)

        with self.assertRaisesRegex(ValueError, "different initialization parameters"):
            destination.merge(source)
        self.assertNotIn("source", destination)

    def test_merge_redis(self):
        with patch("redis.Redis", fake_redis):
            lsh1 = MinHashLSH(
                threshold=0.5,
                num_perm=16,
                storage_config={"type": "redis", "redis": {"host": "localhost", "port": 6379}},
            )
            lsh2 = MinHashLSH(
                threshold=0.5,
                num_perm=16,
                storage_config={"type": "redis", "redis": {"host": "localhost", "port": 6379}},
            )

            m1 = MinHash(16)
            m1.update(b"a")
            m2 = MinHash(16)
            m2.update(b"b")
            lsh1.insert("a", m1)
            lsh1.insert("b", m2)

            m3 = MinHash(16)
            m3.update(b"c")
            m4 = MinHash(16)
            m4.update(b"d")
            lsh2.insert("c", m3)
            lsh2.insert("d", m4)

            lsh1.merge(lsh2)
            for t in lsh1.hashtables:
                self.assertTrue(len(t) >= 1)
                items = []
                for H in t:
                    items.extend(t[H])
                self.assertTrue(pickle.dumps("c") in items)
                self.assertTrue(pickle.dumps("d") in items)
            self.assertTrue("a" in lsh1)
            self.assertTrue("b" in lsh1)
            self.assertTrue("c" in lsh1)
            self.assertTrue("d" in lsh1)
            for i, H in enumerate(lsh1.keys[pickle.dumps("c")]):
                self.assertTrue(pickle.dumps("c") in lsh1.hashtables[i][H])

            self.assertTrue(lsh1.merge, lsh2)
            self.assertRaises(ValueError, lsh1.merge, lsh2, check_overlap=True)

            m5 = MinHash(16)
            m5.update(b"e")
            lsh3 = MinHashLSH(
                threshold=0.5,
                num_perm=16,
                storage_config={"type": "redis", "redis": {"host": "localhost", "port": 6379}},
            )
            lsh3.insert("a", m5)

            self.assertRaises(ValueError, lsh1.merge, lsh3, check_overlap=True)

            m6 = MinHash(16)
            m6.update(b"e")
            lsh4 = MinHashLSH(
                threshold=0.5,
                num_perm=16,
                storage_config={"type": "redis", "redis": {"host": "localhost", "port": 6379}},
            )
            lsh4.insert("a", m6)

            lsh1.merge(lsh4, check_overlap=False)

    def test_redis_deletion_session(self):
        with patch("redis.Redis", fake_redis):
            lsh = MinHashLSH(
                threshold=0.5,
                num_perm=16,
                storage_config={"type": "redis", "redis": {"host": "localhost", "port": 6379}},
            )

            # The library's RedisBuffer is not completely compatible with mockredis.
            # In order to account for this, we:
            # 1. Replace the buffer with mockredis's own simple pipeline, which can execute.
            # 2. Patch __init__ to prevent the buffer flush from resetting the storage object
            #    and wiping mock data.
            storage_objects = [lsh.keys, *lsh.hashtables]
            for storage in storage_objects:
                storage._buffer = storage._redis.pipeline()

            m1 = MinHash(16)
            m1.update(b"a")
            m2 = MinHash(16)
            m2.update(b"b")
            m3 = MinHash(16)
            m3.update(b"c")
            lsh.insert("a", m1)
            lsh.insert("b", m2)
            lsh.insert("c", m3)

            keys_to_delete = ["a", "b"]

            with (
                patch.object(type(lsh.keys), "__init__", lambda self, config, name: None),
                lsh.deletion_session() as session,
            ):
                for key in keys_to_delete:
                    session.remove(key)

            # Verify deletions
            self.assertTrue("a" not in lsh)
            self.assertTrue("b" not in lsh)
            self.assertTrue("c" in lsh)

            # Verify underlying storage
            for table in lsh.hashtables:
                for H in table:
                    items = table[H]
                    self.assertTrue("a".encode("utf8") not in items)
                    self.assertTrue("b".encode("utf8") not in items)


class TestWeightedMinHashLSH(unittest.TestCase):
    def test_init(self):
        lsh = MinHashLSH(threshold=0.8)
        self.assertTrue(lsh.is_empty())
        b1, r1 = lsh.b, lsh.r
        lsh = MinHashLSH(threshold=0.8, weights=(0.2, 0.8))
        b2, r2 = lsh.b, lsh.r
        self.assertTrue(b1 < b2)
        self.assertTrue(r1 > r2)

    def test__H(self):
        """Check _H output consistent bytes length given
        the same concatenated hash value size.
        """
        mg = WeightedMinHashGenerator(100, sample_size=128)
        for _l in range(2, mg.sample_size + 1, 16):
            m = mg.minhash(np.random.randint(1, 99999999, 100))
            lsh = MinHashLSH(num_perm=128)
            lsh.insert("m", m)
            sizes = [len(H) for ht in lsh.hashtables for H in ht]
            self.assertTrue(all(sizes[0] == s for s in sizes))

    def test_insert(self):
        lsh = MinHashLSH(threshold=0.5, num_perm=4)
        mg = WeightedMinHashGenerator(10, 4)
        m1 = mg.minhash(np.random.uniform(1, 10, 10))
        m2 = mg.minhash(np.random.uniform(1, 10, 10))
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

        mg = WeightedMinHashGenerator(10, 5)
        m3 = mg.minhash(np.random.uniform(1, 10, 10))
        self.assertRaises(ValueError, lsh.insert, "c", m3)

    def test_query(self):
        lsh = MinHashLSH(threshold=0.5, num_perm=4)
        mg = WeightedMinHashGenerator(10, 4)
        m1 = mg.minhash(np.random.uniform(1, 10, 10))
        m2 = mg.minhash(np.random.uniform(1, 10, 10))
        lsh.insert("a", m1)
        lsh.insert("b", m2)
        result = lsh.query(m1)
        self.assertTrue("a" in result)
        result = lsh.query(m2)
        self.assertTrue("b" in result)

        mg = WeightedMinHashGenerator(10, 5)
        m3 = mg.minhash(np.random.uniform(1, 10, 10))
        self.assertRaises(ValueError, lsh.query, m3)

    def test_remove(self):
        lsh = MinHashLSH(threshold=0.5, num_perm=4)
        mg = WeightedMinHashGenerator(10, 4)
        m1 = mg.minhash(np.random.uniform(1, 10, 10))
        m2 = mg.minhash(np.random.uniform(1, 10, 10))
        lsh.insert("a", m1)
        lsh.insert("b", m2)

        lsh.remove("a")
        self.assertTrue("a" not in lsh.keys)
        for table in lsh.hashtables:
            for H in table:
                self.assertGreater(len(table[H]), 0)
                self.assertTrue("a" not in table[H])

        self.assertRaises(ValueError, lsh.remove, "c")

    def test_pickle(self):
        lsh = MinHashLSH(threshold=0.5, num_perm=4)
        mg = WeightedMinHashGenerator(10, 4)
        m1 = mg.minhash(np.random.uniform(1, 10, 10))
        m2 = mg.minhash(np.random.uniform(1, 10, 10))
        lsh.insert("a", m1)
        lsh.insert("b", m2)
        lsh2 = pickle.loads(pickle.dumps(lsh))

        result = lsh2.query(m1)
        self.assertTrue("a" in result)
        result = lsh2.query(m2)
        self.assertTrue("b" in result)


class TestAsyncMinHashLSHKeySerialization(unittest.IsolatedAsyncioTestCase):
    async def test_prepickle_paths_use_restricted_serialization(self):
        minhash = MinHash(16)
        minhash.update(b"value")
        lsh = AsyncMinHashLSH(
            num_perm=16,
            params=(4, 4),
            storage_config={"type": "aiomongo"},
            prepickle=True,
        )
        serialized_key = dumps_key("safe")
        band_hashes = [lsh._H(minhash.hashvalues[start:end]) for start, end in lsh.hashranges]
        lsh.keys = Mock(
            has_key=AsyncMock(return_value=True),
            insert=AsyncMock(),
            get=AsyncMock(return_value=band_hashes),
            getmany=AsyncMock(return_value=[band_hashes]),
            remove=AsyncMock(),
        )
        lsh.hashtables = [
            Mock(
                insert=AsyncMock(),
                get=AsyncMock(return_value=[serialized_key]),
                has_key=AsyncMock(return_value=True),
                remove_val=AsyncMock(),
                remove=AsyncMock(),
            )
            for _ in range(lsh.b)
        ]

        await lsh._insert("safe", minhash, check_duplication=False)
        self.assertEqual(await lsh.query(minhash), ["safe"])
        self.assertTrue(await lsh.has_key("safe"))
        self.assertEqual(await lsh._query_b(minhash, lsh.b), {"safe"})
        self.assertEqual(len(await lsh.get_subset_counts("safe")), lsh.b)

        for hashtable in lsh.hashtables:
            hashtable.get.return_value = []
        await lsh.remove("safe")
        lsh.keys.remove.assert_awaited_once_with(serialized_key, buffer=False)


if __name__ == "__main__":
    unittest.main()
