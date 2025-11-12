import os

import numpy as np
import pytest

from datasketch.lsh import MinHashLSH
from datasketch.minhash import MinHash
from datasketch.weighted_minhash import WeightedMinHashGenerator

STORAGE_CONFIG_REDIS = {
    "basename": b"lsh_test",
    "type": "redis",
    "redis": {"host": "localhost", "port": 6379},
}

STORAGE_CONFIG_CASSANDRA = {
    "basename": b"test",
    "type": "cassandra",
    "cassandra": {
        "seeds": ["127.0.0.1"],
        "keyspace": "lsh_test",
        "replication": {"class": "SimpleStrategy", "replication_factor": "1"},
        "drop_keyspace": True,
        "drop_tables": True,
    },
}

DO_TEST_REDIS = os.environ.get("DO_TEST_REDIS") == "true"
DO_TEST_CASSANDRA = os.environ.get("DO_TEST_CASSANDRA") == "true"


def _clear_redis_keys(pattern="lsh_test*"):
    if not DO_TEST_REDIS:
        return
    try:
        import redis

        r = redis.Redis(host="localhost", port=6379)
        for key in r.scan_iter(match=pattern):
            r.delete(key)
    except (ImportError, ConnectionError):
        pass  # Redis not available, skip cleanup


@pytest.fixture(
    params=[
        pytest.param(
            STORAGE_CONFIG_REDIS,
            marks=pytest.mark.skipif(not DO_TEST_REDIS, reason="DO_TEST_REDIS not set"),
            id="redis",
        ),
        pytest.param(
            STORAGE_CONFIG_CASSANDRA,
            marks=pytest.mark.skipif(not DO_TEST_CASSANDRA, reason="DO_TEST_CASSANDRA not set"),
            id="cassandra",
        ),
    ]
)
def storage_config(request):
    return request.param


class TestMinHashLSH:
    def teardown_method(self, method):
        _clear_redis_keys()

    def test_init(self, storage_config):
        lsh = MinHashLSH(threshold=0.8, storage_config=storage_config, prepickle=False)
        assert lsh.is_empty()
        b1, r1 = lsh.b, lsh.r
        lsh = MinHashLSH(threshold=0.8, weights=(0.2, 0.8))
        b2, r2 = lsh.b, lsh.r
        assert b1 < b2
        assert r1 > r2

    def test__H(self, storage_config):
        """Check _H output consistent bytes length given
        the same concatenated hash value size.
        """
        for _l in range(2, 128 + 1, 16):
            lsh = MinHashLSH(num_perm=128, storage_config=storage_config, prepickle=False)
            m = MinHash()
            m.update(b"abcdefg")
            m.update(b"1234567")
            lsh.insert(b"m", m)
            sizes = [len(H) for ht in lsh.hashtables for H in ht]
            assert all(sizes[0] == s for s in sizes)

            self.teardown_method(None)

    def test_insert(self, storage_config):
        lsh = MinHashLSH(threshold=0.5, num_perm=16, storage_config=storage_config, prepickle=False)
        m1 = MinHash(16)
        m1.update(b"a")
        m2 = MinHash(16)
        m2.update(b"b")
        lsh.insert(b"a", m1)
        lsh.insert(b"b", m2)
        for t in lsh.hashtables:
            assert len(t) >= 1
            items = []
            for H in t:
                items.extend(t[H])
            assert b"a" in items
            assert b"b" in items
        assert b"a" in lsh
        assert b"b" in lsh
        for i, H in enumerate(lsh.keys[b"a"]):
            assert b"a" in lsh.hashtables[i][H]

    def test_query(self, storage_config):
        lsh = MinHashLSH(threshold=0.5, num_perm=16, storage_config=storage_config, prepickle=False)
        m1 = MinHash(16)
        m1.update(b"a")
        m2 = MinHash(16)
        m2.update(b"b")
        lsh.insert(b"a", m1)
        lsh.insert(b"b", m2)
        result = lsh.query(m1)
        assert b"a" in result
        result = lsh.query(m2)
        assert b"b" in result
        m3 = MinHash(18)
        with pytest.raises(ValueError):
            lsh.query(m3)

    def test_query_buffer(self, storage_config):
        lsh = MinHashLSH(threshold=0.5, num_perm=16, storage_config=storage_config, prepickle=False)
        m1 = MinHash(16)
        m1.update(b"a")
        m2 = MinHash(16)
        m2.update(b"b")
        lsh.insert(b"a", m1)
        lsh.insert(b"b", m2)
        lsh.add_to_query_buffer(m1)
        result = lsh.collect_query_buffer()
        assert b"a" in result
        lsh.add_to_query_buffer(m2)
        result = lsh.collect_query_buffer()
        assert b"b" in result
        m3 = MinHash(18)
        with pytest.raises(ValueError):
            lsh.add_to_query_buffer(m3)

    def test_remove(self, storage_config):
        lsh = MinHashLSH(threshold=0.5, num_perm=16, storage_config=storage_config, prepickle=False)
        m1 = MinHash(16)
        m1.update(b"a")
        m2 = MinHash(16)
        m2.update(b"b")
        lsh.insert(b"a", m1)
        lsh.insert(b"b", m2)

        lsh.remove(b"a")
        assert b"a" not in lsh.keys
        for table in lsh.hashtables:
            for H in table:
                assert len(table[H]) > 0
                assert b"a" not in table[H]

        with pytest.raises(ValueError):
            lsh.remove(b"c")

    def test_get_subset_counts(self, storage_config):
        m1 = MinHash(16)
        m1.update(b"a")
        m2 = MinHash(16)
        m2.update(b"b")

        lsh_c = MinHashLSH(threshold=0.5, num_perm=16, storage_config=storage_config, prepickle=False)
        lsh_c.insert(b"a", m1)
        lsh_c.insert(b"b", m2)
        lsh_m = MinHashLSH(threshold=0.5, num_perm=16)
        lsh_m.insert(b"a", m1)
        lsh_m.insert(b"b", m2)

        assert lsh_c.get_subset_counts(b"a") == lsh_m.get_subset_counts(b"a")
        assert lsh_c.get_subset_counts(b"b") == lsh_m.get_subset_counts(b"b")

    def test_insertion_session(self, storage_config):
        lsh = MinHashLSH(threshold=0.5, num_perm=16, storage_config=storage_config, prepickle=False)
        m1 = MinHash(16)
        m1.update(b"a")
        m2 = MinHash(16)
        m2.update(b"b")
        data = [(b"a", m1), (b"b", m2)]
        with lsh.insertion_session() as session:
            for key, minhash in data:
                session.insert(key, minhash)
        for t in lsh.hashtables:
            assert len(t) >= 1
            items = []
            for H in t:
                items.extend(t[H])
            assert b"a" in items
            assert b"b" in items
        assert b"a" in lsh
        assert b"b" in lsh
        for i, H in enumerate(lsh.keys[b"a"]):
            assert b"a" in lsh.hashtables[i][H]

    def test_deletion_session(self, storage_config):
        lsh = MinHashLSH(threshold=0.5, num_perm=16, storage_config=storage_config, prepickle=False)
        m1 = MinHash(16)
        m1.update(b"a")
        m2 = MinHash(16)
        m2.update(b"b")
        m3 = MinHash(16)
        m3.update(b"c")
        lsh.insert(b"a", m1)
        lsh.insert(b"b", m2)
        lsh.insert(b"c", m3)

        keys_to_delete = [b"a", b"b"]
        with lsh.deletion_session() as session:
            for key in keys_to_delete:
                session.remove(key)

        assert b"a" not in lsh.keys
        assert b"b" not in lsh.keys
        assert b"c" in lsh.keys

        for table in lsh.hashtables:
            for H in table:
                items = table[H]
                assert b"a" not in items
                assert b"b" not in items

    def test_get_counts(self, storage_config):
        lsh = MinHashLSH(threshold=0.5, num_perm=16, storage_config=storage_config, prepickle=False)
        m1 = MinHash(16)
        m1.update(b"a")
        m2 = MinHash(16)
        m2.update(b"b")
        lsh.insert(b"a", m1)
        lsh.insert(b"b", m2)
        counts = lsh.get_counts()
        assert len(counts) == lsh.b
        for table in counts:
            assert sum(table.values()) == 2


class TestWeightedMinHashLSH:
    def teardown_method(self, method):
        _clear_redis_keys()

    def test_init(self, storage_config):
        lsh = MinHashLSH(threshold=0.8, storage_config=storage_config, prepickle=False)
        assert lsh.is_empty()
        b1, r1 = lsh.b, lsh.r

        lsh = MinHashLSH(threshold=0.8, weights=(0.2, 0.8), storage_config=storage_config, prepickle=False)
        b2, r2 = lsh.b, lsh.r
        assert b1 < b2
        assert r1 > r2

    def test__H(self, storage_config):
        """Check _H output consistent bytes length given
        the same concatenated hash value size.
        """
        mg = WeightedMinHashGenerator(100, sample_size=128)
        for _l in range(2, mg.sample_size + 1, 16):
            m = mg.minhash(np.random.randint(1, 99999999, 100))
            lsh = MinHashLSH(num_perm=128, storage_config=storage_config, prepickle=False)
            lsh.insert(b"m", m)
            sizes = [len(H) for ht in lsh.hashtables for H in ht]
            assert all(sizes[0] == s for s in sizes)

            self.teardown_method(None)

    def test_insert(self, storage_config):
        lsh = MinHashLSH(threshold=0.5, num_perm=4, storage_config=storage_config, prepickle=False)
        mg = WeightedMinHashGenerator(10, 4)
        m1 = mg.minhash(np.random.uniform(1, 10, 10))
        m2 = mg.minhash(np.random.uniform(1, 10, 10))
        lsh.insert(b"a", m1)
        lsh.insert(b"b", m2)
        for t in lsh.hashtables:
            assert len(t) >= 1
            items = []
            for H in t:
                items.extend(t[H])
            assert b"a" in items
            assert b"b" in items
        assert b"a" in lsh
        assert b"b" in lsh
        for i, H in enumerate(lsh.keys[b"a"]):
            assert b"a" in lsh.hashtables[i][H]

        mg = WeightedMinHashGenerator(10, 5)
        m3 = mg.minhash(np.random.uniform(1, 10, 10))
        with pytest.raises(ValueError):
            lsh.insert(b"c", m3)

    def test_query(self, storage_config):
        lsh = MinHashLSH(threshold=0.5, num_perm=4, storage_config=storage_config, prepickle=False)
        mg = WeightedMinHashGenerator(10, 4)
        m1 = mg.minhash(np.random.uniform(1, 10, 10))
        m2 = mg.minhash(np.random.uniform(1, 10, 10))
        lsh.insert(b"a", m1)
        lsh.insert(b"b", m2)
        result = lsh.query(m1)
        assert b"a" in result
        result = lsh.query(m2)
        assert b"b" in result

        mg = WeightedMinHashGenerator(10, 5)
        m3 = mg.minhash(np.random.uniform(1, 10, 10))
        with pytest.raises(ValueError):
            lsh.query(m3)

    def test_remove(self, storage_config):
        lsh = MinHashLSH(threshold=0.5, num_perm=4, storage_config=storage_config, prepickle=False)
        mg = WeightedMinHashGenerator(10, 4)
        m1 = mg.minhash(np.random.uniform(1, 10, 10))
        m2 = mg.minhash(np.random.uniform(1, 10, 10))
        lsh.insert(b"a", m1)
        lsh.insert(b"b", m2)

        lsh.remove(b"a")
        assert b"a" not in lsh.keys
        for table in lsh.hashtables:
            for H in table:
                assert len(table[H]) > 0
                assert b"a" not in table[H]

        with pytest.raises(ValueError):
            lsh.remove(b"c")


if __name__ == "__main__":
    unittest.main()
