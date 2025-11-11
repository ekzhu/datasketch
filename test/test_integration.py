import os

import numpy as np
import pytest

from datasketch.lsh import MinHashLSH
from datasketch.minhash import MinHash
from datasketch.weighted_minhash import WeightedMinHashGenerator

STORAGE_CONFIG_REDIS = {
    "basename": b"test_redis",
    "type": "redis",
    "redis": {"host": "localhost", "port": 6379},
}

STORAGE_CONFIG_CASSANDRA = {
    "basename": b"test_cassandra",
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


def _clear_redis_keys(pattern="test_*"):
    """Clear Redis keys matching the given pattern."""
    if not DO_TEST_REDIS:
        return
    try:
        import redis

        r = redis.Redis(host="localhost", port=6379)
        for key in r.scan_iter(match=pattern):
            r.delete(key)
    except Exception:
        pass  # Best effort cleanup


@pytest.fixture(
    params=[
        pytest.param(
            ("redis", STORAGE_CONFIG_REDIS),
            marks=pytest.mark.skipif(not DO_TEST_REDIS, reason="DO_TEST_REDIS not set"),
            id="redis",
        ),
        pytest.param(
            ("cassandra", STORAGE_CONFIG_CASSANDRA),
            marks=pytest.mark.skipif(not DO_TEST_CASSANDRA, reason="DO_TEST_CASSANDRA not set"),
            id="cassandra",
        ),
    ]
)
def storage_config(request):
    """Fixture that provides storage configurations for enabled backends."""
    backend_name, config = request.param

    # Cleanup after test
    yield backend_name, config

    # Clear Redis data after each test to avoid key conflicts
    if backend_name == "redis":
        _clear_redis_keys(f"test_{backend_name}*")


class TestMinHashLSHIntegration:
    """Integration tests for MinHashLSH with external storage backends."""

    def test_init(self, storage_config):
        """Test LSH initialization and parameter optimization."""
        _, config = storage_config
        lsh = MinHashLSH(threshold=0.8, storage_config=config, prepickle=False)
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
        backend, config = storage_config
        for _l in range(2, 128 + 1, 16):
            lsh = MinHashLSH(num_perm=128, storage_config=config, prepickle=False)
            m = MinHash()
            m.update(b"abcdefg")
            m.update(b"1234567")
            lsh.insert("m", m)
            sizes = [len(H) for ht in lsh.hashtables for H in ht]
            assert all(sizes[0] == s for s in sizes)

            if backend == "redis":
                _clear_redis_keys("test_redis*")

    def test_insert(self, storage_config):
        _, config = storage_config
        lsh = MinHashLSH(threshold=0.5, num_perm=16, storage_config=config, prepickle=False)
        m1 = MinHash(16)
        m1.update(b"a")
        m2 = MinHash(16)
        m2.update(b"b")

        lsh.insert("a", m1)
        lsh.insert("b", m2)

        for t in lsh.hashtables:
            assert len(t) >= 1
            items = []
            for H in t:
                items.extend(t[H])
            assert "a" in items
            assert "b" in items

        assert "a" in lsh
        assert "b" in lsh

        for i, H in enumerate(lsh.keys["a"]):
            assert "a" in lsh.hashtables[i][H]

    def test_query(self, storage_config):
        _, config = storage_config
        lsh = MinHashLSH(threshold=0.5, num_perm=16, storage_config=config, prepickle=False)
        m1 = MinHash(16)
        m1.update(b"a")
        m2 = MinHash(16)
        m2.update(b"b")

        lsh.insert("a", m1)
        lsh.insert("b", m2)

        # Query should return the inserted keys
        result = lsh.query(m1)
        assert "a" in result
        result = lsh.query(m2)
        assert "b" in result

        # Query with wrong num_perm should raise ValueError
        m3 = MinHash(18)
        with pytest.raises(ValueError):
            lsh.query(m3)

    def test_query_buffer(self, storage_config):
        """Test buffered query operations."""
        _, config = storage_config
        lsh = MinHashLSH(threshold=0.5, num_perm=16, storage_config=config, prepickle=False)
        m1 = MinHash(16)
        m1.update(b"a")
        m2 = MinHash(16)
        m2.update(b"b")

        lsh.insert("a", m1)
        lsh.insert("b", m2)

        # Test query buffer for m1
        lsh.add_to_query_buffer(m1)
        result = lsh.collect_query_buffer()
        assert "a" in result

        # Test query buffer for m2
        lsh.add_to_query_buffer(m2)
        result = lsh.collect_query_buffer()
        assert "b" in result

        # Invalid num_perm should raise ValueError
        m3 = MinHash(18)
        with pytest.raises(ValueError):
            lsh.add_to_query_buffer(m3)

    def test_remove(self, storage_config):
        """Test removing items from LSH index."""
        _, config = storage_config
        lsh = MinHashLSH(threshold=0.5, num_perm=16, storage_config=config, prepickle=False)
        m1 = MinHash(16)
        m1.update(b"a")
        m2 = MinHash(16)
        m2.update(b"b")

        lsh.insert("a", m1)
        lsh.insert("b", m2)

        # Remove item and verify it's gone
        lsh.remove("a")
        assert "a" not in lsh.keys

        # Check item removed from hashtables
        for table in lsh.hashtables:
            for H in table:
                assert len(table[H]) > 0
                assert "a" not in table[H]

        # Removing non-existent key should raise ValueError
        with pytest.raises(ValueError):
            lsh.remove("c")

    def test_get_subset_counts(self, storage_config):
        """Test get_subset_counts returns same results as in-memory LSH."""
        _, config = storage_config
        m1 = MinHash(16)
        m1.update(b"a")
        m2 = MinHash(16)
        m2.update(b"b")

        # Create LSH with external storage
        lsh_external = MinHashLSH(threshold=0.5, num_perm=16, storage_config=config, prepickle=False)
        lsh_external.insert("a", m1)
        lsh_external.insert("b", m2)

        # Create LSH with in-memory storage
        lsh_memory = MinHashLSH(threshold=0.5, num_perm=16)
        lsh_memory.insert("a", m1)
        lsh_memory.insert("b", m2)

        # Results should match
        assert lsh_external.get_subset_counts("a") == lsh_memory.get_subset_counts("a")
        assert lsh_external.get_subset_counts("b") == lsh_memory.get_subset_counts("b")

    def test_insertion_session(self, storage_config):
        """Test bulk insertion using insertion session."""
        _, config = storage_config
        lsh = MinHashLSH(threshold=0.5, num_perm=16, storage_config=config, prepickle=False)
        m1 = MinHash(16)
        m1.update(b"a")
        m2 = MinHash(16)
        m2.update(b"b")

        # Use insertion session for bulk insert
        data = [("a", m1), ("b", m2)]
        with lsh.insertion_session() as session:
            for key, minhash in data:
                session.insert(key, minhash)

        # Verify insertions
        for t in lsh.hashtables:
            assert len(t) >= 1
            items = []
            for H in t:
                items.extend(t[H])
            assert "a" in items
            assert "b" in items

        assert "a" in lsh
        assert "b" in lsh

        for i, H in enumerate(lsh.keys["a"]):
            assert "a" in lsh.hashtables[i][H]

    def test_deletion_session(self, storage_config):
        """Test bulk deletion using deletion session."""
        _, config = storage_config
        lsh = MinHashLSH(threshold=0.5, num_perm=16, storage_config=config, prepickle=False)
        m1 = MinHash(16)
        m1.update(b"a")
        m2 = MinHash(16)
        m2.update(b"b")
        m3 = MinHash(16)
        m3.update(b"c")

        lsh.insert("a", m1)
        lsh.insert("b", m2)
        lsh.insert("c", m3)

        # Use deletion session for bulk delete
        keys_to_delete = ["a", "b"]
        with lsh.deletion_session() as session:
            for key in keys_to_delete:
                session.remove(key)

        # Verify deletions
        assert "a" not in lsh.keys
        assert "b" not in lsh.keys
        assert "c" in lsh.keys

        for table in lsh.hashtables:
            for H in table:
                items = table[H]
                assert "a" not in items
                assert "b" not in items

    def test_get_counts(self, storage_config):
        """Test getting hashtable bucket counts."""
        _, config = storage_config
        lsh = MinHashLSH(threshold=0.5, num_perm=16, storage_config=config, prepickle=False)
        m1 = MinHash(16)
        m1.update(b"a")
        m2 = MinHash(16)
        m2.update(b"b")

        lsh.insert("a", m1)
        lsh.insert("b", m2)

        counts = lsh.get_counts()
        assert len(counts) == lsh.b
        for table in counts:
            assert sum(table.values()) == 2


class TestWeightedMinHashLSHIntegration:
    """Integration tests for Weighted MinHash with external storage backends."""

    def test_init(self, storage_config):
        """Test LSH initialization with weighted MinHash."""
        _, config = storage_config
        lsh = MinHashLSH(threshold=0.8, storage_config=config, prepickle=False)
        assert lsh.is_empty()
        b1, r1 = lsh.b, lsh.r

        lsh = MinHashLSH(threshold=0.8, weights=(0.2, 0.8), storage_config=config)
        b2, r2 = lsh.b, lsh.r
        assert b1 < b2
        assert r1 > r2

    def test_H_consistent_length(self, storage_config):
        """Check _H output has consistent bytes length for weighted MinHash."""
        backend, config = storage_config
        mg = WeightedMinHashGenerator(100, sample_size=128)
        for _l in range(2, mg.sample_size + 1, 16):
            m = mg.minhash(np.random.randint(1, 99999999, 100))
            lsh = MinHashLSH(num_perm=128, storage_config=config, prepickle=False)
            lsh.insert("m", m)
            sizes = [len(H) for ht in lsh.hashtables for H in ht]
            assert all(sizes[0] == s for s in sizes)
            # Clear Redis between iterations to avoid key conflicts
            if backend == "redis":
                _clear_redis_keys("test_redis*")

    def test_insert(self, storage_config):
        """Test inserting weighted MinHash objects."""
        _, config = storage_config
        lsh = MinHashLSH(threshold=0.5, num_perm=4, storage_config=config, prepickle=False)
        mg = WeightedMinHashGenerator(10, 4)
        m1 = mg.minhash(np.random.uniform(1, 10, 10))
        m2 = mg.minhash(np.random.uniform(1, 10, 10))

        lsh.insert("a", m1)
        lsh.insert("b", m2)

        # Verify insertions
        for t in lsh.hashtables:
            assert len(t) >= 1
            items = []
            for H in t:
                items.extend(t[H])
            assert "a" in items
            assert "b" in items

        assert "a" in lsh
        assert "b" in lsh

        for i, H in enumerate(lsh.keys["a"]):
            assert "a" in lsh.hashtables[i][H]

        # Wrong sample size should raise ValueError
        mg = WeightedMinHashGenerator(10, 5)
        m3 = mg.minhash(np.random.uniform(1, 10, 10))
        with pytest.raises(ValueError):
            lsh.insert("c", m3)

    def test_query(self, storage_config):
        """Test querying with weighted MinHash."""
        _, config = storage_config
        lsh = MinHashLSH(threshold=0.5, num_perm=4, storage_config=config, prepickle=False)
        mg = WeightedMinHashGenerator(10, 4)
        m1 = mg.minhash(np.random.uniform(1, 10, 10))
        m2 = mg.minhash(np.random.uniform(1, 10, 10))

        lsh.insert("a", m1)
        lsh.insert("b", m2)

        result = lsh.query(m1)
        assert "a" in result
        result = lsh.query(m2)
        assert "b" in result

        # Wrong sample size should raise ValueError
        mg = WeightedMinHashGenerator(10, 5)
        m3 = mg.minhash(np.random.uniform(1, 10, 10))
        with pytest.raises(ValueError):
            lsh.query(m3)

    def test_remove(self, storage_config):
        """Test removing weighted MinHash items."""
        _, config = storage_config
        lsh = MinHashLSH(threshold=0.5, num_perm=4, storage_config=config, prepickle=False)
        mg = WeightedMinHashGenerator(10, 4)
        m1 = mg.minhash(np.random.uniform(1, 10, 10))
        m2 = mg.minhash(np.random.uniform(1, 10, 10))

        lsh.insert("a", m1)
        lsh.insert("b", m2)

        lsh.remove("a")
        assert "a" not in lsh.keys

        for table in lsh.hashtables:
            for H in table:
                assert len(table[H]) > 0
                assert "a" not in table[H]

        with pytest.raises(ValueError):
            lsh.remove("c")
