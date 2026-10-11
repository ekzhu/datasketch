import pickle
import unittest
from unittest.mock import patch

import mockredis
import numpy as np
import pytest

from datasketch.lshensemble import MinHashLSHEnsemble
from datasketch.minhash import MinHash


def fake_redis(**kwargs):
    redis = mockredis.mock_redis_client(**kwargs)
    redis.connection_pool = None
    redis.response_callbacks = None
    return redis


class TestMinHashLSHEnsemble(unittest.TestCase):
    def test_init(self):
        lsh = MinHashLSHEnsemble(threshold=0.8)
        self.assertTrue(lsh.is_empty())

    def _data(self, count):
        sizes = np.random.randint(1, 100, count)
        for key, size in enumerate(sizes):
            m = MinHash()
            for i in range(size):
                m.update(("%d" % i).encode("utf8"))
            yield (key, m, size)

    def test_index(self):
        lsh = MinHashLSHEnsemble(threshold=0.8)
        lsh.index(self._data(64))
        self.assertFalse(lsh.is_empty())
        self.assertTrue(41 in lsh)

    def test_query(self):
        lsh = MinHashLSHEnsemble(threshold=0.9)
        data = list(self._data(64))
        lsh.index(data)
        for key, minhash, size in data:
            keys = lsh.query(minhash, size)
            self.assertTrue(key in keys)

    def test_pickle(self):
        lsh = MinHashLSHEnsemble(threshold=0.9)
        data = list(self._data(32))
        lsh.index(data)
        buf = pickle.dumps(lsh)
        lsh2 = pickle.loads(buf)
        for _key, minhash, size in data:
            keys1 = lsh.query(minhash, size)
            keys2 = lsh2.query(minhash, size)
            self.assertTrue(set(keys1) == set(keys2))

    def test_index_redis(self):
        storage_config = {
            "type": "redis",
            "redis": {
                "host": "localhost",
                "port": 6379,
            },
        }
        with patch("redis.Redis", fake_redis):
            lsh = MinHashLSHEnsemble(threshold=0.8, storage_config=storage_config)
            lsh.index(self._data(64))
            self.assertFalse(lsh.is_empty())
            self.assertTrue(41 in lsh)

    def test_query_redis(self):
        storage_config = {
            "type": "redis",
            "redis": {
                "host": "localhost",
                "port": 6379,
            },
        }
        with patch("redis.Redis", fake_redis):
            lsh = MinHashLSHEnsemble(threshold=0.9, storage_config=storage_config)
            data = list(self._data(64))
            lsh.index(data)
            for key, minhash, size in data:
                keys = lsh.query(minhash, size)
                self.assertTrue(key in keys)


@pytest.mark.parametrize("container", [list, tuple, lambda entries: (entry for entry in entries)])
@pytest.mark.parametrize("size", [0, -1])
@pytest.mark.parametrize("invalid_first", [True, False])
def test_index_rejects_nonpositive_sizes_without_mutating_index(container, size, invalid_first):
    minhash = MinHash(num_perm=16)
    minhash.update(b"token")
    entries = [("invalid", minhash, size), ("valid", minhash, 1)]
    if not invalid_first:
        entries.reverse()
    lsh = MinHashLSHEnsemble(num_perm=16, num_part=2, m=2)

    with pytest.raises(ValueError, match="Set size must be positive"):
        lsh.index(container(entries))

    assert lsh.is_empty()
    assert "invalid" not in lsh
    assert "valid" not in lsh
    lsh.index([("valid", minhash, 1)])
    assert list(lsh.query(minhash, 1)) == ["valid"]


@pytest.mark.parametrize("container", [list, tuple, lambda entries: (entry for entry in entries)])
def test_index_accepts_positive_sizes_for_each_container(container):
    minhash = MinHash(num_perm=16)
    minhash.update(b"token")
    lsh = MinHashLSHEnsemble(num_perm=16, num_part=2, m=2)

    lsh.index(container([("document", minhash, 1)]))

    assert not lsh.is_empty()
    assert "document" in lsh
    assert list(lsh.query(minhash, 1)) == ["document"]


if __name__ == "__main__":
    unittest.main()
