import asyncio
import os
import pickle
import random
import string
from itertools import chain, islice

import numpy as np
import pytest
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient

from datasketch.aio import AsyncMinHashLSH
from datasketch.minhash import MinHash
from datasketch.weighted_minhash import WeightedMinHashGenerator

STORAGE_CONFIG_MONGO = {"type": "aiomongo"}
MONGO_URL = os.environ.get("MONGO_UNIT_TEST_URL")
if MONGO_URL:
    STORAGE_CONFIG_MONGO["mongo"] = {"url": MONGO_URL}
else:
    STORAGE_CONFIG_MONGO["mongo"] = {
        "host": "localhost",
        "port": 27017,
        "db": "lsh_test",
    }

STORAGE_CONFIG_REDIS = {
    "basename": b"async_lsh_test",
    "type": "aioredis",
    "redis": {"host": "localhost", "port": 6379},
    "prepickle": False,
}

DO_TEST_MONGO = os.environ.get("DO_TEST_MONGO") == "true"
DO_TEST_REDIS = os.environ.get("DO_TEST_REDIS") == "true"


def _clear_mongo():
    dsn = MONGO_URL or "mongodb://{host}:{port}".format(**STORAGE_CONFIG_MONGO["mongo"])
    MongoClient(dsn).drop_database(STORAGE_CONFIG_MONGO["mongo"]["db"])


async def _clear_redis():
    import redis.asyncio as redis

    r = redis.Redis(host="localhost", port=6379)
    async for key in r.scan_iter(match="async_lsh_test*"):
        await r.delete(key)
    await r.aclose()


@pytest.fixture(
    params=[
        pytest.param(
            STORAGE_CONFIG_MONGO,
            marks=pytest.mark.skipif(not DO_TEST_MONGO, reason="DO_TEST_MONGO not set"),
            id="mongo",
        ),
        pytest.param(
            STORAGE_CONFIG_REDIS,
            marks=pytest.mark.skipif(not DO_TEST_REDIS, reason="DO_TEST_REDIS not set"),
            id="redis",
        ),
    ]
)
def storage_config(request):
    return request.param


class TestAsyncMinHashLSH:
    @pytest.fixture(autouse=True)
    async def cleanup(self, storage_config):
        yield
        if storage_config["type"] == "aiomongo":
            _clear_mongo()
        elif storage_config["type"] == "aioredis":
            await _clear_redis()

    async def test_init(self, storage_config):
        async with AsyncMinHashLSH(storage_config=storage_config, threshold=0.8, prepickle=False) as lsh:
            assert await lsh.is_empty()
            b1, r1 = lsh.b, lsh.r

        async with AsyncMinHashLSH(
            storage_config=storage_config,
            threshold=0.8,
            weights=(0.2, 0.8),
            prepickle=False,
        ) as lsh:
            b2, r2 = lsh.b, lsh.r
        assert b1 < b2
        assert r1 > r2

    async def test__H(self, storage_config):
        for _l in range(2, 128 + 1, 16):
            m = MinHash()
            m.update(b"abcdefg")
            m.update(b"1234567")
            async with AsyncMinHashLSH(storage_config=storage_config, num_perm=128, prepickle=False) as lsh:
                await lsh.insert(b"m", m)
                sizes = []
                for ht in lsh.hashtables:
                    keys = await ht.keys()
                    for H in keys:
                        sizes.append(len(H))
                assert all(sizes[0] == s for s in sizes)

            if storage_config["type"] == "aioredis":
                await _clear_redis()

    async def test_insert(self, storage_config):
        async with AsyncMinHashLSH(storage_config=storage_config, threshold=0.5, num_perm=16, prepickle=False) as lsh:
            seq = [
                b"aahhb",
                b"aahh",
                b"aahhc",
                b"aac",
                b"kld",
                b"bhg",
                b"kkd",
                b"yow",
                b"ppi",
                b"eer",
            ]
            objs = [MinHash(16) for _ in range(len(seq))]
            for e, obj in zip(seq, objs):
                for i in e:
                    obj.update(bytes([i]))

            data = [(e, m) for e, m in zip(seq, objs)]
            for key, minhash in data:
                await lsh.insert(key, minhash)
            for t in lsh.hashtables:
                assert await t.size() >= 1
                items = []
                for H in await t.keys():
                    items.extend(await t.get(H))
                assert b"aahh" in items
                assert b"bhg" in items
            assert await lsh.has_key(b"aahh")
            assert await lsh.has_key(b"bhg")
            for i, H in enumerate(await lsh.keys.get(b"aahhb")):
                assert b"aahhb" in await lsh.hashtables[i].get(H)

            m3 = MinHash(18)
            with pytest.raises(ValueError):
                await lsh.insert(b"c", m3)

    async def test_insert_non_bytes_key_raises_error(self, storage_config):
        """Test that inserting non-bytes keys with prepickle=False raises TypeError."""
        async with AsyncMinHashLSH(storage_config=storage_config, threshold=0.5, num_perm=16, prepickle=False) as lsh:
            m1 = MinHash(16)
            m1.update(b"a")

            # Should raise TypeError when trying to insert with string key
            with pytest.raises(TypeError):
                await lsh.insert("string_key", m1)

            # Should raise TypeError when trying to insert with int key
            with pytest.raises(TypeError):
                await lsh.insert(123, m1)

    async def test_query(self, storage_config):
        async with AsyncMinHashLSH(storage_config=storage_config, threshold=0.5, num_perm=16, prepickle=False) as lsh:
            m1 = MinHash(16)
            m1.update(b"a")
            m2 = MinHash(16)
            m2.update(b"b")
            m3 = MinHash(16)
            m3.update(b"b")
            fs = (
                lsh.insert(b"a", m1, check_duplication=False),
                lsh.insert(b"b", m2, check_duplication=False),
                lsh.insert(b"b", m3, check_duplication=False),
            )
            await asyncio.gather(*fs)
            result = await lsh.query(m1)
            assert b"a" in result
            result = await lsh.query(m2)
            assert b"b" in result

            m3 = MinHash(18)
            with pytest.raises(ValueError):
                await lsh.query(m3)

    async def test_remove(self, storage_config):
        async with AsyncMinHashLSH(storage_config=storage_config, threshold=0.5, num_perm=16, prepickle=False) as lsh:
            m1 = MinHash(16)
            m1.update(b"a")
            m2 = MinHash(16)
            m2.update(b"b")
            m3 = MinHash(16)
            m3.update(b"a")
            await lsh.insert(b"a", m1)
            await lsh.insert(b"b", m2)
            await lsh.insert(b"a1", m3)

            await lsh.remove(b"a")
            assert not await lsh.has_key(b"a")
            assert await lsh.has_key(b"a1")
            hashtable_correct = False
            for table in lsh.hashtables:
                for H in await table.keys():
                    table_vals = await table.get(H)
                    assert len(table_vals) > 0
                    assert b"a" not in table_vals
                    if b"a1" in table_vals:
                        hashtable_correct = True
            assert hashtable_correct

            with pytest.raises(ValueError):
                await lsh.remove(b"c")

    async def test_pickle(self, storage_config):
        async with AsyncMinHashLSH(storage_config=storage_config, threshold=0.5, num_perm=16, prepickle=False) as lsh:
            m1 = MinHash(16)
            m1.update(b"a")
            m2 = MinHash(16)
            m2.update(b"b")
            await lsh.insert(b"a", m1)
            await lsh.insert(b"b", m2)
            pickled = pickle.dumps(lsh)

        async with pickle.loads(pickled) as lsh2:
            result = await lsh2.query(m1)
            assert b"a" in result
            result = await lsh2.query(m2)
            assert b"b" in result
            await lsh2.close()

    async def test_insertion_session(self, storage_config):
        def chunk(it, size):
            it = iter(it)
            return iter(lambda: tuple(islice(it, size)), ())

        _chunked_str = chunk((random.choice(string.ascii_lowercase) for _ in range(10000)), 4)
        seq = frozenset(
            chain(
                ("".join(s).encode() for s in _chunked_str),
                (
                    b"aahhb",
                    b"aahh",
                    b"aahhc",
                    b"aac",
                    b"kld",
                    b"bhg",
                    b"kkd",
                    b"yow",
                    b"ppi",
                    b"eer",
                ),
            )
        )
        objs = [MinHash(16) for _ in range(len(seq))]
        for e, obj in zip(seq, objs):
            for i in e:
                obj.update(bytes([i]))

        data = [(e, m) for e, m in zip(seq, objs)]

        async with AsyncMinHashLSH(storage_config=storage_config, threshold=0.5, num_perm=16, prepickle=False) as lsh:
            async with lsh.insertion_session(batch_size=1000) as session:
                fs = (session.insert(key, minhash, check_duplication=False) for key, minhash in data)
                await asyncio.gather(*fs)

            for t in lsh.hashtables:
                assert await t.size() >= 1
                items = []
                for H in await t.keys():
                    items.extend(await t.get(H))
                assert b"aahhb" in items
                assert b"kld" in items
            assert await lsh.has_key(b"aahhb")
            assert await lsh.has_key(b"kld")
            for i, H in enumerate(await lsh.keys.get(b"aahh")):
                assert b"aahh" in await lsh.hashtables[i].get(H)

    async def test_remove_session(self, storage_config):
        def chunk(it, size):
            it = iter(it)
            return iter(lambda: tuple(islice(it, size)), ())

        _chunked_str = chunk((random.choice(string.ascii_lowercase) for _ in range(10000)), 4)
        seq = frozenset(
            chain(
                ("".join(s).encode() for s in _chunked_str),
                (
                    b"aahhb",
                    b"aahh",
                    b"aahhc",
                    b"aac",
                    b"kld",
                    b"bhg",
                    b"kkd",
                    b"yow",
                    b"ppi",
                    b"eer",
                ),
            )
        )
        objs = [MinHash(16) for _ in range(len(seq))]
        for e, obj in zip(seq, objs):
            for i in e:
                obj.update(bytes([i]))

        data = [(e, m) for e, m in zip(seq, objs)]
        keys_to_remove = (
            b"aahhb",
            b"aahh",
            b"aahhc",
            b"aac",
            b"kld",
            b"bhg",
            b"kkd",
            b"yow",
            b"ppi",
            b"eer",
        )
        keys_left = frozenset(seq) - frozenset(keys_to_remove)

        async with AsyncMinHashLSH(storage_config=storage_config, threshold=0.5, num_perm=16, prepickle=False) as lsh:
            async with lsh.insertion_session(batch_size=1000) as session:
                fs = (session.insert(key, minhash, check_duplication=False) for key, minhash in data)
                await asyncio.gather(*fs)

            async with lsh.delete_session(batch_size=3) as session:
                fs = (session.remove(key) for key in keys_to_remove)
                await asyncio.gather(*fs)

            for t in lsh.hashtables:
                assert await t.size() >= 1
                items = []
                for H in await t.keys():
                    items.extend(await t.get(H))
                for key in keys_to_remove:
                    assert key not in items
                for key in keys_left:
                    assert key in items

            for key in keys_to_remove:
                assert not (await lsh.has_key(key))
            for key in keys_left:
                assert await lsh.has_key(key)

    async def test_get_counts(self, storage_config):
        async with AsyncMinHashLSH(storage_config=storage_config, threshold=0.5, num_perm=16, prepickle=False) as lsh:
            m1 = MinHash(16)
            m1.update(b"a")
            m2 = MinHash(16)
            m2.update(b"b")
            await lsh.insert(b"a", m1)
            await lsh.insert(b"b", m2)
            counts = await lsh.get_counts()
            assert len(counts) == lsh.b
            for table in counts:
                assert sum(table.values()) == 2

    async def test_get_subset_counts(self, storage_config):
        """Test get_subset_counts which uses the getmany() method."""
        async with AsyncMinHashLSH(storage_config=storage_config, threshold=0.5, num_perm=16, prepickle=False) as lsh:
            m1 = MinHash(16)
            m1.update(b"a")
            m2 = MinHash(16)
            m2.update(b"b")
            m3 = MinHash(16)
            m3.update(b"c")
            await lsh.insert(b"a", m1)
            await lsh.insert(b"b", m2)
            await lsh.insert(b"c", m3)

            # Test get_subset_counts with a subset of keys
            subset_counts = await lsh.get_subset_counts(b"a", b"b")
            assert len(subset_counts) == lsh.b
            for table in subset_counts:
                assert sum(table.values()) == 2

            # Test with all keys
            all_counts = await lsh.get_subset_counts(b"a", b"b", b"c")
            assert len(all_counts) == lsh.b
            for table in all_counts:
                assert sum(table.values()) == 3

            # Test with single key
            single_counts = await lsh.get_subset_counts(b"a")
            assert len(single_counts) == lsh.b
            for table in single_counts:
                assert sum(table.values()) == 1

    @pytest.mark.skipif(not DO_TEST_MONGO, reason="MongoDB-specific test")
    async def test_arbitrary_url(self):
        config = {
            "type": "aiomongo",
            "mongo": {"url": MONGO_URL or "mongodb://localhost/lsh_test"},
        }
        async with AsyncMinHashLSH(storage_config=config, threshold=0.5, num_perm=16) as lsh:
            m1 = MinHash(16)
            m1.update(b"a")
            await lsh.insert(b"a", m1)

        database = AsyncIOMotorClient(config["mongo"]["url"]).get_default_database("lsh_test")
        collection_names = await database.list_collection_names()
        assert len(collection_names) > 0
        await database.client.drop_database(database.name)

    @pytest.mark.skipif(not DO_TEST_MONGO, reason="MongoDB-specific test")
    async def test_arbitrary_collection(self):
        config = STORAGE_CONFIG_MONGO.copy()
        config["mongo"] = config["mongo"].copy()
        config["mongo"]["collection_name"] = "unit_test_collection"
        async with AsyncMinHashLSH(storage_config=config, threshold=0.5, num_perm=16) as lsh:
            m1 = MinHash(16)
            m1.update(b"a")
            await lsh.insert(b"a", m1)

        dsn = MONGO_URL or "mongodb://{host}:{port}/{db}".format(**config["mongo"])
        collection = AsyncIOMotorClient(dsn).get_default_database("lsh_test").get_collection("unit_test_collection")
        count = await collection.count_documents({})

        assert count >= 1
        _clear_mongo()


class TestWeightedMinHashLSH:
    @pytest.fixture(autouse=True)
    async def cleanup(self, storage_config):
        yield
        if storage_config["type"] == "aiomongo":
            _clear_mongo()
        elif storage_config["type"] == "aioredis":
            await _clear_redis()

    async def test_init(self, storage_config):
        async with AsyncMinHashLSH(storage_config=storage_config, threshold=0.8, prepickle=False) as lsh:
            assert await lsh.is_empty()
            b1, r1 = lsh.b, lsh.r
        async with AsyncMinHashLSH(
            storage_config=storage_config,
            threshold=0.8,
            weights=(0.2, 0.8),
            prepickle=False,
        ) as lsh:
            b2, r2 = lsh.b, lsh.r
        assert b1 < b2
        assert r1 > r2

    async def test__H(self, storage_config):
        mg = WeightedMinHashGenerator(100, sample_size=128)
        for _l in range(2, mg.sample_size + 1, 16):
            m = mg.minhash(np.random.randint(1, 99999999, 100))
            async with AsyncMinHashLSH(storage_config=storage_config, num_perm=128, prepickle=False) as lsh:
                await lsh.insert(b"m", m)
                fs = (ht.keys() for ht in lsh.hashtables)
                hashtables = await asyncio.gather(*fs)
                sizes = [len(H) for H in hashtables]
                assert all(sizes[0] == s for s in sizes)

            if storage_config["type"] == "aioredis":
                await _clear_redis()

    async def test_insert(self, storage_config):
        async with AsyncMinHashLSH(storage_config=storage_config, threshold=0.5, num_perm=4, prepickle=False) as lsh:
            mg = WeightedMinHashGenerator(10, 4)
            m1 = mg.minhash(np.random.uniform(1, 10, 10))
            m2 = mg.minhash(np.random.uniform(1, 10, 10))
            await lsh.insert(b"a", m1)
            await lsh.insert(b"b", m2)
            for t in lsh.hashtables:
                assert await t.size() >= 1
                items = []
                for H in await t.keys():
                    items.extend(await t.get(H))
                assert b"a" in items
                assert b"b" in items
            assert await lsh.has_key(b"a")
            assert await lsh.has_key(b"b")
            for i, H in enumerate(await lsh.keys.get(b"a")):
                assert b"a" in await lsh.hashtables[i].get(H)

            mg = WeightedMinHashGenerator(10, 5)
            m3 = mg.minhash(np.random.uniform(1, 10, 10))
            with pytest.raises(ValueError):
                await lsh.insert(b"c", m3)

    async def test_query(self, storage_config):
        async with AsyncMinHashLSH(storage_config=storage_config, threshold=0.5, num_perm=4, prepickle=False) as lsh:
            mg = WeightedMinHashGenerator(10, 4)
            m1 = mg.minhash(np.random.uniform(1, 10, 10))
            m2 = mg.minhash(np.random.uniform(1, 10, 10))
            await lsh.insert(b"a", m1)
            await lsh.insert(b"b", m2)
            result = await lsh.query(m1)
            assert b"a" in result
            result = await lsh.query(m2)
            assert b"b" in result

            mg = WeightedMinHashGenerator(10, 5)
            m3 = mg.minhash(np.random.uniform(1, 10, 10))
            with pytest.raises(ValueError):
                await lsh.query(m3)

    async def test_remove(self, storage_config):
        async with AsyncMinHashLSH(storage_config=storage_config, threshold=0.5, num_perm=4, prepickle=False) as lsh:
            mg = WeightedMinHashGenerator(10, 4)
            m1 = mg.minhash(np.random.uniform(1, 10, 10))
            m2 = mg.minhash(np.random.uniform(1, 10, 10))
            await lsh.insert(b"a", m1)
            await lsh.insert(b"b", m2)

            await lsh.remove(b"a")
            assert not await lsh.has_key(b"a")
            for table in lsh.hashtables:
                for H in await table.keys():
                    table_vals = await table.get(H)
                    assert len(table_vals) > 0
                    assert b"a" not in table_vals

            with pytest.raises(ValueError):
                await lsh.remove(b"c")

    async def test_pickle(self, storage_config):
        async with AsyncMinHashLSH(storage_config=storage_config, threshold=0.5, num_perm=4, prepickle=False) as lsh:
            mg = WeightedMinHashGenerator(10, 4)
            m1 = mg.minhash(np.random.uniform(1, 10, 10))
            m2 = mg.minhash(np.random.uniform(1, 10, 10))
            await lsh.insert(b"a", m1)
            await lsh.insert(b"b", m2)
            pickled = pickle.dumps(lsh)

        async with pickle.loads(pickled) as lsh2:
            result = await lsh2.query(m1)
            assert b"a" in result
            result = await lsh2.query(m2)
            assert b"b" in result
            await lsh2.close()


class TestAsyncMinHashLSHWithPrepickle:
    """Test AsyncMinHashLSH with prepickle=True to verify string keys work."""

    @pytest.fixture(autouse=True)
    async def cleanup(self, storage_config):
        yield
        if storage_config["type"] == "aiomongo":
            _clear_mongo()
        elif storage_config["type"] == "aioredis":
            await _clear_redis()

    async def test_insert_query_with_string_keys(self, storage_config):
        """Test that string keys work with prepickle=True."""
        async with AsyncMinHashLSH(storage_config=storage_config, threshold=0.5, num_perm=16, prepickle=True) as lsh:
            m1 = MinHash(16)
            m1.update(b"a")
            m2 = MinHash(16)
            m2.update(b"b")

            # String keys should work with prepickle=True
            await lsh.insert("string_key_1", m1)
            await lsh.insert("string_key_2", m2)

            result = await lsh.query(m1)
            assert "string_key_1" in result

            result = await lsh.query(m2)
            assert "string_key_2" in result

    async def test_insert_with_int_keys(self, storage_config):
        """Test that integer keys work with prepickle=True."""
        async with AsyncMinHashLSH(storage_config=storage_config, threshold=0.5, num_perm=16, prepickle=True) as lsh:
            m1 = MinHash(16)
            m1.update(b"a")
            m2 = MinHash(16)
            m2.update(b"b")

            # Integer keys should work with prepickle=True
            await lsh.insert(123, m1)
            await lsh.insert(456, m2)

            result = await lsh.query(m1)
            assert 123 in result

            result = await lsh.query(m2)
            assert 456 in result

    async def test_pickle_with_prepickle_true(self, storage_config):
        """Test pickle/unpickle with prepickle=True."""
        async with AsyncMinHashLSH(storage_config=storage_config, threshold=0.5, num_perm=16, prepickle=True) as lsh:
            m1 = MinHash(16)
            m1.update(b"a")
            await lsh.insert("test_key", m1)
            pickled = pickle.dumps(lsh)

        async with pickle.loads(pickled) as lsh2:
            result = await lsh2.query(m1)
            assert "test_key" in result
            await lsh2.close()
