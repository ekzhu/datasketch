#!/usr/bin/env python
# -*- coding: utf-8 -*-

from time import time
import string
import random
import asyncio
import concurrent.futures
from functools import partial
import aioredis
from itertools import islice, chain
from datasketch.experimental.aio.lsh import AsyncMinHashLSH
from datasketch import MinHashLSH, MinHash

aioSTORAGE_CONFIG_REDIS = {'type': 'aioredis', 'redis': {'host': 'localhost', 'port': 6379}}
syncSTORAGE_CONFIG_REDIS = {'type': 'redis', 'redis': {'host': 'localhost', 'port': 6379}}


def timeit(func):
    async def aiohelper(*args, **params):
        start = time()
        print("===== Executing function '{}' =====".format(func.__name__))
        result = await func(*args, **params)
        print("Elapsed time: {:.03f} sec".format(time() - start))
        return result

    def helper(*args, **params):
        start = time()
        print("===== Executing function '{}' =====".format(func.__name__))
        result = func(*args, **params)
        print("Elapsed time: {:.03f} sec".format(time() - start))
        return result

    return aiohelper if asyncio.iscoroutinefunction(func) else helper


def prepare_data(length: int):
    def chunk(it, size):
        it = iter(it)
        return iter(lambda: tuple(islice(it, size)), ())

    _chunked_str = chunk((random.choice(string.ascii_lowercase) for _ in range(length)), 4)
    seq = frozenset(chain((''.join(s) for s in _chunked_str),
                          ('aahhb', 'aahh', 'aahhc', 'aac', 'kld', 'bhg', 'kkd', 'yow', 'ppi', 'eer')))
    objs = [MinHash(16) for _ in range(len(seq))]
    for e, obj in zip(seq, objs):
        for i in e:
            obj.update(i.encode('utf-8'))

    return [(e, m) for e, m in zip(seq, objs)]


@timeit
async def insertion_session_aioredis(aiolsh: AsyncMinHashLSH, data: list, batch_size: int):
    async with aiolsh.session(batch_size=batch_size) as session:
        fs = (session.insert(key, minhash, check_duplication=False) for key, minhash in data)
        await asyncio.gather(*fs)


@timeit
async def query_aioredis(aiolsh: AsyncMinHashLSH, data: list):
    fs = (aiolsh.query(minhash) for key, minhash in data)
    return await asyncio.gather(*fs)


@timeit
async def insert_aioredis(aiolsh: AsyncMinHashLSH, data: list):
    fs = (aiolsh.insert(key, minhash, check_duplication=False) for key, minhash in data)
    await asyncio.gather(*fs)


@timeit
def insertion_session_syncredis(lsh: MinHashLSH, data: list, buffer_size: int):
    with lsh.insertion_session(buffer_size=buffer_size) as session:
        for key, minhash in data:
            session.insert(key, minhash, check_duplication=False)


@timeit
async def aioinsert_syncredis_with_executor(lsh: MinHashLSH, data: list, executor):
    loop = asyncio.get_event_loop()
    tasks = (loop.run_in_executor(executor,
                                  partial(lsh.insert, check_duplication=False), key, minhash) for key, minhash in data)
    await asyncio.gather(*tasks)


@timeit
async def aioquery_syncredis(lsh: MinHashLSH, data: list, executor):
    loop = asyncio.get_event_loop()
    tasks = (loop.run_in_executor(executor,
                                  lsh.query, minhash) for _, minhash in data)
    return await asyncio.gather(*tasks)


@timeit
def query_syncredis(lsh: MinHashLSH, data: list):
    return [lsh.query(minhash) for key, minhash in data]


async def run_async_test(data: list, batch_size: int):
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=100)
    lsh = MinHashLSH(storage_config=syncSTORAGE_CONFIG_REDIS,
                     threshold=0.5, num_perm=16)
    await aioinsert_syncredis_with_executor(lsh, data, executor)
    await aioquery_syncredis(lsh, data, executor)

    async with AsyncMinHashLSH(storage_config=aioSTORAGE_CONFIG_REDIS,
                               threshold=0.5, num_perm=16) as lsh2:
        await insert_aioredis(lsh2, data)

    async with AsyncMinHashLSH(storage_config=aioSTORAGE_CONFIG_REDIS,
                               threshold=0.5, num_perm=16) as lsh3:
        await insertion_session_aioredis(lsh3, data, batch_size)
        await query_aioredis(lsh3, data)

    dsn = 'redis://{host}:{port}'.format(**aioSTORAGE_CONFIG_REDIS['redis'])
    redis = await aioredis.create_redis(dsn)
    await redis.flushdb()
    redis.close()
    await redis.wait_closed()


def run_sync_test(data: list, batch_size: int):
    lsh = MinHashLSH(storage_config=syncSTORAGE_CONFIG_REDIS,
                     threshold=0.5, num_perm=16)
    insertion_session_syncredis(lsh, data, buffer_size=batch_size)
    query_syncredis(lsh, data)


if __name__ == '__main__':
    length, batch_size = 50000, 500
    print("Number of objects <{}>".format(length // 4))
    test_data = prepare_data(length)

    print("Synchronous tests")
    run_sync_test(test_data, batch_size)

    print()

    print("Asynchronous tests")
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run_async_test(test_data, batch_size))
    loop.close()
