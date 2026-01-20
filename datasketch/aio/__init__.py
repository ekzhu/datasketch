"""Async MinHash LSH module.

This module provides asynchronous implementations of MinHash LSH for use with
async storage backends like MongoDB (via motor) and Redis (via redis.asyncio).

Example:
    .. code-block:: python

        from datasketch.aio import AsyncMinHashLSH
        from datasketch import MinHash

        async def main():
            async with AsyncMinHashLSH(
                storage_config={"type": "aiomongo", "mongo": {"host": "localhost", "port": 27017}},
                threshold=0.5,
                num_perm=128
            ) as lsh:
                m = MinHash(num_perm=128)
                m.update(b"data")
                await lsh.insert("key", m)
                result = await lsh.query(m)

"""

from datasketch.aio.lsh import (
    AsyncMinHashLSH,
    AsyncMinHashLSHDeleteSession,
    AsyncMinHashLSHInsertionSession,
)

__all__ = [
    "AsyncMinHashLSH",
    "AsyncMinHashLSHDeleteSession",
    "AsyncMinHashLSHInsertionSession",
]
