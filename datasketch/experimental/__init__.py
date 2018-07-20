import sys

if sys.version_info >= (3, 6):
    from datasketch.experimental.aio.lsh import AsyncMinHashLSH
    # Alias
    AsyncWeightedMinHashLSH = AsyncMinHashLSH
