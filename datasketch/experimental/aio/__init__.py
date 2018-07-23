import sys

if sys.version_info < (3, 6):
    raise ImportError("Can't use AsyncMinHash module. Python version should be >=3.6")

from datasketch.experimental.aio.lsh import AsyncMinHashLSH
# Alias
AsyncWeightedMinHashLSH = AsyncMinHashLSH