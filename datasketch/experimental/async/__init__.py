import sys

if sys.version_info < (3, 6):
    from datasketch.experimental.async.lsh import AsyncMinHashLSH
    # Alias
    AsyncWeightedMinHashLSH = AsyncMinHashLSH
