import sys

if sys.version_info < (3, 5):
    raise RuntimeError("Support only python >= 3.5")

from datasketch.async.lsh import AsyncMinHashLSH

# Alias
AsyncWeightedMinHashLSH = AsyncMinHashLSH
