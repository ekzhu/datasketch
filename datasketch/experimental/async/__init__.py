import sys
from datasketch.experimental.async.lsh import AsyncMinHashLSH

if sys.version_info < (3, 6):
    raise RuntimeError("Support only python >= 3.6")

# Alias
AsyncWeightedMinHashLSH = AsyncMinHashLSH
