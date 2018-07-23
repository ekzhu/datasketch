"""
WARNING:

datasketch.experimental is dedicated to new modules that are to be merged into
the stable interface of datasketch. So their interfaces may change in future
versions.

To add a new class or function, register it here in this file. For example:

from new_module import NewModuleClass
"""
import sys

if sys.version_info < (3, 6):
    raise ImportError("Can't use AsyncMinHash module. Python version should be >=3.6")

from datasketch.experimental.aio.lsh import AsyncMinHashLSH
# Alias
AsyncWeightedMinHashLSH = AsyncMinHashLSH
