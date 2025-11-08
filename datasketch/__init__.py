import importlib.metadata
from typing import Final

try:
    _version = importlib.metadata.version(__name__)
except importlib.metadata.PackageNotFoundError:
    _version = "0.0.0"  # Fallback for development mode
__version__: Final[str] = _version

from datasketch.b_bit_minhash import bBitMinHash
from datasketch.hashfunc import sha1_hash32
from datasketch.hnsw import HNSW
from datasketch.hyperloglog import HyperLogLog, HyperLogLogPlusPlus
from datasketch.lean_minhash import LeanMinHash
from datasketch.lsh import MinHashLSH
from datasketch.lsh_bloom import MinHashLSHBloom
from datasketch.lshensemble import MinHashLSHEnsemble
from datasketch.lshforest import MinHashLSHForest
from datasketch.minhash import MinHash
from datasketch.weighted_minhash import WeightedMinHash, WeightedMinHashGenerator

# Alias
WeightedMinHashLSH = MinHashLSH
WeightedMinHashLSHForest = MinHashLSHForest


__all__ = [
    "HNSW",
    "HyperLogLog",
    "HyperLogLogPlusPlus",
    "LeanMinHash",
    "MinHash",
    "MinHashLSH",
    "MinHashLSHBloom",
    "MinHashLSHEnsemble",
    "MinHashLSHForest",
    "WeightedMinHash",
    "WeightedMinHashGenerator",
    "WeightedMinHashLSH",
    "WeightedMinHashLSHForest",
    "bBitMinHash",
    "sha1_hash32",
]
