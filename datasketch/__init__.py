from datasketch.hyperloglog import HyperLogLog, HyperLogLogPlusPlus
from datasketch.minhash import MinHash
from datasketch.b_bit_minhash import bBitMinHash
from datasketch.lsh import MinHashLSH
from datasketch.weighted_minhash import WeightedMinHash, WeightedMinHashGenerator
from datasketch.lshforest import MinHashLSHForest
from datasketch.lshensemble import MinHashLSHEnsemble
from datasketch.lean_minhash import LeanMinHash
from datasketch.hashfunc import sha1_hash32

# Alias
WeightedMinHashLSH = MinHashLSH
WeightedMinHashLSHForest = MinHashLSHForest

# Version
from datasketch.version import __version__
