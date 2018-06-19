from datasketch.hyperloglog import HyperLogLog, HyperLogLogPlusPlus
from datasketch.minhash import MinHash
from datasketch.b_bit_minhash import bBitMinHash
from datasketch.lsh import MinHashLSH
from datasketch.weighted_minhash import WeightedMinHash, WeightedMinHashGenerator
from datasketch.lshforest import MinHashLSHForest
from datasketch.lshensemble import MinHashLSHEnsemble
from datasketch.lean_minhash import LeanMinHash

# Alias
WeightedMinHashLSH = MinHashLSH
WeightedMinHashLSHForest = MinHashLSHForest
