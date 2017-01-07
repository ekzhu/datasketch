"""
Data sketch module for Python
=============================

datasketch is a Python module integrating various data sketch
algorithms. It aims to provide efficient approximation alternatives
to exact solutions to data mining and data integration problems.
"""
from datasketch.hyperloglog import HyperLogLog, HyperLogLogPlusPlus
from datasketch.minhash import MinHash
from datasketch.b_bit_minhash import bBitMinHash
from datasketch.lsh import MinHashLSH, WeightedMinHashLSH
from datasketch.weighted_minhash import WeightedMinHash, WeightedMinHashGenerator
from datasketch.lshforest import MinHashLSHForest, WeightedMinHashLSHForest
