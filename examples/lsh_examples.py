'''
Some examples for LSH
'''

from hashlib import sha1
import numpy as np
from datasketch.minhash import MinHash
from datasketch.weighted_minhash import WeightedMinHashGenerator
from datasketch.lsh import WeightedMinHashLSH, MinHashLSH

data1 = ['minhash', 'is', 'a', 'probabilistic', 'data', 'structure', 'for',
        'estimating', 'the', 'similarity', 'between', 'datasets']
data2 = ['minhash', 'is', 'a', 'probability', 'data', 'structure', 'for',
        'estimating', 'the', 'similarity', 'between', 'documents']
data3 = ['minhash', 'is', 'probability', 'data', 'structure', 'for',
        'estimating', 'the', 'similarity', 'between', 'documents']

v1 = np.random.uniform(1, 10, 10)
v2 = np.random.uniform(1, 10, 10)
v3 = np.random.uniform(1, 10, 10)

def eg1():
    m1 = MinHash()
    m2 = MinHash()
    m3 = MinHash()
    for d in data1:
        m1.update(d.encode('utf8'))
    for d in data2:
        m2.update(d.encode('utf8'))
    for d in data3:
        m3.update(d.encode('utf8'))

    # Create LSH index
    lsh = MinHashLSH(threshold=0.5)
    lsh.insert("m2", m2)
    lsh.insert("m3", m3)
    result = lsh.query(m1)
    print("Approximate neighbours with Jaccard similarity > 0.5", result)

def eg2():
    mg = WeightedMinHashGenerator(10, 5)
    m1 = mg.minhash(v1)
    m2 = mg.minhash(v2)
    m3 = mg.minhash(v3)
    print("Estimated Jaccard m1, m2", m1.jaccard(m2))
    print("Estimated Jaccard m1, m3", m1.jaccard(m3))
    # Create LSH index
    lsh = WeightedMinHashLSH(threshold=0.1, sample_size=5)
    lsh.insert("m2", m2)
    lsh.insert("m3", m3)
    result = lsh.query(m1)
    print("Approximate neighbours with weighted Jaccard similarity > 0.1", result)

if __name__ == "__main__":
    eg1()
    eg2()
