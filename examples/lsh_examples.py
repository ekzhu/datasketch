'''
Some examples for LSH
'''

from hashlib import sha1
import numpy as np
from datasketch.minhash import MinHash
from datasketch.weighted_minhash import WeightedMinHashGenerator
from datasketch.lsh import MinHashLSH

set1 = set(['minhash', 'is', 'a', 'probabilistic', 'data', 'structure', 'for',
            'estimating', 'the', 'similarity', 'between', 'datasets'])
set2 = set(['minhash', 'is', 'a', 'probability', 'data', 'structure', 'for',
            'estimating', 'the', 'similarity', 'between', 'documents'])
set3 = set(['minhash', 'is', 'probability', 'data', 'structure', 'for',
            'estimating', 'the', 'similarity', 'between', 'documents'])

v1 = np.random.uniform(1, 10, 10)
v2 = np.random.uniform(1, 10, 10)
v3 = np.random.uniform(1, 10, 10)

def eg1():
    m1 = MinHash(num_perm=128)
    m2 = MinHash(num_perm=128)
    m3 = MinHash(num_perm=128)
    for d in set1:
        m1.update(d.encode('utf8'))
    for d in set2:
        m2.update(d.encode('utf8'))
    for d in set3:
        m3.update(d.encode('utf8'))

    # Create LSH index
    lsh = MinHashLSH(threshold=0.5, num_perm=128)
    lsh.insert("m2", m2)
    lsh.insert("m3", m3)
    result = lsh.query(m1)
    print("Approximate neighbours with Jaccard similarity > 0.5", result)

    # Merge two LSH index
    lsh1 = MinHashLSH(threshold=0.5, num_perm=128)
    lsh1.insert("m2", m2)
    lsh1.insert("m3", m3)

    lsh2 = MinHashLSH(threshold=0.5, num_perm=128)
    lsh2.insert("m1", m1)

    lsh1.merge(lsh2)
    print("Does m1 exist in the lsh1...", "m1" in lsh1.keys)
    # if check_overlap flag is set to True then it will check the overlapping of the keys in the two MinHashLSH
    lsh1.merge(lsh2,check_overlap=True)

def eg2():
    mg = WeightedMinHashGenerator(10, 5)
    m1 = mg.minhash(v1)
    m2 = mg.minhash(v2)
    m3 = mg.minhash(v3)
    print("Estimated Jaccard m1, m2", m1.jaccard(m2))
    print("Estimated Jaccard m1, m3", m1.jaccard(m3))
    # Create LSH index
    lsh = MinHashLSH(threshold=0.1, num_perm=5)
    lsh.insert("m2", m2)
    lsh.insert("m3", m3)
    result = lsh.query(m1)
    print("Approximate neighbours with weighted Jaccard similarity > 0.1", result)

if __name__ == "__main__":
    eg1()
    eg2()
