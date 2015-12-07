'''
Some examples for LSH
'''

from hashlib import sha1
from datasketch.minhash import MinHash
from datasketch.lsh import LSH

data1 = ['minhash', 'is', 'a', 'probabilistic', 'data', 'structure', 'for',
        'estimating', 'the', 'similarity', 'between', 'datasets']
data2 = ['minhash', 'is', 'a', 'probability', 'data', 'structure', 'for',
        'estimating', 'the', 'similarity', 'between', 'documents']
data3 = ['minhash', 'is', 'probability', 'data', 'structure', 'for',
        'estimating', 'the', 'similarity', 'between', 'documents']

def eg1():
    m1 = MinHash()
    m2 = MinHash()
    m3 = MinHash()
    for d in data1:
        m1.digest(sha1(d.encode('utf8')))
    for d in data2:
        m2.digest(sha1(d.encode('utf8')))
    for d in data3:
        m3.digest(sha1(d.encode('utf8')))

    # Create LSH index
    lsh = LSH(threshold=0.5)
    lsh.insert("m2", m2)
    lsh.insert("m3", m3)
    result = lsh.query(m1)
    print("Approximate neighbours with Jaccard similarity > 0.5", result)

if __name__ == "__main__":
    eg1()
