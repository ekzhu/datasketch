'''
Some examples for MinHash
'''

from hashlib import sha1
from datasketch.minhash import MinHash, jaccard

data1 = ['minhash', 'is', 'a', 'probabilistic', 'data', 'structure', 'for',
        'estimating', 'the', 'similarity', 'between', 'datasets']
data2 = ['minhash', 'is', 'a', 'probability', 'data', 'structure', 'for',
        'estimating', 'the', 'similarity', 'between', 'documents']

def eg1():
    m1 = MinHash()
    m2 = MinHash()
    for d in data1:
        m1.digest(sha1(d.encode('utf8')))
    for d in data2:
        m2.digest(sha1(d.encode('utf8')))
    print("Estimated Jaccard for data1 and data2 is", jaccard([m1, m2]))

    s1 = set(data1)
    s2 = set(data2)
    actual_jaccard = float(len(s1.intersection(s2))) /\
            float(len(s1.union(s2)))
    print("Actual Jaccard for data1 and data2 is", actual_jaccard)

if __name__ == "__main__":
    eg1()
