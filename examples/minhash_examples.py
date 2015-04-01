'''
Some examples for MinHash
'''

from hashlib import sha1
from datasketch import minhash

data1 = ['minhash', 'is', 'a', 'probabilistic', 'data', 'structure', 'for',
        'estimating', 'the', 'similarity', 'between', 'datasets']
data2 = ['minhash', 'is', 'a', 'probability', 'data', 'structure', 'for',
        'estimating', 'the', 'similarity', 'between', 'documents']

def eg1():
    m1 = minhash.MinHash(1, 64)
    m2 = minhash.MinHash(1, 64)
    for d in data1:
        m1.digest(sha1(d.encode('utf8')))
    for d in data2:
        m2.digest(sha1(d.encode('utf8')))
    print("Estimated Jaccard for m1 and m2 is", minhash.jaccard([m1, m2]))

    s1 = set(data1)
    s2 = set(data2)
    actual_jaccard = float(len(s1.intersection(s2))) /\
            float(len(s1.union(s2)))
    print("The actual Jaccard for m1 and m2 is", actual_jaccard)

if __name__ == "__main__":
    eg1()
