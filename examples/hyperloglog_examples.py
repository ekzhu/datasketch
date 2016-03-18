'''
Some examples for MinHash
'''

from datasketch.hyperloglog import HyperLogLog

data1 = ['hyperloglog', 'is', 'a', 'probabilistic', 'data', 'structure', 'for',
         'estimating', 'the', 'cardinality', 'of', 'dataset', 'dataset', 'a']
data2 = ['hyperloglog', 'is', 'a', 'probabilistic', 'DATA', 'structure', 'for',
         'estimating', 'the', 'number', 'of', 'distinct', 'values', 'of',
         'dataset', 'dataset', 'a']

def eg1():
    h = HyperLogLog()
    for d in data1:
        h.update(d.encode('utf8'))
    print("Estimated cardinality is", h.count())

    s1 = set(data1)
    print("Actual cardinality is", len(s1))

def eg2():
    h1 = HyperLogLog()
    h2 = HyperLogLog()
    for d in data1:
        h1.update(d.encode('utf8'))
    for d in data2:
        h2.update(d.encode('utf8'))
    u = HyperLogLog.union(h1, h2)
    print("Estimated union cardinality is", u.count())

    s1 = set(data1)
    s2 = set(data2)
    su = s1.union(s2)
    print("Actual union cardinality is", len(su))



if __name__ == "__main__":
    eg1()
    eg2()
