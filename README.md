# datasketch

[![Build Status](https://travis-ci.org/ekzhu/datasketch.svg?branch=master)](https://travis-ci.org/ekzhu/datasketch)

datasketch gives you probabilistic data structures that can process
vary large amount of data super fast, with little loss of accuracy.

datasketch must be used with Python 2.7 or above.

## Install

To install datasketch using `pip`:

    pip install datasketch -U

## MinHash

MinHash lets you estimate the Jaccard similarity between datasets of
arbitrary sizes in linear time using a small and fixed memory space.
It can also be used to compute Jaccard similarity between data streams.
MinHash is introduced by Andrei Z. Broder in this
[paper](http://cs.brown.edu/courses/cs253/papers/nearduplicate.pdf)

```python
from hashlib import sha1
from datasketch.minhash import MinHash, jaccard

data1 = ['minhash', 'is', 'a', 'probabilistic', 'data', 'structure', 'for',
        'estimating', 'the', 'similarity', 'between', 'datasets']
data2 = ['minhash', 'is', 'a', 'probability', 'data', 'structure', 'for',
        'estimating', 'the', 'similarity', 'between', 'documents']

m1, m2 = MinHash(), MinHash()
for d in data1:
	m1.digest(sha1(d.encode('utf8')))
for d in data2:
	m2.digest(sha1(d.encode('utf8')))
print("Estimated Jaccard for data1 and data2 is", jaccard([m1, m2]))

s1 = set(data1)
s2 = set(data2)
actual_jaccard = float(len(s1.intersection(s2)))/float(len(s1.union(s2)))
print("Actual Jaccard for data1 and data2 is", actual_jaccard)
```

You can adjust the accuracy by customizing the number of permutation functions
used in MinHash.

```python
# This will give better accuracy than the default setting (128).
m = MinHash(num_perm=256)
```

The trade-off for better accuracy is slower speed and higher memory usage.
Because using more permutation functions means 1) more CPU instructions
for every hash digested and 2) more hash values to be stored.
The speed and memory usage of MinHash are both linearly proportional
to the number of permutation functions used.

You can union two MinHash object using the `merge` function.
This makes MinHash useful in parallel MapReduce style data analysis.

```python
# The makes m1 the union of m2 and the original m1.
m1.merge(m2)
```

You can serialize a MinHash object into a byte buffer, and reconstruct a MinHash
from a byte buffer. You can also obtain the byte size of the MinHash object
using the `bytesize` function.

```python
m = MinHash()
buf = bytearray(m.bytesize())
m.serialize(buf)
n = MinHash.deserialize(buf)
```

## HyperLogLog

HyperLogLog is capable of estimating the cardinality (the number of
distinct values) of dataset in a single pass, using a small and fixed
memory space.
HyperLogLog is first introduced in this
[paper](http://algo.inria.fr/flajolet/Publications/FlFuGaMe07.pdf)
by Philippe Flajolet, Éric Fusy, Olivier Gandouet and Frédéric Meunier.

```python
from hashlib import sha1
from datasketch.hyperloglog import HyperLogLog

data1 = ['hyperloglog', 'is', 'a', 'probabilistic', 'data', 'structure', 'for',
'estimating', 'the', 'cardinality', 'of', 'dataset', 'dataset', 'a']

h = HyperLogLog()
for d in data1:
  h.digest(sha1(d.encode('utf8')))
print("Estimated cardinality is", h.count())

s1 = set(data1)
print("Actual cardinality is", len(s1))
```

As in MinHash, you can also control the accuracy of HyperLogLog by changing
the parameter p.

```python
# This will give better accuracy than the default setting (8).
h = HyperLogLog(p=12)
```

Interestingly, there is no speed penalty for using higher p value.
However the memory usage is exponential to the p value.

As in MinHash, you can also merge two HyperLogLogs to create a union HyperLogLog.

```python
h1 = HyperLogLog()
h2 = HyperLogLog()
h1.digest(sha1('test'.encode('utf8')))
# The makes h1 the union of h2 and the original h1.
h1.merge(h2)
# This will return the cardinality of the union
h1.count()
```

You can serialize a HyperLogLog object into a byte buffer, and reconstruct a
HyperLogLog
from a byte buffer. You can also obtain the byte size of the HyperLogLog object
using the `bytesize` function.

```python
h = HyperLogLog()
buf = bytearray(h.bytesize())
h.serialize(buf)
n = HyperLogLog.deserialize(buf)
```
