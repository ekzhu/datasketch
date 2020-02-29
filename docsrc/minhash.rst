.. _minhash:

MinHash
=======

:class:`datasketch.MinHash` lets you estimate the `Jaccard
similarity <https://en.wikipedia.org/wiki/Jaccard_index>`__
(resemblance) between
`sets <https://en.wikipedia.org/wiki/Set_(mathematics)>`__ of
arbitrary sizes in linear time using a small and fixed memory space. It
can also be used to compute Jaccard similarity between data streams.
MinHash is introduced by Andrei Z. Broder in this
`paper <http://cs.brown.edu/courses/cs253/papers/nearduplicate.pdf>`__

.. code:: python

    from datasketch import MinHash

    data1 = ['minhash', 'is', 'a', 'probabilistic', 'data', 'structure', 'for',
            'estimating', 'the', 'similarity', 'between', 'datasets']
    data2 = ['minhash', 'is', 'a', 'probability', 'data', 'structure', 'for',
            'estimating', 'the', 'similarity', 'between', 'documents']

    m1, m2 = MinHash(), MinHash()
    for d in data1:
        m1.update(d.encode('utf8'))
    for d in data2:
        m2.update(d.encode('utf8'))
    print("Estimated Jaccard for data1 and data2 is", m1.jaccard(m2))

    s1 = set(data1)
    s2 = set(data2)
    actual_jaccard = float(len(s1.intersection(s2)))/float(len(s1.union(s2)))
    print("Actual Jaccard for data1 and data2 is", actual_jaccard)

You can adjust the accuracy by customizing the number of permutation
functions used in MinHash.

.. code:: python

    # This will give better accuracy than the default setting (128).
    m = MinHash(num_perm=256)

The trade-off for better accuracy is slower speed and higher memory
usage. Because using more permutation functions means 
    
    1. more CPU instructions for every data value hashed and 
    2. more hash values to be stored. 

The speed and memory usage of MinHash are both linearly
proportional to the number of permutation functions used.

.. figure:: /_static/hashfunc/minhash_benchmark_sha1.png
   :alt: MinHash Benchmark

You can union two MinHash object using the ``merge`` function. This
makes MinHash useful in parallel MapReduce style data analysis.

.. code:: python

    # The makes m1 the union of m2 and the original m1.
    m1.merge(m2)

MinHash can be used for estimating the number of distinct elements, or
cardinality. The analysis is presented in `Cohen
1994 <http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=365694>`__.

.. code:: python

    # Returns the estimation of the cardinality of
    # all data values seen so far.
    m.count()

If you are handling billions of MinHash objects, consider using 
:class:`datasketch.LeanMinHash` to reduce your memory footprint.

Use Different Hash Functions
----------------------------

MinHash by default uses the SHA1 hash funciton from Python's built-in 
`hashlib <https://docs.python.org/3.7/library/hashlib.html>`__ library.
You can also change the hash function using the `hashfunc` parameter
in the constructor.

.. code:: python

    # Let's use MurmurHash3.
    import mmh3
    
    # We need to define a new hash function that outputs an integer that
    # can be encoded in 32 bits.
    def _hash_func(d):
        return mmh3.hash32(d)

    # Use this function in MinHash constructor.
    m = MinHash(hashfunc=_hash_func)

Different hash functions have different performance-accuracy trade-off,
you can use the benchmark code in `benchmark/minhash_benchmark.py` to 
run some tests. Here are the results for some popular hash functions
available in Python.

MurmurHash3: `mmh3 <https://pypi.org/project/mmh3/>`__

.. figure:: /_static/hashfunc/minhash_benchmark_mmh3.png
   :alt: MinHash Benchmark

`xxhash <https://pypi.org/project/xxhash/>`__

.. figure:: /_static/hashfunc/minhash_benchmark_xxh.png
   :alt: MinHash Benchmark

`Farmhash <https://pypi.org/project/pyfarmhash>`__

.. figure:: /_static/hashfunc/minhash_benchmark_farmhash.png
   :alt: MinHash Benchmark

Common Issues with MinHash
--------------------------

1. `High estimation error when set sizes differ by a lot <https://github.com/ekzhu/datasketch/issues/85>`__
2. `Use Inclusion-Exclusion Principle (i.e., merge() and count() functions) instead of jaccard() to estimate similarity <https://github.com/ekzhu/datasketch/issues/85>`__
3. `Storing MinHash for later use <https://github.com/ekzhu/datasketch/issues/122>`__

`See more issues <https://github.com/ekzhu/datasketch/issues?utf8=%E2%9C%93&q=minhash>`__
