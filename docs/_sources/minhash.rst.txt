.. _minhash:

MinHash
=======

:class:`datasketch.MinHash` lets you estimate the `Jaccard
similarity <https://en.wikipedia.org/wiki/Jaccard_index>`__
(resemblance) between
`**sets** <https://en.wikipedia.org/wiki/Set_(mathematics)>`__ of
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
usage. Because using more permutation functions means 1) more CPU
instructions for every data value hashed and 2) more hash values to be
stored. The speed and memory usage of MinHash are both linearly
proportional to the number of permutation functions used.

.. figure:: https://github.com/ekzhu/datasketch/raw/master/plots/minhash_benchmark.png
   :alt: MinHash Benchmark

   MinHash Benchmark

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

