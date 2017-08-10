.. _minhash_lsh:

MinHash LSH
===========

Suppose you have a very large collection of
`sets <https://en.wikipedia.org/wiki/Set_(mathematics)>`__. Giving a
query, which is also a set, you want to find sets in your collection
that have Jaccard similarities above certain threshold, and you want to
do it with many other queries. To do this efficiently, you can create a
MinHash for every set, and when a query comes, you compute the Jaccard
similarities between the query MinHash and all the MinHash of your
collection, and return the sets that satisfy your threshold.

The said approach is still an O(n) algorithm, meaning the query cost
increases linearly with respect to the number of sets. A popular
alternative is to use Locality Sensitive Hashing (LSH) index. LSH can be
used with MinHash to achieve sub-linear query cost - that is a huge
improvement. The details of the algorithm can be found in `Chapter 3,
Mining of Massive
Datasets <http://infolab.stanford.edu/~ullman/mmds/ch3.pdf>`__,

This package includes the classic version of MinHash LSH. It is
important to note that the query does not give you the exact result, due
to the use of MinHash and LSH. There will be false positives - sets that
do not satisfy your threshold but returned, and false negatives -
qualifying sets that are not returned. However, the property of LSH
assures that sets with higher Jaccard similarities always have higher
probabilities to get returned than sets with lower similarities.
Moreover, LSH can be optimized so that there can be a "jump" in
probability right at the threshold, making the qualifying sets much more
likely to get returned than the rest.

.. code:: python
        
        from datasketch import MinHash, MinHashLSH

        set1 = set(['minhash', 'is', 'a', 'probabilistic', 'data', 'structure', 'for',
                    'estimating', 'the', 'similarity', 'between', 'datasets'])
        set2 = set(['minhash', 'is', 'a', 'probability', 'data', 'structure', 'for',
                    'estimating', 'the', 'similarity', 'between', 'documents'])
        set3 = set(['minhash', 'is', 'probability', 'data', 'structure', 'for',
                    'estimating', 'the', 'similarity', 'between', 'documents'])
        
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

The Jaccard similarity threshold must be set at initialization, and
cannot be changed. So does the number of permutation functions (``num_perm``) parameter. 
Similar to MinHash, more permutation functions improves the accuracy,
but also increases query cost, since more processing is required as the
MinHash gets bigger. 
I experimented with the `20 News Group
Dataset <http://scikit-learn.org/stable/datasets/twenty_newsgroups.html>`__,
which has an average cardinality of 193 (3-shingles). The average
recall, average precision, and 90 percentile query time vs. number of permutation 
functions
are plotted below. 
See the `benchmark` 
directory in the source code repository for more experiment and 
plotting code.

.. figure:: /_static/lsh_benchmark.png
   :alt: MinHashLSH Benchmark

There are other optional parameters that can be used to tune the index.
See the documentation of :class:`datasketch.MinHashLSH` for details.

MinHash LSH does not support Top-K queries.
See :ref:`minhash_lsh_forest` for an alternative.
In addition, Jaccard similarity may not be the best measure if your intention is to
find sets having high intersection with the query.
For intersection search, see :ref:`minhash_lsh_ensemble`.

MinHash LSH at scale
--------------------
MinHash LSH supports a Redis backend for querying large datasets as part of
a production environment. The Redis storage backend is supported using

.. code:: python

      from datasketch import MinHashLSH

      lsh = MinHashLSH(
         threshold=0.5, num_perm=128, storage_config={
            'type': 'redis',
            'redis': {'host': 'localhost', 'port': 6379
         })

To insert a large number of MinHashes in sequence, it is advisable to use
an insertion session. This reduces the number of network calls during
bulk insertion.

Note that for the Redis storage, keys must be ``bytes`` objects.

.. code:: python

      data_list = [("m1", m1), ("m2", m2), ("m3", m3)]

      with lsh.insertion_session() as session:
         for key, minhash in data_list:
            session.insert(key, minhash)

Note that querying the LSH object during an open insertion session may result in
inconsistency.
