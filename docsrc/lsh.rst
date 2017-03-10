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

    data1 = ['minhash', 'is', 'a', 'probabilistic', 'data', 'structure', 'for',
            'estimating', 'the', 'similarity', 'between', 'datasets']
    data2 = ['minhash', 'is', 'a', 'probability', 'data', 'structure', 'for',
            'estimating', 'the', 'similarity', 'between', 'documents']
    data3 = ['minhash', 'is', 'probability', 'data', 'structure', 'for',
            'estimating', 'the', 'similarity', 'between', 'documents']

    # Create MinHash objects
    m1 = MinHash(num_perm=128)
    m2 = MinHash(num_perm=128)
    m3 = MinHash(num_perm=128)
    for d in data1:
        m1.update(d.encode('utf8'))
    for d in data2:
        m2.update(d.encode('utf8'))
    for d in data3:
        m3.update(d.encode('utf8'))

    # Create an MinHashLSH index optimized for Jaccard threshold 0.5,
    # that accepts MinHash objects with 128 permutations functions
    lsh = MinHashLSH(threshold=0.5, num_perm=128)

    # Insert m2 and m3 into the index
    lsh.insert("m2", m2)
    lsh.insert("m3", m3)

    # Check for membership using the key
    print("m2" in lsh)
    print("m3" in lsh)

    # Using m1 as the query, retrieve the keys of the qualifying datasets
    result = lsh.query(m1)
    print("Candidates with Jaccard similarity > 0.5", result)

    # Remove key from lsh
    lsh.remove("m2")

The Jaccard similarity threshold must be set at initialization, and
cannot be changed. So does the ``num_perm`` parameter. Similar to
MinHash, higher ``num_perm`` can improve the accuracy of :class:`datasketch.MinHashLSH`,
but increase query cost, since more processing is required as the
MinHash gets bigger. Unlike MinHash, the benefit of higher ``num_perm``
seems to be limited for :class:`datasketch.MinHashLSH` - it looks like when ``num_perm``
becomes greater than the dataset cardinality, both precision and recall
starts to decrease. I experimented with the `20 News Group
Dataset <http://scikit-learn.org/stable/datasets/twenty_newsgroups.html>`__,
which has an average cardinality of 193 (3-shingles). The average
recall, average precision, and 90 percentile query time vs. ``num_perm``
are plotted below. 
See the `benchmark <https://github.com/ekzhu/datasketch/tree/master/benchmark>`_ 
directory for the experiment and plotting code.

.. figure:: https://github.com/ekzhu/datasketch/raw/master/plots/lsh_benchmark.png
   :alt: MinHashLSH Benchmark

   MinHashLSH Benchmark

There are other optional parameters that be used to tune the index:

.. code:: python

    # Use defaults: threshold=0.5, num_perm=128, weights=(0.5, 0.5)
    lsh = MinHashLSH()

    # `weights` controls the relative importance between minizing false positive
    # and minizing false negative when building the `MinHashLSH`.
    # `weights` must sum to 1.0, and the format is
    # (false positive weight, false negative weight).
    # For example, if minizing false negative (or maintaining high recall) is more
    # important, assign more weight toward false negative: weights=(0.4, 0.6).
    # Note: try to live with a small difference between weights (i.e. < 0.5).
    lsh = MinHashLSH(weights=(0.4, 0.6))

