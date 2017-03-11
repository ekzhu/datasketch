.. _minhash_lsh_forest:

MinHash LSH Forest
==================

MinHash LSH is useful for radius (or threshold) queries. However,
**top-k** queries are often more useful in some cases. `LSH
Forest <http://ilpubs.stanford.edu:8090/678/1/2005-14.pdf>`__ by Bawa et
al. is a general LSH data structure that makes top-k query possible for
many different types of LSH indexes, which include MinHash LSH. I
implemented the MinHash LSH Forest, which takes a MinHash data sketch of
the query set, and returns the top-k matching sets that have the highest
Jaccard similarities with the query set.

The interface of :class:`datasketch.MinHashLSHForest` is similar to 
:class:`datasketch.MinHashLSH`,
however, it is very important to call ``index`` method after adding the
keys. Without calling the ``index`` method, the keys won't be
searchable.

.. code:: python

    from datasketch import MinHashLSHForest, MinHash

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

    # Create a MinHash LSH Forest with the same num_perm parameter
    forest = MinHashLSHForest(num_perm=128)

    # Add m2 and m3 into the index
    forest.add("m2", m2)
    forest.add("m3", m3)

    # IMPORTANT: must call index() otherwise the keys won't be searchable
    forest.index()

    # Check for membership using the key
    print("m2" in forest)
    print("m3" in forest)

    # Using m1 as the query, retrieve top 2 keys that have the higest Jaccard
    result = forest.query(m1, 2)
    print("Top 2 candidates", result)

The plot below shows the `mean average precision
(MAP) <https://www.kaggle.com/wiki/MeanAveragePrecision>`__ of linear
scan with MinHash and MinHash LSH Forest. Synthetic data was used. See
`benchmark <https://github.com/ekzhu/datasketch/blob/master/benchmark/lshforest_benchmark.py>`_ 
for details.

.. figure:: https://github.com/ekzhu/datasketch/raw/master/plots/lshforest_benchmark.png
   :alt: MinHashLSHForest Benchmark

   MinHashLSHForest Benchmark

(Optional) If you have read the LSH Forest
`paper <http://ilpubs.stanford.edu:8090/678/1/2005-14.pdf>`__, and
understand the data structure, you may want to customize another
parameter for :class:`datasketch.MinHashLSHForest` -- ``l``, the number of prefix trees
(or "LSH Trees" as in the paper) in the LSH Forest index. Different from
the paper, this implementation fixes the number of LSH functions, in
this case ``num_perm``, and makes the maximum depth of every prefix tree
dependent on ``num_perm`` and ``l``:

.. code:: python

    # The maximum depth of a prefix tree depends on num_perm and l
    k = int(num_perm / l)

This way the interface of the :class:`datasketch.MinHashLSHForest` is in coherence with
the interface of ``MinHash``.

.. code:: python

    # There is another optional parameter l (default l=8).
    forest = MinHashLSHForest(num_perm=250, l=10)
