datasketch
==========

|Build Status| |DOI|

`Documentation <https://ekzhu.github.io/datasketch/documentation.html>`__

datasketch gives you probabilistic data structures that can process and
search vary large amount of data super fast, with little loss of
accuracy.

This package contains the following data sketches:

+--------------------+-----------------------------------------------+
| Data Sketch        | Usage                                         |
+====================+===============================================+
| MinHash            | estimate Jaccard similarity and cardinality   |
+--------------------+-----------------------------------------------+
| b-Bit MinHash      | estimate Jaccard similarity                   |
+--------------------+-----------------------------------------------+
| Weighted MinHash   | estimate weighted Jaccard similarity          |
+--------------------+-----------------------------------------------+
| HyperLogLog        | estimate cardinality                          |
+--------------------+-----------------------------------------------+
| HyperLogLog++      | estimate cardinality                          |
+--------------------+-----------------------------------------------+

The following indexes for data sketches are provided to support
sub-linear query time:

+----------------------+-----------------------------+------------------------+
| Index                | For Data Sketch             | Supported Query Type   |
+======================+=============================+========================+
| MinHash LSH          | MinHash, Weighted MinHash   | Radius (Threshold)     |
+----------------------+-----------------------------+------------------------+
| MinHash LSH Forest   | MinHash, Weighted MinHash   | Top-K                  |
+----------------------+-----------------------------+------------------------+

datasketch must be used with Python 2.7 or above and NumPy 1.11 or
above. Scipy is optional, but with it the LSH initialization can be much
faster.

Install
-------

To install datasketch using ``pip``:

::

    pip install datasketch -U

This will also install NumPy as dependency.

MinHash
-------

`Documentation <https://ekzhu.github.io/datasketch/documentation.html#datasketch.MinHash>`__

MinHash lets you estimate the `Jaccard
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

MinHash LSH
-----------

`Documentation <https://ekzhu.github.io/datasketch/documentation.html#datasketch.MinHashLSH>`__

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
MinHash, higher ``num_perm`` can improve the accuracy of ``MinHashLSH``,
but increase query cost, since more processing is required as the
MinHash gets bigger. Unlike MinHash, the benefit of higher ``num_perm``
seems to be limited for ``MinHashLSH`` - it looks like when ``num_perm``
becomes greater than the dataset cardinality, both precision and recall
starts to decrease. I experimented with the `20 News Group
Dataset <http://scikit-learn.org/stable/datasets/twenty_newsgroups.html>`__,
which has an average cardinality of 193 (3-shingles). The average
recall, average precision, and 90 percentile query time vs. ``num_perm``
are plotted below. See the ``benchmark`` directory for the experiment
and plotting code.

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

MinHash LSH Forest
------------------

`Documentation <https://ekzhu.github.io/datasketch/documentation.html#datasketch.MinHashLSHForest>`__

MinHash LSH is useful for radius (or threshold) queries. However,
**top-k** queries are often more useful in some cases. `LSH
Forest <http://ilpubs.stanford.edu:8090/678/1/2005-14.pdf>`__ by Bawa et
al. is a general LSH data structure that makes top-k query possible for
many different types of LSH indexes, which include MinHash LSH. I
implemented the MinHash LSH Forest, which takes a MinHash data sketch of
the query set, and returns the top-k matching sets that have the highest
Jaccard similarities with the query set.

The interface of ``MinHashLSHForest`` is similar to ``MinHashLSH``,
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
``benchmark/lshforest_benchmark.py`` for details.

.. figure:: https://github.com/ekzhu/datasketch/raw/master/plots/lshforest_benchmark.png
   :alt: MinHashLSHForest Benchmark

   MinHashLSHForest Benchmark

(Optional) If you have read the LSH Forest
`paper <http://ilpubs.stanford.edu:8090/678/1/2005-14.pdf>`__, and
understand the data structure, you may want to customize another
parameter for ``MinHashLSHForest`` -- ``l``, the number of prefix trees
(or "LSH Trees" as in the paper) in the LSH Forest index. Different from
the paper, this implementation fixes the number of LSH functions, in
this case ``num_perm``, and makes the maximum depth of every prefix tree
dependent on ``num_perm`` and ``l``:

.. code:: python

    # The maximum depth of a prefix tree depends on num_perm and l
    k = int(num_perm / l)

This way the interface of the ``MinHashLSHForest`` is in coherence with
the interface of ``MinHash``.

.. code:: python

    # There is another optional parameter l (default l=8).
    forest = MinHashLSHForest(num_perm=250, l=10)

Weighted MinHash
----------------

`Documentation <https://ekzhu.github.io/datasketch/documentation.html#datasketch.WeightedMinHash>`__

MinHash can be used to compress unweighted
`set <https://en.wikipedia.org/wiki/Set_(mathematics)>`__ or binary
vector, and estimate the `unweighted Jaccard
similarity <https://en.wikipedia.org/wiki/Jaccard_index>`__. It is
possible to modify MinHash for `weighted
Jaccard <https://en.wikipedia.org/wiki/Jaccard_index#Generalized_Jaccard_similarity_and_distance>`__
on `**multisets** <https://en.wikipedia.org/wiki/Multiset>`__ by
expanding each item (or dimension) by its weight (usually its count in
the multiset). However this approach does not support real number
weights, and doing so can be very expensive if the weights are very
large. `Weighted
MinHash <http://static.googleusercontent.com/media/research.google.com/en//pubs/archive/36928.pdf>`__
is created by Sergey Ioffe, and its performance does not depend on the
weights - as long as the universe of all possible items (or dimension
for vectors) is known. This makes it unsuitable for stream processing,
when the knowledge of unseen items cannot be assumed.

In this library, ``WeightedMinHash`` objects can only be created from
vectors using ``WeightedMinHashGenerator``, which takes the dimension as
a required parameter.

.. code:: python

    # Using default sample_size 256 and seed 1
    wmg = WeightedMinHashGenerator(1000)

You can specify the number of samples (similar to number of permutation
functions in MinHash) and the random seed.

.. code:: python

    wmg = WeightedMinHashGenerator(1000, sample_size=512, seed=12)

Here is a usage example.

.. code:: python

    from datasketch import WeightedMinHashGenerator

    v1 = [1, 3, 4, 5, 6, 7, 8, 9, 10, 4]
    v2 = [2, 4, 3, 8, 4, 7, 10, 9, 0, 0]

    # WeightedMinHashGenerator requires dimension as the first argument
    wmg = WeightedMinHashGenerator(len(v1))
    wm1 = wmg.minhash(v1) # wm1 is of the type WeightedMinHash
    wm2 = wmg.minhash(v2)
    print("Estimated Jaccard is", wm1.jaccard(wm2))

It is possible to make ``WeightedMinHash`` have a ``update`` interface
similar to ``MinHash`` and use it for stream data processing. However,
this makes the cost of ``update`` increase linearly with respect to the
weight. Thus, ``update`` is not implemented for ``WeightedMinHash`` in
this library.

Weighted MinHash as similar accuracy and performance profiles as
MinHash. As you increase the number of samples, you get better accuracy,
at the expense of slower speed.

.. figure:: https://github.com/ekzhu/datasketch/raw/master/plots/weighted_minhash_benchmark.png
   :alt: Weighted MinHash Benchmark

   Weighted MinHash Benchmark

The ``MinHashLSH`` index can also be used to index ``WeightedMinHash``.

.. code:: python

    import numpy as np
    from datasketch import WeightedMinHashGenerator
    from datasketch import MinHashLSH

    v1 = np.random.uniform(1, 10, 10)
    v2 = np.random.uniform(1, 10, 10)
    v3 = np.random.uniform(1, 10, 10)
    mg = WeightedMinHashGenerator(10, 5)
    m1 = mg.minhash(v1)
    m2 = mg.minhash(v2)
    m3 = mg.minhash(v3)

    # Create weighted MinHash LSH index
    lsh = MinHashLSH(threshold=0.1, sample_size=5)
    lsh.insert("m2", m2)
    lsh.insert("m3", m3)
    result = lsh.query(m1)
    print("Approximate neighbours with weighted Jaccard similarity > 0.1", result)

b-Bit MinHash
-------------

`b-Bit
MinHash <http://research.microsoft.com/pubs/120078/wfc0398-liPS.pdf>`__
is created by Ping Li and Arnd Christian König. It is a compression of
MinHash - it stores only the lowest b-bits of each minimum hashed values
in the MinHash, allowing one to trade accuracy for less storage cost.

When the actual Jaccard similarity, or resemblance, is large (>= 0.5),
b-Bit MinHash's estimation for Jaccard has very small loss of accuracy
comparing to the original MinHash. On the other hand, when the actual
Jaccard is small, b-Bit MinHash gives bad estimation for Jaccard, and it
tends to over-estimate.

.. figure:: https://github.com/ekzhu/datasketch/raw/master/plots/b_bit_minhash_benchmark.png
   :alt: b-Bit MinHash Benchmark

   b-Bit MinHash Benchmark

To create a b-Bit MinHash object from an existing MinHash object:

.. code:: python

    from datasketch import bBitMinHash

    # minhash is an existing MinHash object.
    bm = bBitMinHash(minhash)

To estimate Jaccard similarity using two b-Bit MinHash objects:

.. code:: python

    # Estimate Jaccard given bm1 and bm2, both must have the same
    # value for parameter b.
    bm1.jaccard(bm2)

The default value for parameter b is 1. Estimation accuracy can be
improved by keeping more bits - increasing the value for parameter b.

.. code:: python

    # Using higher value for b can improve accuracy, at the expense of
    # using more storage space.
    bm = bBitMinHash(minhash, b=4)

**Note:** for this implementation, using different values for the
parameter b won't make a difference in the in-memory size of b-Bit
MinHash, as the underlying storage is a NumPy integer array. However,
the size of a serialized b-Bit MinHash is determined by the parameter b
(and of course the number of permutation functions in the original
MinHash).

Because b-Bit MinHash only retains the lowest b-bits of the minimum
hashed values in the original MinHash, it is not mergable. Thus it has
no ``merge`` function.

HyperLogLog
-----------

`Documentation <https://ekzhu.github.io/datasketch/documentation.html#datasketch.HyperLogLog>`__

HyperLogLog is capable of estimating the cardinality (the number of
distinct values) of dataset in a single pass, using a small and fixed
memory space. HyperLogLog is first introduced in this
`paper <http://algo.inria.fr/flajolet/Publications/FlFuGaMe07.pdf>`__ by
Philippe Flajolet, Éric Fusy, Olivier Gandouet and Frédéric Meunier.

.. code:: python

    from datasketch import HyperLogLog

    data1 = ['hyperloglog', 'is', 'a', 'probabilistic', 'data', 'structure', 'for',
    'estimating', 'the', 'cardinality', 'of', 'dataset', 'dataset', 'a']

    h = HyperLogLog()
    for d in data1:
      h.update(d.encode('utf8'))
    print("Estimated cardinality is", h.count())

    s1 = set(data1)
    print("Actual cardinality is", len(s1))

As in MinHash, you can also control the accuracy of HyperLogLog by
changing the parameter p.

.. code:: python

    # This will give better accuracy than the default setting (8).
    h = HyperLogLog(p=12)

Interestingly, there is no speed penalty for using higher p value.
However the memory usage is exponential to the p value.

.. figure:: https://github.com/ekzhu/datasketch/raw/master/plots/hyperloglog_benchmark.png
   :alt: HyperLogLog Benchmark

   HyperLogLog Benchmark

As in MinHash, you can also merge two HyperLogLogs to create a union
HyperLogLog.

.. code:: python

    h1 = HyperLogLog()
    h2 = HyperLogLog()
    h1.update('test'.encode('utf8'))
    # The makes h1 the union of h2 and the original h1.
    h1.merge(h2)
    # This will return the cardinality of the union
    h1.count()

HyperLogLog++
-------------

`Documentation <https://ekzhu.github.io/datasketch/documentation.html#datasketch.HyperLogLogPlusPlus>`__

`HyperLogLog++ <http://research.google.com/pubs/pub40671.html>`__ is an
enhanced version of HyperLogLog by Google with the following changes: \*
Use 64-bit hash values instead of the 32-bit used by HyperLogLog \* A
more stable bias correction scheme based on experiments on many datasets
\* Sparse representation (not implemented here)

HyperLogLog++ object shares the same interface as HyperLogLog. So you
can use all the HyperLogLog functions in HyperLogLog++.

.. code:: python

    from datasketch import HyperLogLogPlusPlus

    # Initialize an HyperLogLog++ object.
    hpp = HyperLogLogPlusPlus()
    # Everything else is the same as HyperLogLog

Serialization
-------------

All data sketches supports efficient serialization using Python's
``pickle`` module. For example:

.. code:: python

    import pickle

    m1 = MinHash()
    # Serialize the MinHash objects to bytes
    bytes = pickle.dumps(m1)
    # Reconstruct the serialized MinHash object
    m2 = pickle.loads(bytes)
    # m1 and m2 should be equal
    print(m1 == m2)

.. |Build Status| image:: https://travis-ci.org/ekzhu/datasketch.svg?branch=master
   :target: https://travis-ci.org/ekzhu/datasketch
.. |DOI| image:: https://zenodo.org/badge/32555448.svg
   :target: https://zenodo.org/badge/latestdoi/32555448
