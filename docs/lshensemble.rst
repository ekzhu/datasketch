.. _minhash_lsh_ensemble:

MinHash LSH Ensemble
====================

.. _containment:

Containment
-----------

`Jaccard similarity <https://en.wikipedia.org/wiki/Jaccard_index>`_ is great for
measuring resemblance between two sets, however, it can be a biased measure for 
set intersection. This issue can be illustrated in the following Venn diagrams.
The left pair (Q and X) have roughly the same intersection size as the right 
pair (Q' and X').
But the Jaccard similarities, computed as 
:math:`\frac{|Q \cap X|}{|Q \cup X|}` and :math:`\frac{|Q' \cap X'|}{|Q' \cup X'|}`
respectively,
are very different, with the latter being much smaller, because its union
size :math:`|Q' \cup X'|` is much larger.
This shows that Jaccard similarity is biased when measuring intersection size, 
as large sets are penalized. 

.. figure:: /_static/containment.png
   :alt: MinHashLSHEnsemble Benchmark
   :scale: 80 %
   :align: center

We can use a better measure for intersection, called *containment*.
It is computed as the intersection size divided by the size of one of the set,
in this case Q.

.. math::
        Containment = \frac{|Q \cap X|}{|Q|}

It soon becomes clear why we use Q in the denominator, when you think about this 
search problem: suppose you have a large collection of sets, given a 
query, which is also a set, you want to find sets in your collection that have 
intersection with the query above a certain threshold.
We can think of the query as set Q, and an arbitrary set in the collection as X.
Since the query set here is fixed (for a specific search problem), the
denominator :math:`|Q|` is a constant, and the intersection size is solely determined
by the containment. Thus, we can search for sets in the collection with containment
above a certain threshold instead, and the containment threshold can be easily
deduced from the intersection threshold by multiplying :math:`|Q|`.

Another way to think about containment: it is a "normalized" intersection
(with value between 0 and 1), which measures the fraction of the query set Q
contained in X.

LSH for Containment
-------------------

Similar to :ref:`minhash_lsh`, there is an LSH index for containment search --
given a query set, find sets with containment above a certain threshold.
It is *LSH Ensemble* by `E. Zhu et al <http://www.vldb.org/pvldb/vol9/p1185-zhu.pdf>`_.
This package implements a slightly simplified version of the index,
:class:`datasketch.MinHashLSHEnsemble`.
The full implementation is in `Go <https://golang.org/>`_. It can be found at 
`github.com/ekzhu/lshensemble <https://github.com/ekzhu/lshensemble>`_.

Just like MinHash LSH, LSH Ensemble also works directly with :ref:`minhash` data sketches.

.. code:: python

        from datasketch import MinHashLSHEnsemble, MinHash

        set1 = set(["cat", "dog", "fish", "cow"])
        set2 = set(["cat", "dog", "fish", "cow", "pig", "elephant", "lion", "tiger",
                     "wolf", "bird", "human"])
        set3 = set(["cat", "dog", "car", "van", "train", "plane", "ship", "submarine",
                     "rocket", "bike", "scooter", "motorcyle", "SUV", "jet", "horse"])

        # Create MinHash objects
        m1 = MinHash(num_perm=128)
        m2 = MinHash(num_perm=128)
        m3 = MinHash(num_perm=128)
        for d in set1:
            m1.update(d.encode('utf8'))
        for d in set2:
            m2.update(d.encode('utf8'))
        for d in set3:
            m3.update(d.encode('utf8'))

        # Create an LSH Ensemble index with threshold and number of partition
        # settings.
        lshensemble = MinHashLSHEnsemble(threshold=0.8, num_perm=128, 
            num_part=32)

        # Index takes an iterable of (key, minhash, size)
        lshensemble.index([("m2", m2, len(set2)), ("m3", m3, len(set3))])

        # Check for membership using the key
        print("m2" in lshensemble)
        print("m3" in lshensemble)

        # Using m1 as the query, get an result iterator 
        print("Sets with containment > 0.8:")
        for key in lshensemble.query(m1, len(set1)):
            print(key)

Benchmarks
----------

LSH Ensemble partitions the indexed sets by size and tunes the LSH
parameters separately for each partition, so more partitions
(``num_part``) help in proportion to how *skewed* the set sizes are. The
benchmark below makes that concrete: containment threshold queries on a
synthetic corpus, with ``num_part`` on the x-axis and one figure per
index-size distribution, comparing the ensemble against a linear scan
(``benchmark/indexes/lshensemble_synthetic_benchmark.py``). Each query is
a fixed 200 tokens with planted neighbours spread across containments
(0.05--0.95); queries are held out of the index; ground truth is exact
containment at or above the threshold of 0.5. The linear scan converts
the MinHash Jaccard estimate of every indexed set to a containment
estimate and thresholds it -- it does not depend on ``num_part``, so it
is drawn as a reference line.

**No skew: partitioning cannot help.** When every indexed set is the
same size there is nothing to partition by, so the optimal partitioning
is degenerate and accuracy is flat across ``num_part``.

.. figure:: /_static/lshensemble_constant_benchmark.png
   :alt: MinHashLSHEnsemble vs. num_part, constant index sizes
   :align: center

**Moderate skew: a modest, saturating gain.** With log-uniform sizes
spanning 100 to 10,000 tokens, a single partition must compromise across
the whole range; partitioning recovers some precision (about 0.12 to
0.34 from one partition to 32) but with diminishing returns.

.. figure:: /_static/lshensemble_loguniform_benchmark.png
   :alt: MinHashLSHEnsemble vs. num_part, log-uniform index sizes
   :align: center

**Heavy skew: partitioning pays off strongly.** Real open-data corpora
have power-law cardinalities -- most sets small, a heavy tail of very
large ones (the shape the `LSH Ensemble paper
<http://www.vldb.org/pvldb/vol9/p1185-zhu.pdf>`_ models; here the median
set is ~170 tokens with a tail to 10,000). Precision climbs from 0.15
with one partition to 0.69 with 32 (a 4--5x gain), because optimal
partitioning confines the heavy tail of large sets -- whose equivalent
Jaccard thresholds are near zero and would otherwise flood every query
with candidates -- to a few partitions of their own, letting the many
small-set partitions run with tight parameters.

.. figure:: /_static/lshensemble_zipf_benchmark.png
   :alt: MinHashLSHEnsemble vs. num_part, Zipfian index sizes
   :align: center

Across all three, recall stays high (near or above the scan) while query
time grows with the partition count but remains well under the scan's --
so partitioning trades a little speed for precision. Two caveats when
reading the absolute numbers. The ensemble returns *candidates* --
everything that collided in some partition's bands -- so its precision is
meant to be lifted by verifying candidates against the query (with exact
containment, or the same estimate-and-convert the scan uses); the gap up
to the scan's reference line is what such a check recovers. And
containment of a small query in a much larger set is a tiny Jaccard
similarity, which MinHash estimates with large relative noise, so heavy
skew is intrinsically hard for any method built on these sketches --
more permutations (``num_perm``) soften it.

There are other optional parameters that can be used to tune the index;
see the documentation of :class:`datasketch.MinHashLSHEnsemble`.

Common Issues with LSH Ensemble
-------------------------------

1. `Can I get containment values from LSH Ensemble? <https://github.com/ekzhu/datasketch/issues/97>`__
2. `How to use LSH Ensemble to find containment-based duplicates? <https://github.com/ekzhu/datasketch/issues/76#issuecomment-521275633>`__

