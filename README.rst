datasketch: Big Data Looks Small
================================

.. image:: https://static.pepy.tech/badge/datasketch/month
    :target: https://pepy.tech/project/datasketch

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.598238.svg
   :target: https://zenodo.org/doi/10.5281/zenodo.598238

datasketch gives you probabilistic data structures that can process and
search very large amount of data super fast, with little loss of
accuracy.

This package contains the following data sketches:

+-------------------------+-----------------------------------------------+
| Data Sketch             | Usage                                         |
+=========================+===============================================+
| `MinHash`_              | estimate Jaccard similarity and cardinality   |
+-------------------------+-----------------------------------------------+
| `Weighted MinHash`_     | estimate weighted Jaccard similarity          |
+-------------------------+-----------------------------------------------+
| `HyperLogLog`_          | estimate cardinality                          |
+-------------------------+-----------------------------------------------+
| `HyperLogLog++`_        | estimate cardinality                          |
+-------------------------+-----------------------------------------------+

The following indexes for data sketches are provided to support
sub-linear query time:

+---------------------------+-----------------------------+------------------------+
| Index                     | For Data Sketch             | Supported Query Type   |
+===========================+=============================+========================+
| `MinHash LSH`_            | MinHash, Weighted MinHash   | Jaccard Threshold      |
+---------------------------+-----------------------------+------------------------+
| `MinHash LSH Forest`_     | MinHash, Weighted MinHash   | Jaccard Top-K          |
+---------------------------+-----------------------------+------------------------+
| `MinHash LSH Ensemble`_   | MinHash                     | Containment Threshold  |
+---------------------------+-----------------------------+------------------------+
| `HNSW`_                   | Any                         | Custom Metric Top-K    |
+---------------------------+-----------------------------+------------------------+

datasketch must be used with Python 3.7 or above, NumPy 1.11 or above, and Scipy. 

Note that `MinHash LSH`_ and `MinHash LSH Ensemble`_ also support Redis and Cassandra 
storage layer (see `MinHash LSH at Scale`_).

Install
-------

To install datasketch using ``pip``:

::

    pip install datasketch

This will also install NumPy as dependency.

To install with Redis dependency:

::

    pip install datasketch[redis]

To install with Cassandra dependency:

::

    pip install datasketch[cassandra]


.. _`MinHash`: https://ekzhu.github.io/datasketch/minhash.html
.. _`Weighted MinHash`: https://ekzhu.github.io/datasketch/weightedminhash.html
.. _`HyperLogLog`: https://ekzhu.github.io/datasketch/hyperloglog.html
.. _`HyperLogLog++`: https://ekzhu.github.io/datasketch/hyperloglog.html#hyperloglog-plusplus
.. _`MinHash LSH`: https://ekzhu.github.io/datasketch/lsh.html
.. _`MinHash LSH Forest`: https://ekzhu.github.io/datasketch/lshforest.html
.. _`MinHash LSH Ensemble`: https://ekzhu.github.io/datasketch/lshensemble.html
.. _`Minhash LSH at Scale`: http://ekzhu.github.io/datasketch/lsh.html#minhash-lsh-at-scale
.. _`HNSW`: https://ekzhu.github.io/datasketch/documentation.html#hnsw
