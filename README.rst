datasketch: Big Data Looks Small
================================

.. image:: https://travis-ci.org/ekzhu/datasketch.svg?branch=master
    :target: https://travis-ci.org/ekzhu/datasketch
.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.290602.svg
   :target: https://doi.org/10.5281/zenodo.290602

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

datasketch must be used with Python 2.7 or above and NumPy 1.11 or
above. Scipy is optional, but with it the LSH initialization can be much
faster.

Note that `MinHash LSH`_ also supports a Redis storage layer as well 
as an experimental module with asynchronous interface to 
MongoDB.

Install
-------

To install datasketch using ``pip``:

::

    pip install datasketch -U

This will also install NumPy as dependency.

.. _`MinHash`: https://ekzhu.github.io/datasketch/minhash.html
.. _`Weighted MinHash`: https://ekzhu.github.io/datasketch/weightedminhash.html
.. _`HyperLogLog`: https://ekzhu.github.io/datasketch/hyperloglog.html
.. _`HyperLogLog++`: https://ekzhu.github.io/datasketch/hyperloglog.html#hyperloglog-plusplus
.. _`MinHash LSH`: https://ekzhu.github.io/datasketch/lsh.html
.. _`MinHash LSH Forest`: https://ekzhu.github.io/datasketch/lshforest.html
.. _`MinHash LSH Ensemble`: https://ekzhu.github.io/datasketch/lshensemble.html
