datasketch: Big Data Looks Small
================================

.. image:: https://travis-ci.org/ekzhu/datasketch.svg?branch=master
    :target: https://travis-ci.org/ekzhu/datasketch
.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.290602.svg
   :target: https://doi.org/10.5281/zenodo.290602

datasketch gives you probabilistic data structures that can process and
search vary large amount of data super fast, with little loss of
accuracy.

This package contains the following data sketches:

+-----------------------------+-----------------------------------------------+
| Data Sketch                 | Usage                                         |
+=============================+===============================================+
| :ref:`minhash`              | estimate Jaccard similarity and cardinality   |
+-----------------------------+-----------------------------------------------+
| :ref:`weighted_minhash`     | estimate weighted Jaccard similarity          |
+-----------------------------+-----------------------------------------------+
| :ref:`hyperloglog`          | estimate cardinality                          |
+-----------------------------+-----------------------------------------------+
| :ref:`hyperloglog_plusplus` | estimate cardinality                          |
+-----------------------------+-----------------------------------------------+

The following indexes for data sketches are provided to support
sub-linear query time:

+---------------------------+-----------------------------+------------------------+
| Index                     | For Data Sketch             | Supported Query Type   |
+===========================+=============================+========================+
| :ref:`minhash_lsh`        | MinHash, Weighted MinHash   | Radius (Threshold)     |
+---------------------------+-----------------------------+------------------------+
| :ref:`minhash_lsh_forest` | MinHash, Weighted MinHash   | Top-K                  |
+---------------------------+-----------------------------+------------------------+

datasketch must be used with Python 2.7 or above and NumPy 1.11 or
above. Scipy is optional, but with it the LSH initialization can be much
faster.

Install
-------

To install datasketch using ``pip``:

::

    pip install datasketch -U

This will also install NumPy as dependency.

