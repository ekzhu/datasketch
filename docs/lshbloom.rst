.. _lsh_bloom:

LSHBloom
===========

This algorithm requires the Bloom filter dependency for datasketch:

::

    pip install datasketch[bloom]

Suppose that you want to filter a collection of sets such that the filtered collection consists of those
sets that do not exceed a certain Jaccard similarity threshold, *T*, with any other set in the collection.
We can perform this kind of task quickly and efficiently with MinHashes and LSHBloom.

LSHBloom is a space-efficient alternative to :ref:`minhash_lsh` that operates on the MinHash signature matrix
(like MinHash LSH) but replaces the traditional LSHIndex with `Bloom filters <https://en.wikipedia.org/wiki/Bloom_filter>`__
to significantly reduce disk usage as well as insertion and query speed.

Like MinHash LSH, LSHBloom supports approximate Jaccard similarity matches and has a sub-linear query cost. However, 
whereas MinHash LSH returns a series of candidate duplicates for each query set, LSHBloom 
can only return a binary signal (0 or 1) as to whether the given query set is duplicated (that is,
whether it has a Jaccard similarity :math:`>= T` with any other set in the collection).
With LSHBloom, one loses the ability to manually filter candidate pairs of duplicates, but benefits from
significantly increased resource efficiency as a result. Another difference with MinHash LSH is that one must
provide an estimate of the dataset size and the acceptable false positive overhead per Bloom filter
ahead of time to properly allocate the Bloom filter index. For information on these settings see :ref:`bloom-error`
below.

Further details can be found in the `LSHBloom paper <https://arxiv.org/abs/2411.04257>`__.


.. code:: python
        
        from datasketch import MinHash, MinHashLSHBloom

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

        # Create LSHBloom index
        lsh = MinHashLSHBloom(threshold=0.5, num_perm=128, n=100, fp=0.001)
        lsh.insert("m2", m2)
        lsh.insert("m3", m3)
        is_duplicate = lsh.query(m1)
        print("Is Duplicate: ", is_duplicate)

.. _bloom-error:
Error Overhead
-----------

Because LSHBloom uses Bloom filters to save space, it also introduces some additonal 
false positives. In particular, LSHBloom introduces a new parameter, *fp*, which is the
allowable false positive rate for each Bloom filter in the index. Decreasing the false positive rate
will increase the size of the index and vice versa. Given a MinHash signature matrix with *b* bands,
a dataset of *n* sets, and a desired false positive probability per Bloom filter of *fp* the space 
usage of LSHBloom will be:

.. math::
   m = b * \frac{-n \ln(fp)}{(\ln 2)^2} \text{ bits}

The false positive overhead per Bloom filter also alters the false positive and negative rates compared
to MinHash LSH. In particular, for a given LSHBloom index the net false positive overhead will be

.. math::
   p = 1 - (1 - fp)^b

For MinHash LSH, the false positive and false negative rates respectively are:

.. math::
    FP_{MinHash} = \int_{0}^{T} 1 - (1 - t^r)^b dt 
    \\
    FN_{MinHash} = \int_{T}^{1} 1 - (1 - (1 - t^r)^b) dt

Where *T* is the Jaccard similarity threshold, *b* is the number of bands in the MinHash signature matrix,
and *r* is the band size. 

The effective false positive and false negative rates for LSHBloom are then:

.. math::
    FP_{Bloom} = FP_{MinHash} + (1 - FP_{MinHash}) * p
    \\
    FN_{Bloom} = (1 - p) * FN_{MinHash}

So LSHBloom will slightly increase the false positive rate and slightly decrease the false negative rate,
depending on the value of p.