.. _lsh_bloom:

LSHBloom
===========

LSHBloom is a space-efficient alternative to :ref:`minhash_lsh` that uses `Bloom filters <https://en.wikipedia.org/wiki/Bloom_filter>`__
to significantly reduce disk usage as well as insertion and query speed. The details
of this algorithm can be found in the `LSHBloom paper <https://arxiv.org/abs/2411.04257>`__.

Like MinHash LSH, LSHBloom supports approximate Jaccard similarity matches. However, 
whereas MinHash LSH returns a series of candidate duplicates for each query set, LSHBloom 
can only return a binary signal (0 or 1) as to whether the given query set is duplicated.
That is, one loses the ability to manually filter candidate pairs. Additionally, one must
provide an estimate of the dataset size and the acceptable false positive overhead per Bloom filter
ahead of time to properly allocate the Bloom filter index. 

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

Error Overhead
-----------

Because LSHBloom uses Bloom filters to save space, it also introduces some additonal 
false positives. In particular, LSHBloom introduces a new parameter, *fp*, which is the
allowable false positive rate for each Bloom filter in the index. Decreasing the false positive rate
will increase the size of the index and vice versa. Given a MinHash signature matrix with *b* bands,
a dataset of *n* sets, and a desired false positive probability per Bloom filter of *fp* the space 
usage of LSHBloom in bits will be:

.. math::
   m = b * \frac{-n \ln(fp)}{(\ln 2)^2}

The false positive overhead per Bloom filter also alters the false positive and negative rates compared
to MinHash LSH. In particular, for a given LSHBloom index the net false positive overhead will be

.. math::
   p = 1 - (1 - fp)^b

For MinHash LSH, the false positive and false negative rates respectively are:

.. math::
    FP = \int_{0}^{T} 1 - (1 - t^r)^b dt 
    \\
    FN = \int_{T}^{1} 1 - (1 - (1 - t^r)^b) dt

Where *T* is the Jaccard similarity threshold, *b* is the number of bands in the MinHash signature matrix,
and *r* is the band size. 

The effective false positive and false negative rates for LSHBloom are then:

.. math::
    FP_{bloom} = FP + (1 - FP) * p
    \\
    FN_{bloom} = (1 - p) * FN

So LSHBloom will slightly increase the false positive rate and slightly decrease the false negative rate,
depending on the value of p.