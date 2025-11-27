.. _minhash:

MinHash
=======

:py:class:`datasketch.MinHash` lets you estimate the `Jaccard
similarity <https://en.wikipedia.org/wiki/Jaccard_index>`__
(resemblance) between
`sets <https://en.wikipedia.org/wiki/Set_(mathematics)>`__ of
arbitrary sizes in linear time using a small and fixed memory space. It
can also be used to compute Jaccard similarity between data streams.
MinHash is introduced by Andrei Z. Broder in this
`paper <http://cs.brown.edu/courses/cs253/papers/nearduplicate.pdf>`__.

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
usage, because using more permutation functions means:

1. more CPU instructions for every data value hashed, and
2. more hash values to be stored.

The speed and memory usage of MinHash are both linearly
proportional to the number of permutation functions used.

.. figure:: /_static/hashfunc/minhash_benchmark_sha1.png
   :alt: MinHash Benchmark
   :align: center

You can union two MinHash objects using the
:py:meth:`datasketch.MinHash.merge` function. This
makes MinHash useful in parallel MapReduce-style data analysis.

.. code:: python

    # This makes m1 the union of m2 and the original m1.
    m1.merge(m2)

MinHash can be used for estimating the number of distinct elements, or
cardinality. The analysis is presented in `Cohen
1994 <http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=365694>`__.

.. code:: python

    # Returns the estimation of the cardinality of
    # all data values seen so far.
    m.count()

If you are handling billions of MinHash objects, consider using
:py:class:`datasketch.LeanMinHash` to reduce your memory footprint.

Use Different Hash Functions
----------------------------

MinHash by default uses the SHA1 hash function from Python’s built-in
`hashlib <https://docs.python.org/3.7/library/hashlib.html>`__ library.
You can also change the hash function using the ``hashfunc`` parameter
in the constructor.

.. code:: python

    # Let's use MurmurHash3.
    import mmh3

    # We need to define a new hash function that outputs an integer that
    # can be encoded in 32 bits.
    def _hash_func(d):
        return mmh3.hash32(d)

    # Use this function in MinHash constructor.
    m = MinHash(hashfunc=_hash_func)

Different hash functions have different performance–accuracy trade-offs;
you can use the benchmark code in ``benchmark/minhash_benchmark.py`` to
run some tests. Here are the results for some popular hash functions
available in Python.

MurmurHash3: `mmh3 <https://pypi.org/project/mmh3/>`__

.. figure:: /_static/hashfunc/minhash_benchmark_mmh3.png
   :alt: MinHash Benchmark (mmh3)
   :align: center

`xxhash <https://pypi.org/project/xxhash/>`__

.. figure:: /_static/hashfunc/minhash_benchmark_xxh.png
   :alt: MinHash Benchmark (xxhash)
   :align: center

`Farmhash <https://pypi.org/project/pyfarmhash>`__

.. figure:: /_static/hashfunc/minhash_benchmark_farmhash.png
   :alt: MinHash Benchmark (farmhash)
   :align: center


GPU usage (experimental)
------------------------

:py:class:`datasketch.MinHash` can optionally run part of
:py:meth:`datasketch.MinHash.update_batch` on a CUDA GPU
via `CuPy <https://cupy.dev/>`_. Hashing and permutation *generation* remain on CPU;
only the permutation application and min-reduction may use the GPU.

Control behavior with the constructor argument ``gpu_mode``:

- ``'disable'`` (default): always CPU.
- ``'detect'``: use GPU if available, otherwise fallback to CPU.
- ``'always'``: require GPU; raises ``RuntimeError`` if no CUDA device is available.

.. code-block:: python

    # Force CPU only
    m = MinHash(num_perm=256, gpu_mode="disable")
    m.update_batch(data)

.. code-block:: python

    # Require GPU (raises RuntimeError if no CUDA device)
    m = MinHash(num_perm=256, gpu_mode="always")
    m.update_batch(data)

**Install.** CuPy is an optional dependency (not installed by default).
See CuPy’s docs for wheels matching your CUDA version, e.g.
``pip install cupy-cuda12x`` for CUDA 12.

GPU runtime comparisons
-----------------------

The optional GPU backend accelerates the permutation-and-min step in
:py:meth:`datasketch.MinHash.update_batch`. Hashing still runs on CPU to
preserve ``hashfunc`` semantics.

**Runtime vs number of permutations (fixed ``n``).**

.. figure:: /_static/minhash_gpu/minhash_gpu_size_50000.png
   :alt: Runtime comparison CPU vs GPU across num_perm
   :align: center

   Update time for a fixed batch size (``n = 50,000``) as ``num_perm`` grows.
   GPU shows increasing speedups as ``num_perm`` increases because more parallel
   work amortizes transfer/launch overhead.

**Runtime vs batch size ``n`` (fixed ``num_perm``).**

.. figure:: /_static/minhash_gpu/minhash_gpu_vs_size_k256.png
   :alt: Runtime comparison CPU vs GPU across n
   :align: center

   Update time for a fixed number of permutations (e.g., ``num_perm = 256``)
   as batch size increases. Small batches are often CPU-favorable due to
   transfer overhead; larger batches benefit from GPU parallelism.

**Notes.** Results vary by hardware and CuPy version. The GPU backend is
controlled by ``gpu_mode``: ``"disable"`` (CPU), ``"detect"`` (use GPU if
available, otherwise CPU), and ``"always"`` (require GPU; raises if unavailable).

Common Issues with MinHash
--------------------------

1. `High estimation error when set sizes differ by a lot <https://github.com/ekzhu/datasketch/issues/85>`__
2. `Use Inclusion–Exclusion Principle (i.e., merge() and count() functions) instead of jaccard() to estimate similarity <https://github.com/ekzhu/datasketch/issues/85>`__
3. `Storing MinHash for later use <https://github.com/ekzhu/datasketch/issues/122>`__

`See more issues <https://github.com/ekzhu/datasketch/issues?utf8=%E2%9C%93&q=minhash>`__
