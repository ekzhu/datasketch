from __future__ import annotations

import logging
import os
import warnings
from typing import Optional

import numpy as np
from scipy.integrate import quad as integrate

from datasketch.minhash import MinHash

try:
    import pybloomfilter
except ImportError:
    pybloomfilter = None

logger = logging.getLogger(__name__)

_mersenne_prime = np.uint64((1 << 61) - 1)


def _false_positive_probability(threshold, b, r):
    _probability = lambda s: 1 - (1 - s ** float(r)) ** float(b)
    a, _err = integrate(_probability, 0.0, threshold)
    return a


def _false_negative_probability(threshold, b, r):
    _probability = lambda s: 1 - (1 - (1 - s ** float(r)) ** float(b))
    a, _err = integrate(_probability, threshold, 1.0)
    return a


def _optimal_param(threshold, num_perm, false_positive_weight, false_negative_weight):
    """Compute the optimal `MinHashLSH` parameter that minimizes the weighted sum
    of probabilities of false positive and false negative.
    """
    min_error = float("inf")
    opt = (0, 0)
    for b in range(1, num_perm + 1):
        max_r = int(num_perm / b)
        for r in range(1, max_r + 1):
            fp = _false_positive_probability(threshold, b, r)
            fn = _false_negative_probability(threshold, b, r)
            error = fp * false_positive_weight + fn * false_negative_weight
            if error < min_error:
                min_error = error
                opt = (b, r)
    return opt


if pybloomfilter is not None:

    class BloomTable:
        """Interface to a Bloom Filter meant to model a single band of the MinHash signature matrix.

        Args:
                item_count (int): Number of items expected to be inserted
                    (size of dataset). Used to create Bloom filter.
                fp (float): False positive rate for Bloom filter in (0,1).
                band_size (int): Size of band from MinHash signature matrix
                    this filter is meant to model.
                fname (str): File path where Bloom filter will be saved. If this file
                    already exists, will initialize the Bloom filter from this path.
                max_size (int): Maximum number of elements we should plan to insert
                    into this Bloom filter. Upper bounds the size of the Bloom filter.

        """

        def __init__(self, item_count: int, fp: float, band_size: int, fname: Optional[str] = None):
            self.r = band_size
            self.fname = fname
            if fname is not None and os.path.exists(fname):
                logger.info(f"Loading Bloom Filter at {fname}...")
                self.bloom_filter = pybloomfilter.BloomFilter.open(fname)
            else:
                self.bloom_filter = pybloomfilter.BloomFilter(capacity=item_count, error_rate=fp, filename=self.fname)

        def sync(self):
            if self.fname is not None:
                self.bloom_filter.sync()
            else:
                warnings.warn(
                    "Attempting to save in-memory Bloom filter, this is a no-op.", RuntimeWarning, stacklevel=2
                )

        def assert_size(self, hashvalues: list[int]):
            if not len(hashvalues) == self.r:
                raise RuntimeError(
                    f"Invalid length for indices, {len(hashvalues)}, expected {self.r} hashvalues in band"
                )

        def insert(self, hashvalues: list[int]) -> None:
            """Takes as input the indices for a single band and inserts them into the corresponding bit arrays.

            Args:
                    hashvalues (list[int]): The hashvalues from a single band of a MinHash object.

            """
            self.assert_size(hashvalues)
            # https://en.wikipedia.org/wiki/Universal_hashing#Hashing_vectors
            # as the hashvalues are the result of a universal hashing function,
            # their sum is also a universal hash function
            x = sum(hashvalues) % _mersenne_prime
            self.bloom_filter.add(x)

        def query(self, hashvalues: list[int]) -> bool:
            """Takes as input the indices for a single band and queries them against the corresponding arrays
            returns True if the each query returns True, otherwise returns False.

            Args:
                    hashvalues (list[int]): The hashvalues from a single band of a MinHash object.

            """
            self.assert_size(hashvalues)
            x = sum(hashvalues) % _mersenne_prime
            return x in self.bloom_filter
else:

    class BloomTable:
        def __init__(self, item_count: int, fp: float, band_size: int, fname: Optional[str] = None):
            raise ImportError("Required dependency pybloomfilter is missing, did you `pip install datasketch[bloom]`?")


class MinHashLSHBloom:
    """The :ref:`lsh_bloom` index.
    It supports query with `Jaccard similarity`_ threshold.
    Reference: `LSHBloom paper
    <https://arxiv.org/abs/2411.04257>`_.

    Args:
            threshold (float): The Jaccard similarity threshold between 0.0 and
                    1.0. The initialized LSH index will be optimized for the threshold by
                    minizing the false positive and false negative.
            num_perm (int): The number of permutation functions used
                    by the MinHash to be indexed. For weighted MinHash, this
                    is the sample size (`sample_size`).
            n (int): The number of elements to be inserted (estimate of dataset size).
            fp (float): The false positive rate for each Bloom filter. Must be in (0,1).
            save_dir (str): The directory to save the Bloom filter index to. If Bloom filters
                    already exist in this directory, the index will be loaded from here. If None,
                    an in-memory index will be created - this index can not be persisted.
            weights (Tuple[float, float]): Used to adjust the relative importance of
                    minimizing false positive and false negative when optimizing
                    for the Jaccard similarity threshold.
                    `weights` is a tuple in the format of
                    :code:`(false_positive_weight, false_negative_weight)`.
            params (Optiona[Tuple[int, int]]): The LSH parameters (i.e., number of bands and size
                    of each bands). This is used to bypass the parameter optimization
                    step in the constructor. `threshold` and `weights` will be ignored
                    if this is given.

    Note:
            This algorithm is a space optimized version of MinHashLSH.
            For more details on :ref:`minhash_lsh`, see the documentation.

            This algorithm uses Bloom filters to drastically reduce the space
            that the LSH index occupies on disk. However, it loses the ability
            to retrieve candidate duplicate keys. Rather, it can only tell you
            whether a query set is a duplicate of a set that was inserted previously.
            This enables scaling to datasets of many hundreds of millions or billions
            of documents, but may not be appropriate for all use cases.

    Examples:
            Create an index with 128 permutation functions optimized for Jaccard
            threshold 0.9:

            .. code-block:: python

                    from datasketch import MinHash, MinHashLSH

                    set1 = set(
                        [
                            "minhash",
                            "is",
                            "a",
                            "probabilistic",
                            "data",
                            "structure",
                            "for",
                            "estimating",
                            "the",
                            "similarity",
                            "between",
                            "datasets",
                        ]
                    )
                    set2 = set(
                        [
                            "minhash",
                            "is",
                            "a",
                            "probability",
                            "data",
                            "structure",
                            "for",
                            "estimating",
                            "the",
                            "similarity",
                            "between",
                            "documents",
                        ]
                    )
                    set3 = set(
                        [
                            "minhash",
                            "is",
                            "probability",
                            "data",
                            "structure",
                            "for",
                            "estimating",
                            "the",
                            "similarity",
                            "between",
                            "documents",
                        ]
                    )

                    m1 = MinHash(num_perm=128)
                    m2 = MinHash(num_perm=128)
                    m3 = MinHash(num_perm=128)
                    for d in set1:
                        m1.update(d.encode("utf8"))
                    for d in set2:
                        m2.update(d.encode("utf8"))
                    for d in set3:
                        m3.update(d.encode("utf8"))

                    # Create LSHBloom index
                    lsh = MinHashLSHBloom(threshold=0.5, num_perm=128, n=100, fp=0.0001, save_dir="./index/")
                    lsh.insert(m2)
                    lsh.insert(m3)

                    # Query whether m1 is a duplicate according to the given threshold
                    is_duplicate = lsh.query(m1)

    """

    def __init__(
        self,
        threshold: float = 0.9,
        num_perm: int = 128,
        n: Optional[int] = None,
        fp: Optional[float] = None,
        save_dir: Optional[str] = None,
        weights: tuple[float, float] = (0.5, 0.5),
        params: Optional[tuple[int, int]] = None,
    ) -> None:
        if threshold > 1.0 or threshold < 0.0:
            raise ValueError("threshold must be in [0.0, 1.0]")
        if num_perm < 2:
            raise ValueError("Too few permutation functions")
        if n <= 0:
            raise ValueError("n for LSHBloom must be >= 0")
        if fp >= 1.0 or fp <= 0.0:
            raise ValueError("fp must be in (0.0, 1.0)")
        if save_dir is None:
            warnings.warn(
                "Creating LSHBloom index without save directory, this index will not be persisted.",
                RuntimeWarning,
                stacklevel=2,
            )
        if any(w < 0.0 or w > 1.0 for w in weights):
            raise ValueError("Weight must be in [0.0, 1.0]")
        if sum(weights) != 1.0:
            raise ValueError("Weights must sum to 1.0")
        self.h = num_perm
        if params is not None:
            self.b, self.r = params
            if self.b * self.r > num_perm:
                raise ValueError(
                    "The product of b and r in params is "
                    f"{self.b} * {self.r} = {self.b * self.r} -- it must be less than num_perm {num_perm}. "
                    "Did you forget to specify num_perm?"
                )
        else:
            false_positive_weight, false_negative_weight = weights
            self.b, self.r = _optimal_param(threshold, num_perm, false_positive_weight, false_negative_weight)
        if self.b < 2:
            raise ValueError("The number of bands are too small (b < 2)")

        # create a Bloom filter for each band in the signature matrix
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
        self.hashtables = [
            BloomTable(
                item_count=n,
                fp=fp,
                band_size=self.r,
                fname=os.path.join(save_dir, f"band-{i}.bf") if save_dir is not None else None,
            )
            for i in range(self.b)
        ]
        self.hashranges = [(i * self.r, (i + 1) * self.r) for i in range(self.b)]

    def insert(self, minhash: MinHash):
        """Insert the MinHash or Weighted MinHash
        of a set to the index.

        Args:
                minhash (Union[MinHash, WeightedMinHash]): The MinHash of the set.

        """
        self._insert(minhash)

    def _insert(self, minhash: MinHash):
        if len(minhash) != self.h:
            raise ValueError("Expecting minhash with length %d, got %d" % (self.h, len(minhash)))

        Hs = [minhash.hashvalues[start:end] for start, end in self.hashranges]

        for H, hashtable in zip(Hs, self.hashtables):
            hashtable.insert(H)

    def query(self, minhash) -> bool:
        """Given the MinHash of the query set, determine
        whether any previously inserted sets have
        Jaccard similarity with the query that is
        likely greater than the threshold.

        Results are based on minhash segment collision
        and are thus approximate.

        Args:
                minhash (MinHash): The MinHash of the query set.

        Returns:
                bool: Whether the item is a duplicate or not, based on the given threshold.

        Example:

                .. code-block:: python

                        from datasketch import MinHash, MinHashLSHBloom
                        import numpy as np

                        # Generate 100 random MinHashes.
                        minhashes = MinHash.bulk(np.random.randint(low=0, high=30, size=(100, 10)), num_perm=128)

                        # Create LSHBloom index.
                        lsh = MinHashLSHBloom(threshold=0.5, num_perm=128, n=100, fp=0.0001, save_dir="./index/")
                        for i, m in enumerate(minhashes):
                            lsh.insert(i, m)

                        # Get the duplication result from LSHBloom.
                        query = minhashes[0]
                        is_duplicate = lsh.query(query)
                        print(is_duplicate)

                Output:

                .. code-block::

                        True

                Note that although the threshold is set to 0.5, the results are not
                guaranteed to be above 0.5 because the LSHBloom index is approximate and
                the Jaccard similarity is estimated by MinHash.

        """
        if len(minhash) != self.h:
            raise ValueError("Expecting minhash with length %d, got %d" % (self.h, len(minhash)))

        # if we match in any band, this is a candidate pair
        for (start, end), hashtable in zip(self.hashranges, self.hashtables):
            H = minhash.hashvalues[start:end]
            collision = hashtable.query(H)
            if collision:
                return True
        return False

    def sync(self):
        logger.info("Saving Bloom Index...")
        for table in self.hashtables:
            table.sync()
