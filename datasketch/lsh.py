from __future__ import annotations
import pickle
import struct
from typing import Callable, Dict, Hashable, List, Optional, Tuple, Union
from datasketch.minhash import MinHash
from datasketch.weighted_minhash import WeightedMinHash
from datasketch.storage import ordered_storage, unordered_storage, _random_name

from scipy.integrate import quad as integrate


def _false_positive_probability(threshold, b, r):
    _probability = lambda s: 1 - (1 - s ** float(r)) ** float(b)
    a, err = integrate(_probability, 0.0, threshold)
    return a


def _false_negative_probability(threshold, b, r):
    _probability = lambda s: 1 - (1 - (1 - s ** float(r)) ** float(b))
    a, err = integrate(_probability, threshold, 1.0)
    return a


def _optimal_param(threshold, num_perm, false_positive_weight, false_negative_weight):
    """
    Compute the optimal `MinHashLSH` parameter that minimizes the weighted sum
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


class MinHashLSH(object):
    """
    The :ref:`minhash_lsh` index.
    It supports query with `Jaccard similarity`_ threshold.
    Reference: `Chapter 3, Mining of Massive Datasets
    <http://www.mmds.org/>`_.

    Args:
        threshold (float): The Jaccard similarity threshold between 0.0 and
            1.0. The initialized MinHash LSH will be optimized for the threshold by
            minizing the false positive and false negative.
        num_perm (int): The number of permutation functions used
            by the MinHash to be indexed. For weighted MinHash, this
            is the sample size (`sample_size`).
        weights (Tuple[float, float]): Used to adjust the relative importance of
            minimizing false positive and false negative when optimizing
            for the Jaccard similarity threshold.
            `weights` is a tuple in the format of
            :code:`(false_positive_weight, false_negative_weight)`.
        params (Optiona[Tuple[int, int]]): The LSH parameters (i.e., number of bands and size
            of each bands). This is used to bypass the parameter optimization
            step in the constructor. `threshold` and `weights` will be ignored
            if this is given.
        storage_config (Optional[Dict]): Type of storage service to use for storing
            hashtables and keys.
            `basename` is an optional property whose value will be used as the prefix to
            stored keys. If this is not set, a random string will be generated instead. If you
            set this, you will be responsible for ensuring there are no key collisions.
        prepickle (Optional[bool]): If True, all keys are pickled to bytes before
            insertion. If not specified, a default value is chosen based on the
            `storage_config`.
        hashfunc (Optional[Callable[[bytes], bytes]]): If a hash function is provided it will be used to
            compress the index keys to reduce the memory footprint. This could cause a higher
            false positive rate.

    Note:
        `weights` must sum to 1.0, and the format is
        (false positive weight, false negative weight).
        For example, if minimizing false negative (or maintaining high recall) is more
        important, assign more weight toward false negative: weights=(0.4, 0.6).
        Try to live with a small difference between weights (i.e. < 0.5).

    Examples:

        Create an index with 128 permutation functions optimized for Jaccard
        threshold 0.9:

        .. code-block:: python

            lsh = MinHashLSH(threshold=0.9, num_perm=128)
            print(lsh.b, lsh.r)
            # 5 25

        The built-in optimizer will try to minimize the weighted sum of
        probabilities of false positive and false negative. The algorithm is
        a simple grid search over the space of possible parameters.

        Note that it is possible to get :attr:`b` (number of bands) and
        :attr:`r` (band size) that do not sum to :attr:`num_perm`, leading to
        unused permutation values in the indexed MinHash.
        This is because the optimizer only considers bands of
        the same size, and the number of bands is not necessarily a divisor of
        :attr:`num_perm`.

        Instead of using the built-in optimizer, you can customize the LSH
        parameters your self. The snippet below creates an index with 128
        permutation functions and 16 bands each with size 8, skipping the
        optimization step:

        .. code-block:: python

            lsh = MinHashLSH(num_perm=128, params=(16, 8))
            print(lsh.b, lsh.r)
            # 16 8

        Create an index backed by Redis storage:

        .. code-block:: python

            lsh = MinHashLSH(threshold=0.9, num_perm=128, storage_config={
                'type': 'redis',
                'basename': b'mylsh', # optional, defaults to a random string.
                'redis': {'host': 'localhost', 'port': 6379},
            })

        The `basename` property is optional. It is used to generate key prefixes
        in the storage layer to uniquely identify data associated with this LSH.
        Thus, if you create a new LSH object with the same `basename`, you will
        be using the same underlying data in the storage layer associated with
        a previous LSH object. If you do not set this property, a random string
        will be generated instead.

    """

    def __init__(
        self,
        threshold: float = 0.9,
        num_perm: int = 128,
        weights: Tuple[float, float] = (0.5, 0.5),
        params: Optional[Tuple[int, int]] = None,
        storage_config: Optional[Dict] = None,
        prepickle: Optional[bool] = None,
        hashfunc: Optional[Callable[[bytes], bytes]] = None,
    ) -> None:
        storage_config = {"type": "dict"} if not storage_config else storage_config
        self._buffer_size = 50000
        if threshold > 1.0 or threshold < 0.0:
            raise ValueError("threshold must be in [0.0, 1.0]")
        if num_perm < 2:
            raise ValueError("Too few permutation functions")
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
                    "{} * {} = {} -- it must be less than num_perm {}. "
                    "Did you forget to specify num_perm?".format(
                        self.b, self.r, self.b * self.r, num_perm
                    )
                )
        else:
            false_positive_weight, false_negative_weight = weights
            self.b, self.r = _optimal_param(
                threshold, num_perm, false_positive_weight, false_negative_weight
            )
        if self.b < 2:
            raise ValueError("The number of bands are too small (b < 2)")

        self.prepickle = (
            storage_config["type"] == "redis" if not prepickle else prepickle
        )

        self.hashfunc = hashfunc
        if hashfunc:
            self._H = self._hashed_byteswap
        else:
            self._H = self._byteswap

        basename = storage_config.get("basename", _random_name(11))
        self.hashtables = [
            unordered_storage(
                storage_config,
                name=b"".join([basename, b"_bucket_", struct.pack(">H", i)]),
            )
            for i in range(self.b)
        ]
        self.hashranges = [(i * self.r, (i + 1) * self.r) for i in range(self.b)]
        self.keys = ordered_storage(storage_config, name=b"".join([basename, b"_keys"]))

    @property
    def buffer_size(self) -> int:
        return self._buffer_size

    @buffer_size.setter
    def buffer_size(self, value: int) -> None:
        self.keys.buffer_size = value
        for t in self.hashtables:
            t.buffer_size = value
        self._buffer_size = value

    def insert(
        self,
        key: Hashable,
        minhash: Union[MinHash, WeightedMinHash],
        check_duplication: bool = True,
    ):
        """
        Insert a key to the index, together with a MinHash or Weighted MinHash
        of the set referenced by the key.

        Args:
            key (Hashable): The unique identifier of the set.
            minhash (Union[MinHash, WeightedMinHash]): The MinHash of the set.
            check_duplication (bool): To avoid duplicate keys in the storage
                (`default=True`). It's recommended to not change the default, but
                if you want to avoid the overhead during insert you can set
                `check_duplication = False`.

        """
        self._insert(key, minhash, check_duplication=check_duplication, buffer=False)

    def merge(
            self,
            other: MinHashLSH,
            check_overlap: bool = False      
    ):
        """Merge the other MinHashLSH with this one, making this one the union
        of both.
        
        Note:
            Only num_perm, number of bands and sizes of each band is checked for equivalency of two MinHashLSH indexes.
            Other initialization parameters threshold, weights, storage_config, prepickle and hash_func are not checked.

        Args:
            other (MinHashLSH): The other MinHashLSH.
            check_overlap (bool): Check if there are any overlapping keys before merging and raise if there are any.
                (`default=False`)

        Raises:
            ValueError: If the two MinHashLSH have different initialization
                parameters, or if `check_overlap` is `True` and there are overlapping keys.
        """
        self._merge(other, check_overlap=check_overlap, buffer=False)

    def insertion_session(self, buffer_size: int = 50000) -> MinHashLSHInsertionSession:
        """
        Create a context manager for fast insertion into this index.

        Args:
            buffer_size (int): The buffer size for insert_session mode (default=50000).

        Returns:
            MinHashLSHInsertionSession: The context manager.

        Example:

            Insert 100 MinHashes into an Redis-backed index using a session:

            .. code-block:: python

                from datasketch import MinHash, MinHashLSH
                import numpy as np

                minhashes = []
                for i in range(100):
                    m = MinHash(num_perm=128)
                    m.update_batch(np.random.randint(low=0, high=30, size=10))
                    minhashes.append(m)

                lsh = MinHashLSH(threshold=0.5, num_perm=128, storage_config={
                    'type': 'redis',
                    'redis': {'host': 'localhost', 'port': 6379},
                })
                with lsh.insertion_session() as session:
                    for i, m in enumerate(minhashes):
                        session.insert(i, m)

        """
        return MinHashLSHInsertionSession(self, buffer_size=buffer_size)

    def _insert(
        self,
        key: Hashable,
        minhash: Union[MinHash, WeightedMinHash],
        check_duplication: bool = True,
        buffer: bool = False,
    ):
        if len(minhash) != self.h:
            raise ValueError(
                "Expecting minhash with length %d, got %d" % (self.h, len(minhash))
            )
        if self.prepickle:
            key = pickle.dumps(key)
        if check_duplication and key in self.keys:
            raise ValueError("The given key already exists")
        Hs = [self._H(minhash.hashvalues[start:end]) for start, end in self.hashranges]
        self.keys.insert(key, *Hs, buffer=buffer)
        for H, hashtable in zip(Hs, self.hashtables):
            hashtable.insert(H, key, buffer=buffer)

    def __equivalent(self, other:MinHashLSH) -> bool:
        """
        Returns:
            bool: If the two MinHashLSH have equal num_perm, number of bands, size of each band then two are equivalent.
        """
        return (
            type(self) is type(other) and
            self.h == other.h and
            self.b == other.b and
            self.r == other.r
        )

    def _merge(
        self,
        other: MinHashLSH,
        check_overlap: bool = False,
        buffer: bool = False
    ) -> MinHashLSH:
        if self.__equivalent(other):
            if check_overlap and set(self.keys).intersection(set(other.keys)):
                raise ValueError("The keys are overlapping, duplicate key exists.")
            for key in other.keys:
                Hs = other.keys.get(key)
                self.keys.insert(key, *Hs, buffer=buffer)
                for H, hashtable in zip(Hs, self.hashtables):
                    hashtable.insert(H, key, buffer=buffer)
        else:
            if type(self) is not type(other):
                raise ValueError(f"Cannot merge type MinHashLSH and type {type(other).__name__}.")
            raise ValueError(
                "Cannot merge MinHashLSH with different initialization parameters.")

    def query(self, minhash) -> List[Hashable]:
        """
        Giving the MinHash of the query set, retrieve
        the keys that reference sets with Jaccard
        similarities likely greater than the threshold.

        Results are based on minhash segment collision
        and are thus approximate. For more accurate results,
        filter again with :meth:`MinHash.jaccard`. For exact results,
        filter by computing Jaccard similarity using original sets.

        Args:
            minhash (MinHash): The MinHash of the query set.

        Returns:
            list: a list of unique keys.

        Example:

            Query and rank results using :meth:`MinHash.jaccard`.

            .. code-block:: python

                from datasketch import MinHash, MinHashLSH
                import numpy as np

                # Generate 100 random MinHashes.
                minhashes = MinHash.bulk(
                    np.random.randint(low=0, high=30, size=(100, 10)),
                    num_perm=128
                )

                # Create LSH index.
                lsh = MinHashLSH(threshold=0.5, num_perm=128)
                for i, m in enumerate(minhashes):
                    lsh.insert(i, m)

                # Get the initial results from LSH.
                query = minhashes[0]
                results = lsh.query(query)

                # Rank results using Jaccard similarity estimated by MinHash.
                results = [(query.jaccard(minhashes[key]), key) for key in results]
                results.sort(reverse=True)
                print(results)

            Output:

            .. code-block::

                [(1.0, 0), (0.421875, 4), (0.4140625, 19), (0.359375, 58), (0.3359375, 78), (0.265625, 62), (0.2578125, 11), (0.25, 98), (0.171875, 21)]

            Note that although the threshold is set to 0.5, the results are not
            guaranteed to be above 0.5 because the LSH index is approximate and
            the Jaccard similarity is estimated by MinHash.

        """
        if len(minhash) != self.h:
            raise ValueError(
                "Expecting minhash with length %d, got %d" % (self.h, len(minhash))
            )
        candidates = set()
        for (start, end), hashtable in zip(self.hashranges, self.hashtables):
            H = self._H(minhash.hashvalues[start:end])
            for key in hashtable.get(H):
                candidates.add(key)
        if self.prepickle:
            return [pickle.loads(key) for key in candidates]
        else:
            return list(candidates)

    def add_to_query_buffer(self, minhash: Union[MinHash, WeightedMinHash]) -> None:
        """
        Giving the MinHash of the query set, buffer
        queries to retrieve the keys that references
        sets with Jaccard similarities greater than
        the threshold.

        Buffered queries can be executed using
        :meth:`collect_query_buffer`. The combination of these
        functions is way faster if cassandra backend
        is used with `shared_buffer`.

        Args:
            minhash (MinHash): The MinHash of the query set.
        """
        if len(minhash) != self.h:
            raise ValueError(
                "Expecting minhash with length %d, got %d" % (self.h, len(minhash))
            )
        for (start, end), hashtable in zip(self.hashranges, self.hashtables):
            H = self._H(minhash.hashvalues[start:end])
            hashtable.add_to_select_buffer([H])

    def collect_query_buffer(self) -> List[Hashable]:
        """
        Execute and return buffered queries given
        by :meth:`add_to_query_buffer`.

        If multiple query MinHash were added to the query buffer,
        the intersection of the results of all query MinHash will be returned.

        Returns:
            list: a list of unique keys.
        """
        collected_result_sets = [
            set(collected_result_lists)
            for hashtable in self.hashtables
            for collected_result_lists in hashtable.collect_select_buffer()
        ]
        if not collected_result_sets:
            return []
        if self.prepickle:
            return [
                pickle.loads(key) for key in set.intersection(*collected_result_sets)
            ]
        return list(set.intersection(*collected_result_sets))

    def __contains__(self, key: Hashable) -> bool:
        """
        Args:
            key (Hashable): The unique identifier of a set.

        Returns:
            bool: True only if the key exists in the index.
        """
        if self.prepickle:
            key = pickle.dumps(key)
        return key in self.keys

    def remove(self, key: Hashable) -> None:
        """
        Remove the key from the index.

        Args:
            key (Hashable): The unique identifier of a set.

        Raises:
            ValueError: If the key does not exist.

        """
        if self.prepickle:
            key = pickle.dumps(key)
        if key not in self.keys:
            raise ValueError("The given key does not exist")
        for H, hashtable in zip(self.keys[key], self.hashtables):
            hashtable.remove_val(H, key)
            if not hashtable.get(H):
                hashtable.remove(H)
        self.keys.remove(key)

    def is_empty(self) -> bool:
        """
        Returns:
            bool: `True` only if the index is empty.
        """
        return any(t.size() == 0 for t in self.hashtables)

    def _byteswap(self, hs):
        return bytes(hs.byteswap().data)

    def _hashed_byteswap(self, hs):
        return self.hashfunc(bytes(hs.byteswap().data))

    def _query_b(self, minhash, b):
        if len(minhash) != self.h:
            raise ValueError(
                "Expecting minhash with length %d, got %d" % (self.h, len(minhash))
            )
        if b > len(self.hashtables):
            raise ValueError("b must be less or equal to the number of hash tables")
        candidates = set()
        for (start, end), hashtable in zip(self.hashranges[:b], self.hashtables[:b]):
            H = self._H(minhash.hashvalues[start:end])
            if H in hashtable:
                for key in hashtable[H]:
                    candidates.add(key)
        if self.prepickle:
            return {pickle.loads(key) for key in candidates}
        else:
            return candidates

    def get_counts(self) -> List[Dict[Hashable, int]]:
        """
        Returns a list of length :attr:`b` (i.e., number of hash tables) with
        each element a dictionary mapping hash table bucket key to the number
        of indexed keys stored under each bucket.

        Returns:
            list: a list of dictionaries.
        """
        return [hashtable.itemcounts() for hashtable in self.hashtables]

    def get_subset_counts(self, *keys: Hashable) -> List[Dict[Hashable, int]]:
        """
        Returns the bucket allocation counts (see :meth:`get_counts` above)
        restricted to the list of keys given.

        Args:
            keys (Hashable) : the keys for which to get the bucket allocation
                counts.

        Returns:
            list: a list of dictionaries.
        """
        if self.prepickle:
            key_set = [pickle.dumps(key) for key in set(keys)]
        else:
            key_set = list(set(keys))
        hashtables = [unordered_storage({"type": "dict"}) for _ in range(self.b)]
        Hss = self.keys.getmany(*key_set)
        for key, Hs in zip(key_set, Hss):
            for H, hashtable in zip(Hs, hashtables):
                hashtable.insert(H, key)
        return [hashtable.itemcounts() for hashtable in hashtables]


class MinHashLSHInsertionSession:
    """Context manager for batch insertion of documents into a MinHashLSH.

    Args:
        lsh (MinHashLSH): The MinHashLSH to insert into.
        buffer_size (int): The buffer size for insert_session mode.
    """

    def __init__(self, lsh: MinHashLSH, buffer_size: int):
        self.lsh = lsh
        self.lsh.buffer_size = buffer_size

    def __enter__(self) -> MinHashLSHInsertionSession:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def close(self) -> None:
        self.lsh.keys.empty_buffer()
        for hashtable in self.lsh.hashtables:
            hashtable.empty_buffer()

    def insert(
        self,
        key: Hashable,
        minhash: Union[MinHash, WeightedMinHash],
        check_duplication=True,
    ) -> None:
        """
        Insert a unique key to the index, together
        with a MinHash (or weighted MinHash) of the set referenced by
        the key.

        Args:
            key (Hashable): The unique identifier of the set.
            minhash (Union[MinHash, WeightedMinhash]): The MinHash of the set.
        """
        self.lsh._insert(key, minhash, check_duplication=check_duplication, buffer=True)
