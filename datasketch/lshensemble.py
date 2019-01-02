from collections import deque, Counter
import numpy as np
from datasketch.lsh import integrate, MinHashLSH
from datasketch.lshensemble_partition import optimal_partitions


def _false_positive_probability(threshold, b, r, xq):
    '''
    Compute the false positive probability given the containment threshold.
    xq is the ratio of x/q.
    '''
    _probability = lambda t : 1 - (1 - (t/(1 + xq - t))**float(r))**float(b)
    if xq >= threshold:
        a, err = integrate(_probability, 0.0, threshold)
        return a
    a, err = integrate(_probability, 0.0, xq)
    return a


def _false_negative_probability(threshold, b, r, xq):
    '''
    Compute the false negative probability given the containment threshold
    '''
    _probability = lambda t : 1 - (1 - (1 - (t/(1 + xq - t))**float(r))**float(b))
    if xq >= 1.0:
        a, err = integrate(_probability, threshold, 1.0)
        return a
    if xq >= threshold:
        a, err = integrate(_probability, threshold, xq)
        return a
    return 0.0


def _optimal_param(threshold, num_perm, max_r, xq, false_positive_weight,
        false_negative_weight):
    '''
    Compute the optimal parameters that minimizes the weighted sum
    of probabilities of false positive and false negative.
    xq is the ratio of x/q.
    '''
    min_error = float("inf")
    opt = (0, 0)
    for b in range(1, num_perm+1):
        for r in range(1, max_r+1):
            if b*r > num_perm:
                continue
            fp = _false_positive_probability(threshold, b, r, xq)
            fn = _false_negative_probability(threshold, b, r, xq)
            error = fp*false_positive_weight + fn*false_negative_weight
            if error < min_error:
                min_error = error
                opt = (b, r)
    return opt


class MinHashLSHEnsemble(object):
    '''
    The :ref:`minhash_lsh_ensemble` index. It supports
    :ref:`containment` queries.
    The implementation is based on
    `E. Zhu et al. <http://www.vldb.org/pvldb/vol9/p1185-zhu.pdf>`_.

    Args:
        threshold (float): The Containment threshold between 0.0 and
            1.0. The initialized LSH Ensemble will be optimized for the threshold by
            minizing the false positive and false negative.
        num_perm (int, optional): The number of permutation functions used
            by the MinHash to be indexed. For weighted MinHash, this
            is the sample size (`sample_size`).
        num_part (int, optional): The number of partitions in LSH Ensemble.
        m (int, optional): The memory usage factor: an LSH Ensemble uses approximately
            `m` times more memory space than a MinHash LSH with the same number of
            sets indexed. The higher the `m` the better the accuracy.
        weights (tuple, optional): Used to adjust the relative importance of
            minizing false positive and false negative when optimizing
            for the Containment threshold. Similar to the `weights` parameter
            in :class:`datasketch.MinHashLSH`.

    Note:
        Using more partitions (`num_part`) leads to better accuracy, at the
        expense of slower query performance.
        This is different from `the paper`_ and the `Go implementation`_, in which
        more partitions leads to better accuracy AND faster query performance,
        due to parallelism.

    Note:
        More information about the parameter `m` can be found in the
        `Go implementation`_
        of LSH Ensemble, in which `m` is named `MaxK`.

    .. _`Go implementation`: https://github.com/ekzhu/lshensemble
    .. _`the paper`: http://www.vldb.org/pvldb/vol9/p1185-zhu.pdf
    '''

    def __init__(self, threshold=0.9, num_perm=128, num_part=16, m=8, weights=(0.5,0.5)):
        if threshold > 1.0 or threshold < 0.0:
            raise ValueError("threshold must be in [0.0, 1.0]")
        if num_perm < 2:
            raise ValueError("Too few permutation functions")
        if num_part < 1:
            raise ValueError("num_part must be at least 1")
        if m < 2 or m > num_perm:
            raise ValueError("m must be in the range of [2, num_perm]")
        if any(w < 0.0 or w > 1.0 for w in weights):
            raise ValueError("Weight must be in [0.0, 1.0]")
        if sum(weights) != 1.0:
            raise ValueError("Weights must sum to 1.0")
        self.threshold = threshold
        self.h = num_perm
        self.m = m
        rs = self._init_optimal_params(weights)
        # Initialize multiple LSH indexes for each partition
        self.indexes = [dict((r, MinHashLSH(num_perm=self.h,
            params=(int(self.h/r), r))) for r in rs)
            for _ in range(0, num_part)]
        self.lowers = [None for _ in self.indexes]
        self.uppers = [None for _ in self.indexes]

    def _init_optimal_params(self, weights):
        false_positive_weight, false_negative_weight = weights
        self.xqs = np.exp(np.linspace(-5, 5, 10))
        self.params = np.array([_optimal_param(self.threshold, self.h, self.m,
                                               xq,
                                               false_positive_weight,
                                               false_negative_weight)
                                for xq in self.xqs], dtype=np.int)
        # Find all unique r
        rs = set()
        for _, r in self.params:
            rs.add(r)
        return rs

    def _get_optimal_param(self, x, q):
        i = np.searchsorted(self.xqs, float(x)/float(q), side='left')
        if i == len(self.params):
            i = i - 1
        return self.params[i]

    def index(self, entries):
        '''
        Index all sets given their keys, MinHashes, and sizes.
        It can be called only once after the index is created.

        Args:
            entries (`iterable` of `tuple`): An iterable of tuples, each must be
                in the form of `(key, minhash, size)`, where `key` is the unique
                identifier of a set, `minhash` is the MinHash of the set,
                and `size` is the size or number of unique items in the set.

        Note:
            `size` must be positive.
        '''
        if not self.is_empty():
            raise ValueError("Cannot call index again on a non-empty index")
        if not isinstance(entries, list):
            queue = deque([])
            for key, minhash, size in entries:
                if size <= 0:
                    raise ValueError("Set size must be positive")
                queue.append((key, minhash, size))
            entries = list(queue)
        if len(entries) == 0:
            raise ValueError("entries is empty")
        # Create optimal partitions.
        sizes, counts = np.array(sorted(
            Counter(e[2] for e in entries).most_common())).T
        partitions = optimal_partitions(sizes, counts, len(self.indexes))
        for i, (lower, upper) in enumerate(partitions):
            self.lowers[i], self.uppers[i] = lower, upper
        # Insert into partitions.
        entries.sort(key=lambda e : e[2])
        curr_part = 0
        for key, minhash, size in entries:
            if size > self.uppers[curr_part]:
                curr_part += 1
            for r in self.indexes[curr_part]:
                self.indexes[curr_part][r].insert(key, minhash)

    def query(self, minhash, size):
        '''
        Giving the MinHash and size of the query set, retrieve
        keys that references sets with containment with respect to
        the query set greater than the threshold.

        Args:
            minhash (datasketch.MinHash): The MinHash of the query set.
            size (int): The size (number of unique items) of the query set.

        Returns:
            `iterator` of keys.
        '''
        for i, index in enumerate(self.indexes):
            u = self.uppers[i]
            if u is None:
                continue
            b, r = self._get_optimal_param(u, size)
            for key in index[r]._query_b(minhash, b):
                yield key

    def __contains__(self, key):
        '''
        Args:
            key (hashable): The unique identifier of a set.

        Returns:
            bool: True only if the key exists in the index.
        '''
        return any(any(key in index[r] for r in index)
                   for index in self.indexes)

    def is_empty(self):
        '''
        Returns:
            bool: Check if the index is empty.
        '''
        return all(all(index[r].is_empty() for r in index)
                   for index in self.indexes)


if __name__ == "__main__":
    import numpy as np
    xqs = np.exp(np.linspace(-5, 5, 10))
    threshold = 0.5
    max_r = 8
    num_perm = 256
    false_negative_weight, false_positive_weight = 0.5, 0.5
    for xq in xqs:
        b, r = _optimal_param(threshold, num_perm, max_r, xq,
                false_positive_weight, false_negative_weight)
        print("threshold: %.2f, xq: %.3f, b: %d, r: %d" % (threshold, xq, b, r))
