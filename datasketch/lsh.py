import pickle
from datasketch.storage import (
    ordered_storage, unordered_storage, _random_name)

_integration_precision = 0.001
def _integration(f, a, b):
    p = _integration_precision
    area = 0.0
    x = a
    while x < b:
        area += f(x+0.5*p)*p
        x += p
    return area, None

try:
    from scipy.integrate import quad as integrate
except ImportError:
    # For when no scipy installed
    integrate = _integration


def _false_positive_probability(threshold, b, r):
    _probability = lambda s : 1 - (1 - s**float(r))**float(b)
    a, err = integrate(_probability, 0.0, threshold) 
    return a


def _false_negative_probability(threshold, b, r):
    _probability = lambda s : 1 - (1 - (1 - s**float(r))**float(b))
    a, err = integrate(_probability, threshold, 1.0)
    return a


def _optimal_param(threshold, num_perm, false_positive_weight,
        false_negative_weight):
    '''
    Compute the optimal `MinHashLSH` parameter that minimizes the weighted sum
    of probabilities of false positive and false negative.
    '''
    min_error = float("inf")
    opt = (0, 0)
    for b in range(1, num_perm+1):
        max_r = int(num_perm / b)
        for r in range(1, max_r+1):
            fp = _false_positive_probability(threshold, b, r)
            fn = _false_negative_probability(threshold, b, r)
            error = fp*false_positive_weight + fn*false_negative_weight
            if error < min_error:
                min_error = error
                opt = (b, r)
    return opt


class MinHashLSH(object):
    '''
    The :ref:`minhash_lsh` index. 
    It supports query with `Jaccard similarity`_ threshold.
    Reference: `Chapter 3, Mining of Massive Datasets 
    <http://www.mmds.org/>`_.

    Args:
        threshold (float): The Jaccard similarity threshold between 0.0 and
            1.0. The initialized MinHash LSH will be optimized for the threshold by
            minizing the false positive and false negative.
        num_perm (int, optional): The number of permutation functions used
            by the MinHash to be indexed. For weighted MinHash, this
            is the sample size (`sample_size`).
        weights (tuple, optional): Used to adjust the relative importance of 
            minimizing false positive and false negative when optimizing
            for the Jaccard similarity threshold.
            `weights` is a tuple in the format of 
            :code:`(false_positive_weight, false_negative_weight)`.
        params (tuple, optional): The LSH parameters (i.e., number of bands and size
            of each bands). This is used to bypass the parameter optimization
            step in the constructor. `threshold` and `weights` will be ignored 
            if this is given.
        storage_config (dict, optional): Type of storage service to use for storing
            hashtables and keys.
            `basename` is an optional property whose value will be used as the prefix to 
            stored keys. If this is not set, a random string will be generated instead. If you 
            set this, you will be responsible for ensuring there are no key collisions.
        prepickle (bool, optional): If True, all keys are pickled to bytes before
            insertion. If None, a default value is chosen based on the
            `storage_config`.

    Note: 
        `weights` must sum to 1.0, and the format is 
        (false positive weight, false negative weight).
        For example, if minimizing false negative (or maintaining high recall) is more
        important, assign more weight toward false negative: weights=(0.4, 0.6).
        Try to live with a small difference between weights (i.e. < 0.5).
    '''

    def __init__(self, threshold=0.9, num_perm=128, weights=(0.5, 0.5),
                 params=None, storage_config=None, prepickle=None):
        storage_config = {'type': 'dict'} if not storage_config else storage_config
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
                raise ValueError("The product of b and r must be less than num_perm")
        else:
            false_positive_weight, false_negative_weight = weights
            self.b, self.r = _optimal_param(threshold, num_perm,
                    false_positive_weight, false_negative_weight)

        self.prepickle = storage_config['type'] == 'redis' if prepickle is None else prepickle

        basename = storage_config.get('basename', _random_name(11))
        self.hashtables = [
            unordered_storage(storage_config, name=b''.join([basename, b'_bucket_', bytes([i])]))
            for i in range(self.b)]
        self.hashranges = [(i*self.r, (i+1)*self.r) for i in range(self.b)]
        self.keys = ordered_storage(storage_config, name=b''.join([basename, b'_keys']))

    @property
    def buffer_size(self):
        return self._buffer_size

    @buffer_size.setter
    def buffer_size(self, value):
        self.keys.buffer_size = value
        for t in self.hashtables:
            t.buffer_size = value
        self._buffer_size = value

    def insert(self, key, minhash, check_duplication=True):
        '''
        Insert a key to the index, together
        with a MinHash (or weighted MinHash) of the set referenced by 
        the key.

        :param str key: The identifier of the set.
        :param datasketch.MinHash minhash: The MinHash of the set.
        :param bool check_duplication: To avoid duplicate keys in the storage (`default=True`).
                                       It's recommended to not change the default, but
                                       if you want to avoid the overhead during insert
                                       you can set `check_duplication = False`.
        '''
        self._insert(key, minhash, check_duplication=check_duplication, buffer=False)

    def insertion_session(self, buffer_size=50000):
        '''
        Create a context manager for fast insertion into this index.

        :param int buffer_size: The buffer size for insert_session mode (default=50000).

        Returns:
            datasketch.lsh.MinHashLSHInsertionSession
        '''
        return MinHashLSHInsertionSession(self, buffer_size=buffer_size)

    def _insert(self, key, minhash, check_duplication=True, buffer=False):
        if len(minhash) != self.h:
            raise ValueError("Expecting minhash with length %d, got %d"
                    % (self.h, len(minhash)))
        if check_duplication and key in self.keys:
            raise ValueError("The given key already exists")
        Hs = [self._H(minhash.hashvalues[start:end])
              for start, end in self.hashranges]
        if self.prepickle:
            key = pickle.dumps(key)
        self.keys.insert(key, *Hs, buffer=buffer)
        for H, hashtable in zip(Hs, self.hashtables):
            hashtable.insert(H, key, buffer=buffer)

    def query(self, minhash):
        '''
        Giving the MinHash of the query set, retrieve 
        the keys that references sets with Jaccard
        similarities greater than the threshold.
        
        Args:
            minhash (datasketch.MinHash): The MinHash of the query set. 

        Returns:
            `list` of unique keys.
        '''
        if len(minhash) != self.h:
            raise ValueError("Expecting minhash with length %d, got %d"
                    % (self.h, len(minhash)))
        candidates = set()
        for (start, end), hashtable in zip(self.hashranges, self.hashtables):
            H = self._H(minhash.hashvalues[start:end])
            for key in hashtable.get(H):
                candidates.add(key)
        if self.prepickle:
            return [pickle.loads(key) for key in candidates]
        else:
            return list(candidates)

    def __contains__(self, key):
        '''
        Args:
            key (hashable): The unique identifier of a set.

        Returns: 
            bool: True only if the key exists in the index.
        '''
        if self.prepickle:
            key = pickle.dumps(key)
        return key in self.keys

    def remove(self, key):
        '''
        Remove the key from the index.

        Args:
            key (hashable): The unique identifier of a set.

        '''
        if self.prepickle:
            key = pickle.dumps(key)
        if key not in self.keys:
            raise ValueError("The given key does not exist")
        for H, hashtable in zip(self.keys[key], self.hashtables):
            hashtable.remove_val(H, key)
            if not hashtable.get(H):
                hashtable.remove(H)
        self.keys.remove(key)

    def is_empty(self):
        '''
        Returns:
            bool: Check if the index is empty.
        '''
        return any(t.size() == 0 for t in self.hashtables)

    @staticmethod
    def _H(hs):
        return bytes(hs.byteswap().data)

    def _query_b(self, minhash, b):
        if len(minhash) != self.h:
            raise ValueError("Expecting minhash with length %d, got %d"
                    % (self.h, len(minhash)))
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

    def get_counts(self):
        '''
        Returns a list of length ``self.b`` with elements representing the
        number of keys stored under each bucket for the given permutation.
        '''
        counts = [
            hashtable.itemcounts() for hashtable in self.hashtables]
        return counts

    def get_subset_counts(self, *keys):
        '''
        Returns the bucket allocation counts (see :func:`~datasketch.MinHashLSH.get_counts` above)
        restricted to the list of keys given.

        Args:
            keys (hashable) : the keys for which to get the bucket allocation
                counts
        '''
        if self.prepickle:
            key_set = [pickle.dumps(key) for key in set(keys)]
        else:
            key_set = list(set(keys))
        hashtables = [unordered_storage({'type': 'dict'}) for _ in
                      range(self.b)]
        Hss = self.keys.getmany(*key_set)
        for key, Hs in zip(key_set, Hss):
            for H, hashtable in zip(Hs, hashtables):
                hashtable.insert(H, key)
        return [hashtable.itemcounts() for hashtable in hashtables]


class MinHashLSHInsertionSession:
    '''Context manager for batch insertion of documents into a MinHashLSH.
    '''

    def __init__(self, lsh, buffer_size):
        self.lsh = lsh
        self.lsh.buffer_size = buffer_size

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self.lsh.keys.empty_buffer()
        for hashtable in self.lsh.hashtables:
            hashtable.empty_buffer()

    def insert(self, key, minhash, check_duplication=True):
        '''
        Insert a unique key to the index, together
        with a MinHash (or weighted MinHash) of the set referenced by
        the key.

        Args:
            key (hashable): The unique identifier of the set.
            minhash (datasketch.MinHash): The MinHash of the set.
        '''
        self.lsh._insert(key, minhash, check_duplication=check_duplication,
                         buffer=True)
