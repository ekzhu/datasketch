'''
This module implements the Locality Sensitive Hashing index 
with MinHash (MinHash LSH) and Weighted MinHash (Weighted MinHash LSH)
Both indexes supports query with Jaccard similarity threshold.

Reference: Chapter 3, Mining of Massive Datasets 
(http://www.mmds.org/)
'''
from collections import defaultdict


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
    The classic MinHash LSH
    '''

    def __init__(self, threshold=0.9, num_perm=128, weights=(0.5,0.5)):
        '''
        Create an empty `MinHashLSH` index that accepts MinHash objects
        with `num_perm` permutation functions and query
        Jaccard similarity threshold `threshold`.
        The initialized `MinHashLSH` will be optimized for the threshold by
        minizing the false positive and false negative.

        Use `weights` to adjust the relative importance of 
        minizing false positive and false negative when optimizing 
        for the Jaccard similarity threshold.
        `weights` is a tuple in the format of 
        (false_positive_weight, false_negative_weight).
        '''
        if threshold > 1.0 or threshold < 0.0:
            raise ValueError("threshold must be in [0.0, 1.0]") 
        if num_perm < 2:
            raise ValueError("Too few permutation functions")
        if any(w < 0.0 or w > 1.0 for w in weights):
            raise ValueError("Weight must be in [0.0, 1.0]")
        if sum(weights) != 1.0:
            raise ValueError("Weights must sum to 1.0")
        self.threshold = threshold
        self.h = num_perm
        false_positive_weight, false_negative_weight = weights
        self.b, self.r = _optimal_param(threshold, num_perm,
                false_positive_weight, false_negative_weight)
        self.hashtables = [defaultdict(list) for _ in range(self.b)]
        self.hashranges = [(i*self.r, (i+1)*self.r) for i in range(self.b)]
        self.keys = dict()

    def is_empty(self):
        return any(len(t) == 0 for t in self.hashtables)

    def _H(self, hs):
        return bytes(hs.byteswap().data)

    def __contains__(self, key):
        '''
        Return True only if the key exists in the index.
        '''
        return key in self.keys

    def insert(self, key, minhash):
        '''
        Insert a unique `key` to the index, together
        with a `minhash` of the data referenced by the `key`.
        '''
        if len(minhash) != self.h:
            raise ValueError("Expecting minhash with length %d, got %d"
                    % (self.h, len(minhash)))
        if key in self.keys:
            raise ValueError("The given key already exists")
        self.keys[key] = [self._H(minhash.hashvalues[start:end]) 
                for start, end in self.hashranges]
        for H, hashtable in zip(self.keys[key], self.hashtables):
            hashtable[H].append(key)

    def query(self, minhash):
        '''
        Giving the MinHash of the query dataset, retrieve 
        the keys that references datasets with Jaccard
        similarities greater than the threshold set by the index.
        '''
        if len(minhash) != self.h:
            raise ValueError("Expecting minhash with length %d, got %d"
                    % (self.h, len(minhash)))
        candidates = set()
        for (start, end), hashtable in zip(self.hashranges, self.hashtables):
            H = self._H(minhash.hashvalues[start:end])
            if H in hashtable:
                for key in hashtable[H]:
                    candidates.add(key)
        return list(candidates)

    def remove(self, key):
        '''
        Remove the key from the index.
        '''
        if key not in self.keys:
            raise ValueError("The given key does not exist")
        for H, hashtable in zip(self.keys[key], self.hashtables):
            hashtable[H].remove(key)
            if not hashtable[H]:
                del hashtable[H]
        self.keys.pop(key)


class WeightedMinHashLSH(MinHashLSH):
    '''
    The classic MinHash LSH adapted for Weighted MinHash
    '''

    def __init__(self, threshold=0.9, sample_size=128, weights=(0.5,0.5)):
        '''
        Create an empty `WeightedMinHashLSH` index that accepts 
        WeightedMinHash objects
        with `sample_size` samples and query
        Jaccard similarity threshold `threshold`.
        The initialized `WeightedMinHashLSH` will be optimized for the threshold by
        minizing the false positive and false negative.

        Use `weights` to adjust the relative importance of 
        minizing false positive and false negative when optimizing 
        for the Jaccard similarity threshold.
        `weights` is a tuple in the format of 
        (false_positive_weight, false_negative_weight).
        '''
        super(WeightedMinHashLSH, self).__init__(threshold, sample_size, weights)
