'''
This module implements the Locality Sensitive Hashing index 
with MinHash (MinHash LSH). MinHash LSH index supports query
with Jaccard similarity threshold.

Reference: Chapter 3, Mining of Massive Datasets 
(http://www.mmds.org/)
'''

try:
    from .minhash import MinHash
except ImportError:
    # For Python 2
    from minhash import MinHash


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
    Compute the optimal LSH parameter that minimizes the weighted sum
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


class LSH(object):
    '''
    The classic MinHash LSH
    '''

    def __init__(self, threshold=0.9, num_perm=128, weights=(0.5,0.5)):
        '''
        Create an empty LSH index that accepts MinHash objects
        with `num_perm` permutation functions and query
        Jaccard similarity threshold `threshold`.
        The initialized LSH will be optimized for the threshold by
        minizing the false positive and false negative.

        Use `weights` to adjust the relative importance of 
        minizing false positive and false negative when optimizing 
        for the Jaccard similarity threshold.
        `weights` is a tuple in the format of 
        (false_negative_weight, false_negat.
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
        self.num_perm = num_perm
        false_positive_weight, false_negative_weight = weights
        self.b, self.r = _optimal_param(threshold, num_perm,
                false_positive_weight, false_negative_weight)
        self.hashtables = [dict() for _ in range(self.b)]
        self.hashranges = [(i*self.r, (i+1)*self.r) for i in range(self.b)]

    def is_empty(self):
        return any(len(t) == 0 for t in self.hashtables)

    def _H(self, hs):
        return "".join("%.8x" % h for h in hs)

    def insert(self, key, minhash):
        '''
        Insert a `key` to the index, together
        with a `minhash` of the data referenced by the `key`.
        '''
        if not isinstance(minhash, MinHash):
            raise ValueError("minhash must be of MinHash class")
        if len(minhash.hashvalues) != self.num_perm:
            raise ValueError("Expecting minhash with %d permutation functions, got %d"
                    % (self.num_perm, len(minhash.hashvalues)))
        for (start, end), hashtable in zip(self.hashranges, self.hashtables):
            H = self._H(minhash.hashvalues[start:end])
            if H not in hashtable:
                hashtable[H] = []
            hashtable[H].append(key)

    def query(self, minhash):
        '''
        Giving the MinHash of the query dataset, retrieve 
        the keys that references datasets with Jaccard
        similarities greater than the threshold set by the index.
        '''
        if not isinstance(minhash, MinHash):
            raise ValueError("minhash must be of MinHash class")
        if len(minhash.hashvalues) != self.num_perm:
            raise ValueError("Expecting minhash with %d permutation functions, got %d"
                    % (self.num_perm, len(minhash.hashvalues)))
        candidates = set()
        for (start, end), hashtable in zip(self.hashranges, self.hashtables):
            H = self._H(minhash.hashvalues[start:end])
            if H in hashtable:
                for key in hashtable[H]:
                    candidates.add(key)
        return list(candidates)

