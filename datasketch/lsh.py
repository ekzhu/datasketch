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
    The Locality Sensitive Hashing index 
    with MinHash (MinHash LSH). 
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
            minizing false positive and false negative when optimizing 
            for the Jaccard similarity threshold.
            `weights` is a tuple in the format of 
            :code:`(false_positive_weight, false_negative_weight)`.
    
    Note:
        The MinHash LSH index also works with weighted Jaccard similarity
        and weighted MinHash without modification.
    '''

    def __init__(self, threshold=0.9, num_perm=128, weights=(0.5,0.5)):
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

    def insert(self, key, minhash):
        '''
        Insert a unique key to the index, together
        with a MinHash (or weighted MinHash) of the set referenced by 
        the key.

        Args:
            key (hashable): The unique identifier of the set. 
            minhash (datasketch.MinHash): The MinHash of the set. 
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
        Giving the MinHash of the query set, retrieve 
        the keys that references sets with Jaccard
        similarities greater than the threshold.
        
        Args:
            minhash (datasketch.MinHash): The MinHash of the query set. 

        Returns:
            `list` of keys.
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

    def __contains__(self, key):
        '''
        Args:
            key (hashable): The unique identifier of a set.

        Returns: 
            bool: True only if the key exists in the index.
        '''
        return key in self.keys

    def remove(self, key):
        '''
        Remove the key from the index.

        Args:
            key (hashable): The unique identifier of a set.
        '''
        if key not in self.keys:
            raise ValueError("The given key does not exist")
        for H, hashtable in zip(self.keys[key], self.hashtables):
            hashtable[H].remove(key)
            if not hashtable[H]:
                del hashtable[H]
        self.keys.pop(key)

    def is_empty(self):
        '''
        Returns:
            bool: Check if the index is empty.
        '''
        return any(len(t) == 0 for t in self.hashtables)

    def _H(self, hs):
        return bytes(hs.byteswap().data)

