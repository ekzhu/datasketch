'''
This module implements the LSH Forests for MinHash data
sketch. It supports top-k query.

Instead of using prefix trees as the original paper
(http://ilpubs.stanford.edu:8090/678/1/2005-14.pdf), 
I use a sorted array to store the hash values in every
hash table.
'''

from collections import deque, defaultdict
from datasketch.minhash import hashvalue_byte_size


class MinHashLSHForest(object):
    '''
    The LSH Forests for MinHash data sketch.
    Each hash table is a sorted array of the hash values.
    '''

    def __init__(self, r=4, b=32):
        '''
        Creates an empty `MinHashLSHForest` object that accepts
        two parameters `r` and `b`.
        `b` is the number of bands used in LSH and `r` is the 
        number of hash values concatenated to form a band.
        The product ot `r` and `b` must equal to the `num_perm`
        parameter of the `MinHash` objects used.
        '''
        if r <= 0 or b <= 0:
            raise ValueError("r and b must be positive")
        self.r = r
        self.b = b
        self.hashtables = [defaultdict(list) for _ in range(self.b)]
        self.hashranges = [(i*self.r, (i+1)*self.r) for i in range(self.b)]
        self.keys = dict()
        # This is the all the sorted concatenated hash values (bands) for
        # all bands.
        self.sorted_hashtables = [[] for _ in range(self.b)]

    def is_empty(self):
        '''
        Check whether there is any searchable keys in the index.
        Note that keys won't be searchable until `index` is called.
        '''
        return any(len(t) == 0 for t in self.sorted_hashtables)

    def _H(self, hs):
        return bytes(hs.byteswap().data)

    def __contains__(self, key):
        '''
        Return True only if the key has been added to the index.
        '''
        return key in self.keys

    def add(self, key, minhash):
        '''
        Add a unique `key`, together
        with a `minhash` of the data referenced by the `key`.
        The key won't be searchbale until `index` method is called.
        '''
        if len(minhash) != self.r*self.b:
            raise ValueError("The num_perm of MinHash must equal to the product of r and b")
        if key in self.keys:
            raise ValueError("The given key has already been added")
        self.keys[key] = [self._H(minhash.hashvalues[start:end]) 
                for start, end in self.hashranges]
        for H, hashtable in zip(self.keys[key], self.hashtables):
            hashtable[H].append(key)

    def index(self):
        '''
        Index all the keys added so far and make them searchable.
        '''
        for i, hashtable in enumerate(self.hashtables):
            self.sorted_hashtables[i] = [H for H in hashtable.keys()]
            self.sorted_hashtables[i].sort()

    def _query(self, minhash, r, b):
        if r > self.r or r <=0 or b > self.b or b <= 0:
            raise ValueError("parameter outside range")
        # Generate prefixes of concatenated hash values
        hps = [self._H(minhash.hashvalues[start:start+r]) 
                for start, _ in self.hashranges]
        # Caculate the string length of each original hash value
        prefix_size = hashvalue_byte_size * r
        for ht, hp, hashtable in zip(self.sorted_hashtables, hps, self.hashtables):
            k = self._binary_search(len(ht), lambda x : ht[x][:prefix_size] >= hp)
            if k < len(ht) and ht[k][:prefix_size] == hp:
                j = k
                while j < len(ht) and ht[j][:prefix_size] == hp:
                    for key in hashtable[ht[j]]:
                        yield key
                    j += 1

    def query(self, minhash, k):
        '''
        Return the approximate top-`k` keys that have the highest 
        Jaccard similarities to the query `minhash`.
        This will return at most `k` keys as a list.
        '''
        if k <= 0:
            raise ValueError("k must be positive")
        if len(minhash) != self.r*self.b:
            raise ValueError("The num_perm of MinHash must equal to the product of r and b")
        results = set()
        r = self.r
        while r > 0: 
            for key in self._query(minhash, r, self.b):
                results.add(key)
                if len(results) >= k:
                    break
            r -= 1
        return list(results)

    def _binary_search(self, n, func):
        '''
        https://golang.org/src/sort/search.go?s=2247:2287#L49
        '''
        i, j = 0, n
        while i < j:
            h = i + (j - i) / 2
            if not func(h):
                i = h + 1
            else:
                j = h
        return i
