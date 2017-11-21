from collections import defaultdict


class MinHashLSHForest(object):
    '''
    The LSH Forest for MinHash. It supports top-k query in Jaccard
    similarity.
    Instead of using prefix trees as the `original paper
    <http://ilpubs.stanford.edu:8090/678/1/2005-14.pdf>`_,
    I use a sorted array to store the hash values in every
    hash table.

    Args:
        num_perm (int, optional): The number of permutation functions used
            by the MinHash to be indexed. For weighted MinHash, this
            is the sample size (`sample_size`).
        l (int, optional): The number of prefix trees as described in the
            paper.

    Note:
        The MinHash LSH Forest also works with weighted Jaccard similarity
        and weighted MinHash without modification.
    '''

    def __init__(self, num_perm=128, l=8):
        if l <= 0 or num_perm <= 0:
            raise ValueError("num_perm and l must be positive")
        if l > num_perm:
            raise ValueError("l cannot be greater than num_perm")
        # Number of prefix trees
        self.l = l
        # Maximum depth of the prefix tree
        self.k = int(num_perm / l)
        self.hashtables = [defaultdict(list) for _ in range(self.l)]
        self.hashranges = [(i*self.k, (i+1)*self.k) for i in range(self.l)]
        self.keys = dict()
        # This is the sorted array implementation for the prefix trees
        self.sorted_hashtables = [[] for _ in range(self.l)]

    def add(self, key, minhash):
        '''
        Add a unique key, together
        with a MinHash (or weighted MinHash) of the set referenced by the key.

        Note:
            The key won't be searchbale until the
            :func:`datasketch.MinHashLSHForest.index` method is called.

        Args:
            key (hashable): The unique identifier of the set.
            minhash (datasketch.MinHash): The MinHash of the set.
        '''
        if len(minhash) < self.k*self.l:
            raise ValueError("The num_perm of MinHash out of range")
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
        if r > self.k or r <=0 or b > self.l or b <= 0:
            raise ValueError("parameter outside range")
        # Generate prefixes of concatenated hash values
        hps = [self._H(minhash.hashvalues[start:start+r])
                for start, _ in self.hashranges]
        # Set the prefix length for look-ups in the sorted hash values list
        prefix_size = len(hps[0])
        for ht, hp, hashtable in zip(self.sorted_hashtables, hps, self.hashtables):
            i = self._binary_search(len(ht), lambda x : ht[x][:prefix_size] >= hp)
            if i < len(ht) and ht[i][:prefix_size] == hp:
                j = i
                while j < len(ht) and ht[j][:prefix_size] == hp:
                    for key in hashtable[ht[j]]:
                        yield key
                    j += 1

    def query(self, minhash, k):
        '''
        Return the approximate top-k keys that have the highest
        Jaccard similarities to the query set.

        Args:
            minhash (datasketch.MinHash): The MinHash of the query set.
            k (int): The maximum number of keys to return.

        Returns:
            `list` of at most k keys.
        '''
        if k <= 0:
            raise ValueError("k must be positive")
        if len(minhash) < self.k*self.l:
            raise ValueError("The num_perm of MinHash out of range")
        results = set()
        r = self.k
        while r > 0:
            for key in self._query(minhash, r, self.l):
                results.add(key)
                if len(results) >= k:
                    return list(results)
            r -= 1
        return list(results)

    def _binary_search(self, n, func):
        '''
        https://golang.org/src/sort/search.go?s=2247:2287#L49
        '''
        i, j = 0, n
        while i < j:
            h = int(i + (j - i) / 2)
            if not func(h):
                i = h + 1
            else:
                j = h
        return i

    def is_empty(self):
        '''
        Check whether there is any searchable keys in the index.
        Note that keys won't be searchable until `index` is called.

        Returns:
            bool: True if there is no searchable key in the index.
        '''
        return any(len(t) == 0 for t in self.sorted_hashtables)

    def _H(self, hs):
        return bytes(hs.byteswap().data)

    def __contains__(self, key):
        '''
        Returns:
            bool: True only if the key has been added to the index.
        '''
        return key in self.keys
