'''
This Weighted MinHash is based on Sergey Ioffe's paper,
Improved Consistent Sampling, Weighted Minhash and L1 Sketching,
http://static.googleusercontent.com/media/research.google.com/en//pubs/archive/36928.pdf
'''
import collections
import copy
import numpy as np


class WeightedMinHash(object):

    def __init__(self, seed, hashvalues):
        '''
        Create a WeightedMinHash object given the seed
        that was used to generate it and the hash values.
        '''
        self.seed = seed
        self.hashvalues = hashvalues
    
    def __len__(self):
        '''
        Return the number of hash values as the size
        '''
        return len(self.hashvalues)

    def __eq__(self, other):
        '''
        Check equivalence between WeightedMinHash
        '''
        return self.seed == other.seed and \
                np.array_equal(self.hashvalues, other.hashvalues)
    
    def copy(self):
        '''
        Create a copy of this WeightedMinHash by exporting its state.
        '''
        return WeightedMinHash(self.seed, self.digest())
    
    def digest(self):
        '''
        Returns the hash values.
        '''
        return copy.copy(self.hashvalues)

    def jaccard(self, other):
        '''
        Estimate Jaccard similarity (resemblance) using this WeightedMinHash
        and the other.
        '''
        if other.seed != self.seed:
            raise ValueError("Cannot compute Jaccard given WeightedMinHash objects with\
                    different seeds")
        if len(self) != len(other):
            raise ValueError("Cannot compute Jaccard given WeightedMinHash objects with\
                    different numbers of hash values")
        # Check how many pairs of (k, t) hashvalues are equal
        intersection = 0
        for this, that in zip(self.hashvalues, other.hashvalues):
            if np.array_equal(this, that):
                intersection += 1
        return float(intersection) / float(len(self))
 

class WeightedMinHashGenerator(object):

    def __init__(self, dim, sample_size=128, seed=1):
        '''
        Initialize the generator with the number of dimensions of input 
        vectors, number of samples, and the seed for creating random parameters.
        '''
        self.dim = dim
        self.sample_size = sample_size
        self.seed = seed
        generator = np.random.RandomState(seed=seed)
        self.rs = generator.gamma(2, 1, (sample_size, dim)).astype(np.float32)
        self.ln_cs = np.log(generator.gamma(2, 1, (sample_size, dim))).astype(np.float32)
        self.betas = generator.uniform(0, 1, (sample_size, dim)).astype(np.float32)

    def minhash(self, v):
        '''
        Takes a vector of weights as input and returns a
        WeightedMinHash object.
        '''
        if not isinstance(v, collections.Iterable):
            raise TypeError("Input vector must be an iterable")
        if not len(v) == self.dim:
            raise ValueError("Input dimension mismatch, expecting %d" % self.dim)
        if not isinstance(v, np.ndarray):
            v = np.array(v, dtype=np.float32)
        elif v.dtype != np.float32:
            v = v.astype(np.float32)
        hashvalues = np.zeros((self.sample_size, 2), dtype=np.int)
        vzeros = (v == 0)
        if vzeros.all():
            raise ValueError("Input is all zeros")
        v[vzeros] = np.nan
        vlog = np.log(v)
        for i in range(self.sample_size):
            t = np.floor((vlog / self.rs[i]) + self.betas[i])
            ln_y = (t - self.betas[i]) * self.rs[i]
            ln_a = self.ln_cs[i] - ln_y - self.rs[i]
            k = np.nanargmin(ln_a)
            hashvalues[i][0], hashvalues[i][1] = k, int(t[k])
        return WeightedMinHash(self.seed, hashvalues)

