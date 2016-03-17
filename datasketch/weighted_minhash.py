'''
This Weighted MinHash is based on Sergey Ioffe's paper,
Improved Consistent Sampling, Weighted Minhash and L1 Sketching,
http://static.googleusercontent.com/media/research.google.com/en//pubs/archive/36928.pdf
'''
import collections
import numpy as np


class WeightedMinHash(object):

    def __init__(self, hashes, seed):
        '''
        Create a WeightedMinHash object given the hashes and the seed
        used to generate it.
        '''
        self.hashes = hashes
        self.seed = seed
    
    def __len__(self):
        '''
        Return the number of hashes as the size
        '''
        return len(self.hashes)

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
                    different numbers hashes")
        # Check how many pairs of (k, t) hashes are equal
        intersection = 0
        for this, that in zip(self.hashes, other.hashes):
            if np.array_equal(this, that):
                intersection += 1
        return float(intersection) / float(len(self))
 

class WeightedMinHashGenerator(object):

    def __init__(self, dim, sample_size=256, seed=1):
        '''
        Initialize the generator with the number of dimensions of input 
        vectors, number of samples, and the seed for creating random parameters.
        '''
        self.dim = dim
        self.sample_size = sample_size
        self.seed = seed
        generator = np.random.RandomState(seed=seed)
        self.rs = generator.gamma(2, 1, (sample_size, dim))
        self.ln_cs = np.log(generator.gamma(2, 1, (sample_size, dim)))
        self.betas = generator.uniform(0, 1, (sample_size, dim))

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
            v = np.array(v)
        hashes = np.zeros((self.sample_size, 2))
        for i in range(self.sample_size):
            t = np.floor((np.log(v) / self.rs[i]) + self.betas[i])
            ln_y = (t - self.betas[i]) * self.rs[i]
            ln_a = self.ln_cs[i] - ln_y - self.rs[i]
            k = np.argmin(ln_a)
            hashes[i][0], hashes[i][1] = k, int(t[k])
        return WeightedMinHash(hashes, self.seed)

