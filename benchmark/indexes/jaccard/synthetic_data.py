import sys
import argparse
import random
import collections

import scipy.stats
import numpy.random
import numpy


def generate_sets(num_sets, vocabulary_size=10000, max_set_size=1000,
        token_random_seed=41, set_size_random_seed=42):
    vocabulary = numpy.arange(vocabulary_size)
    token_rs = numpy.random.RandomState(token_random_seed)
    set_rs = numpy.random.RandomState(set_size_random_seed)
    sets = collections.deque([])
    for _ in range(num_sets):
        # Get set size.
        size = set_rs.randint(1, max_set_size+1)
        # Sample from vocabulary.
        s = token_rs.choice(vocabulary, size, replace=False)
        sets.append(s)
        sys.stdout.write(f"\rCreated {len(sets)} sets.")
    return list(sets), range(num_sets)
        

def sample_sets(sets, keys, ratio):
    n = len(sets)
    m = int(len(sets) * ratio)
    indices = numpy.random.choice(numpy.arange(n), m, replace=False)
    return [sets[i] for i in indices], [keys[i] for i in indices]

