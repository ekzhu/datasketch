'''
Benchmarking the performance and accuracy of b-bi MinHash.
'''
import time, logging, random
logging.basicConfig(level=logging.INFO)
import pyhash
import numpy as np
from datasketch.minhash import MinHash
from datasketch.b_bit_minhash import bBitMinHash
from similarity_benchmark import _get_exact, _gen_data,\
        Hash, _b_bit_minhash_jaccard

def _run_minhash(A, B, data, seed, bs, num_perm):
    (a_start, a_end), (b_start, b_end) = A, B
    hasher = pyhash.murmur3_32()
    m1 = MinHash(num_perm=num_perm, hashobj=Hash)
    m2 = MinHash(num_perm=num_perm, hashobj=Hash)
    for i in xrange(a_start, a_end):
        m1.update(hasher(data[i], seed=seed))
    for i in xrange(b_start, b_end):
        m2.update(hasher(data[i], seed=seed))
    return [m1.jaccard(m2)] + \
            [_b_bit_minhash_jaccard(m1, m2, b) for b in bs]

def _run_test(A, B, data, n, bs, num_perm):
    logging.info("Run tests with A = (%d, %d), B = (%d, %d), n = %d"
            % (A[0], A[1], B[0], B[1], n))
    runs = np.array([_run_minhash(A, B, data, i, bs, num_perm)
        for i in xrange(n)]).T
    return runs

def run_full_tests(attr_pairs, data, n, bs, num_perm):
    return [_run_test(A, B, data, n, bs, num_perm)
            for A, B in attr_pairs]

def plot(result, bs, exact_sims, num_perm, bins, save):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    num_row = 1
    num_col = len(result)
    basesize = 5
    size = (basesize*num_col, basesize*num_row)
    fig, axes = plt.subplots(num_row, num_col, sharey=True,
            sharex=True, figsize=size)
    for i, runs in enumerate(result):
        minhash = sorted(runs[0])
        bbits = [sorted(r) for r in runs[1:]]
        exact_sim = exact_sims[i]
        ax = axes[i]
        l = ax.plot(minhash, label='MinHash')
        for b, run in zip(bs, bbits):
            l = ax.plot(run, label='%d-bit' % b)
        ax.axhline(exact_sim, color='black', linestyle='--', label='Exact')
        ax.set_title("%d perm funcs, exact = %.2f" % (num_perm, exact_sim))
        ax.grid()
        ax.set_xlabel("Runs with random hash functions")
        if i == 0:
            ax.set_ylabel('Jaccard')
        if i == num_col - 1:
            ax.legend(loc='lower right')
    fig.savefig(save)


if __name__ == "__main__":
    data = _gen_data(5000)
    attr_pairs = [((0, 3000), (2000, 5000)),
                  ((0, 3500), (1500, 5000)),
                  ((0, 4500), (500, 5000))]
    num_perm = 128
    bs = [1, 2, 3]
    n = 100
    save = "b_bit_minhash_benchmark.png"
    bins = [i*0.02 for i in range(51)]
    exact_sims = [_get_exact(A, B) for A, B in attr_pairs]
    result = run_full_tests(attr_pairs, data, n, bs, num_perm)
    plot(result, bs, exact_sims, num_perm, bins, save)
