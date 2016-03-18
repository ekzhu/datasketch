'''
Test the accuracy of similarity estimation using data sketches
Must be using Python 2 as pyhash does not support Python 3
'''
import time, logging, random, struct
import pyhash
import numpy as np
from datasketch.hyperloglog import HyperLogLog
from datasketch.minhash import MinHash
from datasketch.b_bit_minhash import bBitMinHash

logging.basicConfig(level=logging.INFO)

# Produce some bytes
int_bytes = lambda x : ("a-%d-%d" % (x, x)).encode('utf-8')

class Hash(object):
    def __init__(self, h):
        self.h = h
    def digest(self):
        return struct.pack('<I', self.h)

def _gen_data(size):
    return [int_bytes(i) for i in range(size)]

def _get_exact(A, B):
    (a_start, a_end) = A
    (b_start, b_end) = B
    overlap = min(a_end, b_end) - max(a_start, b_start)
    if overlap < 0:
        overlap = 0
    union = max(a_end, b_end) - min(a_start, b_start)
    return float(overlap) / union

def _hyperloglog_jaccard(h1, h2):
    c1 = h1.count()
    c2 = h2.count()
    uc = HyperLogLog.union(h1, h2).count()
    if uc == 0.0:
        return 1.0
    ic = c1 + c2 - uc
    return ic / uc

def _b_bit_minhash_jaccard(m1, m2, b):
    return bBitMinHash(m1, b).jaccard(bBitMinHash(m2, b))

def _run_minhash(A, B, data, seed, num_perm, b):
    (a_start, a_end), (b_start, b_end) = A, B
    hasher = pyhash.murmur3_32()
    m1 = MinHash(num_perm=num_perm, hashobj=Hash)
    m2 = MinHash(num_perm=num_perm, hashobj=Hash)
    for i in xrange(a_start, a_end):
        m1.update(hasher(data[i], seed=seed))
    for i in xrange(b_start, b_end):
        m2.update(hasher(data[i], seed=seed))
    return [m1.jaccard(m2), _b_bit_minhash_jaccard(m1, m2, b)]

def _run_hyperloglog(A, B, data, seed, p):
    (a_start, a_end), (b_start, b_end) = A, B
    hasher = pyhash.murmur3_32()
    h1 = HyperLogLog(p=p, hashobj=Hash)
    h2 = HyperLogLog(p=p, hashobj=Hash)
    for i in xrange(a_start, a_end):
        h1.update(hasher(data[i], seed=seed))
    for i in xrange(b_start, b_end):
        h2.update(hasher(data[i], seed=seed))
    return _hyperloglog_jaccard(h1, h2)

def _run_test(A, B, data, n, p, num_perm, b):
    logging.info("Running MinHash with num_perm = %d" % num_perm)
    minhash_runs, bbit_runs = np.array([_run_minhash(A, B, data,
            i, num_perm, b)
        for i in xrange(n)]).T
    logging.info("Running HyperLogLog with p = %d" % p)
    hll_runs = [_run_hyperloglog(A, B, data, i, p) for i in xrange(n)]
    return (minhash_runs, bbit_runs, hll_runs)


def run_full_tests(A, B, data, n, p_list, num_perm_list, b):
    logging.info("Run tests with A = (%d, %d), B = (%d, %d), n = %d"
            % (A[0], A[1], B[0], B[1], n))
    return [_run_test(A, B, data, n, p, num_perm, b)
            for p, num_perm in zip(p_list, num_perm_list)]


def plot_hist(ax, est_sims, bins, title, exact_sim):
    ax.hist(est_sims, bins, histtype='stepfilled', facecolor='g', alpha=0.75)
    ax.axvline(exact_sim, color='black', linestyle='--')
    ax.set_title(title)
    ax.set_xlabel("Estimation (Actual = %.4f)" % exact_sim)


def plot(result, p_list, num_perm_list, exact_sim, bins, save, b):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    num_row = 3
    num_col = len(result)
    basesize = 5
    size = (basesize*num_col, basesize*num_row)
    fig, axes = plt.subplots(num_row, num_col, sharey=True,
            sharex=True, figsize=size)
    for i, (minhash, bbit, hll) in enumerate(result):
        title = "MinHash %d perm funcs" % num_perm_list[i]
        plot_hist(axes[0][i], minhash, bins, title, exact_sim)
        title = "%d-bit MinHash %d perm funcs" % (b, num_perm_list[i])
        plot_hist(axes[1][i], bbit, bins, title, exact_sim)
        title = "HyperLogLog p = " + r"$2^{%d}$" % p_list[i]
        plot_hist(axes[2][i], hll, bins, title, exact_sim)
    fig.savefig(save)


if __name__ == "__main__":
    data = _gen_data(5000)
    A = (0, 3550)
    B = (1450, 5000)
    exps = [6, 8, 10]
    p_list = exps
    num_perm_list = list([2**i for i in exps])
    b = 1
    n = 100
    save = "similarity_benchmark.png"
    bins = [i*0.02 for i in range(50)]
    exact_sim = _get_exact(A, B)
    result = run_full_tests(A, B, data, n, p_list, num_perm_list, b)
    plot(result, p_list, num_perm_list, exact_sim, bins, save, b)
