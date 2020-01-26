'''
Test the accuracy of inclusion estimation using data sketches
Must be using Python 2 as pyhash does not support Python 3
HyperLogLog inclusion score is computed using cardinality estimate
and inclusion-exclusion principle.
MinHash inclusion score is computed using Jaccard estiamte,
inclusion-exclusion principle, and the exact cardinality.
'''
import time, logging, random, struct
import pyhash
from datasketch.hyperloglog import HyperLogLog
from datasketch.minhash import MinHash

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
    return float(overlap) / abs(a_start - a_end)

def _minhash_inclusion(m1, m2):
    c1 = m1.count()
    c2 = m2.count()
    j = m1.jaccard(m2)
    return (j / (j + 1.0)) * (1.0 + float(c2) / float(c1))

def _hyperloglog_inclusion(h1, h2):
    c1 = h1.count()
    if c1 == 0.0:
        return 1.0
    c2 = h2.count()
    uc = HyperLogLog.union(h1, h2).count()
    ic = c1 + c2 - uc
    return ic / c1

def _run_hyperloglog(A, B, data, seed, p):
    (a_start, a_end), (b_start, b_end) = A, B
    hasher = pyhash.murmur3_32()
    h1 = HyperLogLog(p=p, hashobj=Hash)
    h2 = HyperLogLog(p=p, hashobj=Hash)
    for i in xrange(a_start, a_end):
        h1.update(hasher(data[i], seed=seed))
    for i in xrange(b_start, b_end):
        h2.update(hasher(data[i], seed=seed))
    return _hyperloglog_inclusion(h1, h2)

def _run_minhash(A, B, data, seed, p):
    (a_start, a_end), (b_start, b_end) = A, B
    hasher = pyhash.murmur3_32()
    m1 = MinHash(num_perm=2**p, hashobj=Hash)
    m2 = MinHash(num_perm=2**p, hashobj=Hash)
    for i in xrange(a_start, a_end):
        m1.update(hasher(data[i], seed=seed))
    for i in xrange(b_start, b_end):
        m2.update(hasher(data[i], seed=seed))
    return _minhash_inclusion(m1, m2)

def _run_test(A, B, data, n, p):
    logging.info("Running HyperLogLog with p = %d" % p)
    hll_runs = [_run_hyperloglog(A, B, data, i, p) for i in xrange(n)]
    logging.info("Running MinHash with num_perm = %d" % 2**p)
    minhash_runs = [_run_minhash(A, B, data, i, p) for i in xrange(n)]
    return (hll_runs, minhash_runs)


def run_full_tests(A, B, data, n, p_list):
    logging.info("Run tests with A = (%d, %d), B = (%d, %d), n = %d"
            % (A[0], A[1], B[0], B[1], n))
    return [_run_test(A, B, data, n, p) for p in p_list]


def plot_hist(ax, est_sims, bins, title, exact_sim):
    ax.hist(est_sims, bins, histtype='stepfilled', facecolor='g', alpha=0.75)
    ax.axvline(exact_sim, color='black', linestyle='--')
    ax.set_title(title)
    ax.set_xlabel("Estimation (Actual = %.4f)" % exact_sim)


def plot(result, p_list, exact_sim, bins, save):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    num_row = 2
    num_col = len(result)
    basesize = 5
    size = (basesize*num_col, basesize*num_row)
    fig, axes = plt.subplots(num_row, num_col, sharex=True, figsize=size)
    for i, (hll, minhash) in enumerate(result):
        title = "HyperLogLog p = " + r"$2^{%d}$" % p_list[i]
        plot_hist(axes[0][i], hll, bins, title, exact_sim)
        title = "MinHash num_perm = " + r"$2^{%d}$" % p_list[i]
        plot_hist(axes[1][i], minhash, bins, title, exact_sim)
    fig.savefig(save)


if __name__ == "__main__":
    data = _gen_data(5000)
    A = (0, 3000)
    B = (2500, 5000)
    exps = [6, 8, 10]
    p_list = exps
    n = 100
    save = "inclusion_benchmark.png"
    bins = [i*0.02 for i in range(50)]
    exact_sim = _get_exact(A, B)
    result = run_full_tests(A, B, data, n, p_list)
    plot(result, p_list, exact_sim, bins, save)
