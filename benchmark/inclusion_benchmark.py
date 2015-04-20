'''
Test the accuracy of inclusion estimation using data sketches
Must be using Python 2 as pyhash does not support Python 3
'''
import time, logging, random, struct
import pyhash
from datasketch.hyperloglog import HyperLogLog

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


def _run_hyperloglog(A, B, data, seed, p):
    (a_start, a_end), (b_start, b_end) = A, B
    hasher = pyhash.murmur3_32()
    h1 = HyperLogLog(p=p)
    h2 = HyperLogLog(p=p)
    for i in xrange(a_start, a_end):
        h1.digest(Hash(hasher(data[i], seed=seed)))
    for i in xrange(b_start, b_end):
        h2.digest(Hash(hasher(data[i], seed=seed)))
    return h1.inclusion(h2)


def _run_test(A, B, data, n, p):
    logging.info("Running HyperLogLog with p = %d" % p)
    hll_runs = [_run_hyperloglog(A, B, data, i, p) for i in xrange(n)]
    return hll_runs


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
    num_row = 1
    num_col = len(result)
    basesize = 5
    size = (basesize*num_col, basesize*num_row)
    fig, axes = plt.subplots(num_row, num_col, sharex=True, figsize=size)
    for i, hll in enumerate(result):
        title = "HyperLogLog p = " + r"$2^{%d}$" % p_list[i]
        plot_hist(axes[i], hll, bins, title, exact_sim)
    fig.savefig(save)


if __name__ == "__main__":
    data = _gen_data(5000)
    A = (1, 50)
    B = (25, 100)
    exps = [6, 8, 10]
    p_list = exps
    n = 100
    save = "inclusion_benchmark.png"
    bins = [i*0.02 for i in range(50)]
    exact_sim = _get_exact(A, B)
    result = run_full_tests(A, B, data, n, p_list)
    plot(result, p_list, exact_sim, bins, save)
