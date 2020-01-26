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

def _run_hyperloglog(data, seed, p):
    hasher = pyhash.murmur3_32()
    h = HyperLogLog(p=p, hashobj=Hash)
    for d in data:
        h.update(hasher(d, seed=seed))
    return h.count()

def _run_minhash(data, seed, p):
    hasher = pyhash.murmur3_32()
    m = MinHash(num_perm=2**p, hashobj=Hash)
    for d in data:
        m.update(hasher(d, seed=seed))
    return m.count()

def _run_test(data, n, p):
    logging.info("Running HyperLogLog with p = %d" % p)
    hll_runs = [_run_hyperloglog(data, i, p) for i in xrange(n)]
    logging.info("Running MinHash with num_perm = %d" % 2**p)
    minhash_runs = [_run_minhash(data, i, p) for i in xrange(n)]
    return (hll_runs, minhash_runs)


def run_full_tests(data, n, p_list):
    logging.info("Run tests with n = %d" % (n))
    return [_run_test(data, n, p) for p in p_list]


def plot_hist(ax, est_cards, bins, title, exact_card):
    errors = [float(exact_card - c)/float(exact_card) for c in est_cards]
    errors.sort()
    ax.plot(errors, 'g.', markersize=12)
    # ax.hist(errors, histtype='stepfilled', facecolor='g', alpha=0.75)
    ax.set_title(title)


def plot(result, p_list, exact_card, bins, save):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    num_row = 2
    num_col = len(result)
    basesize = 5
    size = (basesize*num_col, basesize*num_row)
    fig, axes = plt.subplots(num_row, num_col, sharex=True, sharey=True,
            figsize=size)
    for i, (hll, minhash) in enumerate(result):
        title = "HyperLogLog Error Rate p = " + r"$2^{%d}$" % p_list[i]
        plot_hist(axes[0][i], hll, bins, title, exact_card)
        title = "MinHash Error Rate num_perm = " + r"$2^{%d}$" % p_list[i]
        plot_hist(axes[1][i], minhash, bins, title, exact_card)
    fig.suptitle("Exact cardinality = %d" % exact_card)
    fig.savefig(save)


if __name__ == "__main__":
    exact_card = 5000
    data = _gen_data(exact_card)
    exps = [6, 8, 10]
    p_list = exps
    n = 100
    save = "cardinality_benchmark.png"
    bins = 30
    result = run_full_tests(data, n, p_list)
    plot(result, p_list, exact_card, bins, save)
