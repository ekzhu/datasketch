'''
Benchmarking the performance and accuracy of MinHash.

Produces minhash_benchmark.png, comparing the default SHA1 hash function
with each of the optional hash functions (mmh3, xxhash, pyfarmhash) that
is installed. This is the figure shown in docs/minhash.rst.
'''
import math
import time, logging
from numpy import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datasketch.minhash import MinHash

logging.basicConfig(level=logging.INFO)

# Produce some bytes
int_bytes = lambda x : ("a-%d-%d" % (x, x)).encode('utf-8')

def hash_functions():
    # None selects the default hash function of the MinHash scheme (SHA1).
    funcs = {"sha1": None}
    try:
        import mmh3
        funcs["mmh3"] = lambda d: mmh3.hash(d, signed=False)
    except ImportError:
        logging.warning("mmh3 not installed, skipping")
    try:
        import xxhash
        funcs["xxh"] = lambda d: xxhash.xxh32_intdigest(d)
    except ImportError:
        logging.warning("xxhash not installed, skipping")
    try:
        import farmhash
        funcs["farmhash"] = lambda d: farmhash.hash32(d)
    except ImportError:
        logging.warning("pyfarmhash not installed, skipping")
    return funcs

def run_perf(card, num_perm, hashfunc, n_trials=5):
    logging.info("MinHash using %d permutation functions" % num_perm)
    best = None
    for _ in range(n_trials):
        m = MinHash(num_perm=num_perm, hashfunc=hashfunc)
        start = time.perf_counter()
        for i in range(card):
            m.update(int_bytes(i))
        duration = time.perf_counter() - start
        if best is None or duration < best:
            best = duration
    logging.info("Digested %d hashes in %.4f sec" % (card, best))
    return best


def _run_acc(size, data_seed, minhash_seed, num_perm, hashfunc):
    m = MinHash(num_perm=num_perm, seed=minhash_seed, hashfunc=hashfunc)
    s = set()
    random.seed(data_seed)
    for i in range(size):
        v = int_bytes(random.randint(1, size))
        m.update(v)
        s.add(v)
    return (m, s)

def run_acc(size, num_perm, hashfunc, n_trials=30):
    logging.info("MinHash using %d permutation functions" % num_perm)
    errs = []
    for trial in range(n_trials):
        m1, s1 = _run_acc(size, 2 * trial + 1, trial, num_perm, hashfunc)
        m2, s2 = _run_acc(size, 2 * trial + 2, trial, num_perm, hashfunc)
        j = float(len(s1.intersection(s2)))/float(len(s1.union(s2)))
        j_e = m1.jaccard(m2)
        errs.append(abs(j - j_e))
    return sum(errs) / len(errs)

num_perms = range(10, 256, 20)
card = 5000
size = 5000
output = "minhash_benchmark.png"

results = {}
for name, hashfunc in hash_functions().items():
    logging.info("> Benchmarking MinHash with hash function %s" % name)
    logging.info("> Running performance tests")
    run_times = [1000.0 * run_perf(card, n, hashfunc) for n in num_perms]
    logging.info("> Running accuracy tests")
    errs = [run_acc(size, n, hashfunc) for n in num_perms]
    results[name] = (run_times, errs)

logging.info("> Plotting result")
fig, axe = plt.subplots(1, 2, sharex=True, figsize=(10, 4))
ax = axe[1]
for name, (run_times, _) in results.items():
    ax.plot(num_perms, run_times, marker='+', label=name)
# A coarse, zero-based time scale: the differences between hash functions
# are small compared to the total update time.
max_ms = max(max(run_times) for run_times, _ in results.values())
ax.set_ylim(0, math.ceil(max_ms * 1.3 / 10.0) * 10)
ax.set_xlabel("Number of permutation functions")
ax.set_ylabel("Running time (ms)")
ax.set_title("MinHash performance")
ax.grid()
ax.legend()
ax = axe[0]
for name, (_, errs) in results.items():
    ax.plot(num_perms, errs, marker='+', label=name)
ax.set_ylim(bottom=0)
ax.set_xlabel("Number of permutation functions")
ax.set_ylabel("Mean absolute error in Jaccard estimation")
ax.set_title("MinHash accuracy")
ax.grid()

plt.tight_layout()
fig.savefig(output)
logging.info("Plot saved to %s" % output)
