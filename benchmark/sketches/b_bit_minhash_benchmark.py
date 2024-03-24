"""
Benchmarking the performance and accuracy of b-bit MinHash.
"""
import time, logging
from numpy import random
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasketch.minhash import MinHash
from datasketch.b_bit_minhash import bBitMinHash
from datasketch.hashfunc import *

logging.basicConfig(level=logging.INFO)

# Produce some bytes
int_bytes = lambda x: ("a-%d-%d" % (x, x)).encode("utf-8")


def run_perf(card, num_perm, num_bits):
    dur = 0
    n_trials = 5
    for i in range(n_trials):
        m = MinHash(num_perm=num_perm)
        logging.info("MinHash using %d permutation functions" % num_perm)
        start = time.perf_counter()
        for i in range(card):
            m.update(int_bytes(i))

        b = bBitMinHash(m, num_bits)
        duration = time.perf_counter() - start
        dur += duration
        logging.info("Digested %d hashes in %.4f sec" % (card, duration))
    return dur / n_trials


def _run_acc(size, seed, num_perm, num_bits):
    m = MinHash(num_perm=num_perm)
    s = set()
    random.seed(seed)
    for i in range(size):
        v = int_bytes(random.randint(1, size))
        m.update(v)
        s.add(v)

    b = bBitMinHash(m, num_bits)
    return (b, s)


def run_acc(size, num_perm, num_bits):
    logging.info("MinHash using %d permutation functions" % num_perm)
    m1, s1 = _run_acc(size, 1, num_perm, num_bits)
    m2, s2 = _run_acc(size, 4, num_perm, num_bits)
    j = float(len(s1.intersection(s2))) / float(len(s1.union(s2)))
    j_e = m1.jaccard(m2)
    err = abs(j - j_e)
    return err


num_perms = range(10, 256, 20)
num_bits = [1, 2, 3, 4, 8, 12, 16, 32]
bit_colors = colors = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
]
output = "b_bit_minhash_benchmark.png"

logging.info("> Running performance tests")
card = 5000
perf_times = {}
for b in num_bits:
    run_times = [run_perf(card, n, b) for n in num_perms]
    perf_times[b] = run_times


logging.info("> Running accuracy tests")
size = 5000
errors = {}
for b in num_bits:
    errs = [run_acc(size, n, b) for n in num_perms]
    errors[b] = errs

logging.info("> Plotting result")
fig, axe = plt.subplots(1, 2, sharex=True, figsize=(10, 4))
ax = axe[1]
for i, b in enumerate(num_bits):
    ax.plot(
        num_perms, perf_times[b], marker="+", color=bit_colors[i], label=f"{b} bits"
    )
ax.set_xlabel("Number of permutation functions")
ax.set_ylabel("Running time (sec)")
ax.set_title("MinHash performance")
ax.grid()
ax.legend()
ax = axe[0]
for i, b in enumerate(num_bits):
    ax.plot(num_perms, errors[b], marker="+", color=bit_colors[i], label=f"{b} bits")
ax.set_xlabel("Number of permutation functions")
ax.set_ylabel("Absolute error in Jaccard estimation")
ax.set_title("MinHash accuracy")
ax.grid()
ax.legend()

plt.tight_layout()
fig.savefig(output)
logging.info("Plot saved to %s" % output)
