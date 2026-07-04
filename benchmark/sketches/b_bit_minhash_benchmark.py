"""
Benchmarking the accuracy of b-bit MinHash against the full MinHash.

For each true Jaccard similarity, two sets with exactly that similarity
are sketched 100 times, each run using a different set of random hash
functions (a different MinHash seed). The plot shows the sorted Jaccard
estimates of all runs for the full MinHash and for 1-, 2- and 3-bit
MinHash against the exact similarity, so the vertical span of each line
is the spread of the estimator; the standard deviation of each estimator
is printed in each panel.
"""
import logging
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasketch.minhash import MinHash
from datasketch.b_bit_minhash import bBitMinHash

logging.basicConfig(level=logging.INFO)

# Produce some bytes
int_bytes = lambda x: ("a-%d-%d" % (x, x)).encode("utf-8")


def make_sets(jaccard, union_size):
    """Two lists of distinct tokens whose exact Jaccard is `jaccard`."""
    num_common = int(round(jaccard * union_size))
    num_only = (union_size - num_common) // 2
    tokens = [int_bytes(i) for i in range(num_common + 2 * num_only)]
    common = tokens[:num_common]
    a_only = tokens[num_common : num_common + num_only]
    b_only = tokens[num_common + num_only :]
    return common + a_only, common + b_only


def run(jaccard, union_size, num_perm, num_runs, bits):
    estimates = {"minhash": []}
    for b in bits:
        estimates[b] = []
    a, b_set = make_sets(jaccard, union_size)
    for seed in range(1, num_runs + 1):
        m1 = MinHash(num_perm=num_perm, seed=seed)
        m2 = MinHash(num_perm=num_perm, seed=seed)
        m1.update_batch(a)
        m2.update_batch(b_set)
        estimates["minhash"].append(m1.jaccard(m2))
        for b in bits:
            estimates[b].append(bBitMinHash(m1, b).jaccard(bBitMinHash(m2, b)))
    return estimates


num_perm = 128
num_runs = 100
union_size = 5000
jaccards = [0.2, 0.4, 0.8]
bits = [1, 2, 3]
output = "b_bit_minhash_benchmark.png"

fig, axes = plt.subplots(1, len(jaccards), sharex=True, sharey=True, figsize=(15, 4.5))
runs = range(1, num_runs + 1)
for ax, jaccard in zip(axes, jaccards):
    logging.info("> Running with exact Jaccard %.1f" % jaccard)
    estimates = run(jaccard, union_size, num_perm, num_runs, bits)
    series = [("MinHash", estimates["minhash"])]
    series += [("%d-bit" % b, estimates[b]) for b in bits]
    # Sort each estimator's runs so the slope and vertical span of a line
    # show the spread of that estimator directly.
    for label, values in series:
        ax.plot(runs, sorted(values), label=label)
    ax.axhline(jaccard, color="black", linestyle="--", label="Exact")
    # Print the standard deviation of each estimator, in the corner
    # away from the curves.
    y0 = 0.29 if jaccard >= 0.6 else 0.96
    ax.text(0.03, y0, "std of estimates", transform=ax.transAxes, fontsize=9, va="top")
    for i, (label, values) in enumerate(series):
        ax.text(
            0.03,
            y0 - 0.055 * (i + 1),
            "%s: %.3f" % (label, np.std(values)),
            transform=ax.transAxes,
            fontsize=9,
            color="C%d" % i,
            va="top",
        )
    ax.set_title("%d perm funcs, exact = %.1f" % (num_perm, jaccard))
    ax.set_xlabel("Runs with random hash functions, sorted by estimate")
    ax.set_ylim(0.0, 1.0)
    ax.grid()
axes[0].set_ylabel("Jaccard")
axes[-1].legend(loc="lower right")

plt.tight_layout()
fig.savefig(output)
logging.info("Plot saved to %s" % output)
