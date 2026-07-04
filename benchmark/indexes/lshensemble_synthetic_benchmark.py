"""
Effect of the number of partitions on MinHashLSHEnsemble, by index-set
cardinality distribution.

LSH Ensemble partitions indexed sets by size and tunes the LSH
parameters per partition, so more partitions help in proportion to how
skewed the set sizes are. This benchmark makes that concrete: containment
threshold queries on a synthetic corpus, with ``num_part`` on the x-axis
and one figure per index-size distribution (each labelled with its size
range), comparing the ensemble against a linear scan.

Corpus: ``--n-background`` random index sets plus, for each of
``--queries`` held-out queries of a fixed ``--query-size`` tokens, 12
planted neighbours at containments spread over (0.05, 0.95) -- so every
query has true neighbours across the threshold. Index-set sizes follow
one of:

- constant: every set the same size (no skew -- partitioning cannot help);
- log-uniform over a wide range (moderate skew);
- zipf: a truncated Pareto, the power-law shape reported for real
  open-data corpora in the LSH Ensemble paper (heavy skew).

The linear scan converts the MinHash Jaccard estimate of every indexed
set to a containment estimate (c = J(|Q|+|X|) / (|Q|(1+J))) and thresholds
it; it does not depend on ``num_part``, so it appears as a reference line.
Ground truth is exact containment >= threshold; averages cover queries
with non-empty ground truth. ``--plot-from`` re-renders from a saved JSON.

Usage:
    python lshensemble_synthetic_benchmark.py --out-dir figures/
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from datasketch import MinHash, MinHashLSHEnsemble  # noqa: E402

import synthetic_data  # noqa: E402

NUM_PARTS = [1, 2, 4, 8, 16, 32]
NUM_PERM = 256
THRESHOLD = 0.5
QUERY_SIZE = 200
M = 8

# One figure per distribution: key -> (size_dist, index_size_range, label).
DISTS = {
    "constant": ("uniform", (1000, 1001), "Constant index size 1000"),
    "loguniform": ("loguniform", (100, 10_000),
                   "Log-uniform index sizes 100-10,000"),
    "zipf": ("zipf", (100, 10_000), "Zipfian index sizes 100-10,000"),
}


def get_precision_recall(found, reference):
    reference = set(reference)
    intersect = sum(1 for i in found if i in reference)
    precision = float(intersect) / float(len(found)) if found else 0.0
    recall = float(intersect) / float(len(reference)) if reference else 1.0
    if len(found) == len(reference) == 0:
        precision = recall = 1.0
    return precision, recall


def build_minhashes(sets_, num_perm, scheme):
    ms = []
    for s in sets_:
        m = MinHash(num_perm=num_perm, scheme=scheme)
        m.update_batch([int(t).to_bytes(8, "little") for t in s])
        ms.append(m)
    return ms


def containment_estimates(index_hv, query_hv, sizes_index, sizes_query):
    """MinHash-estimated containment (queries x index): c = J(|Q|+|X|) /
    (|Q|(1+J))."""
    est = np.empty((len(query_hv), len(index_hv)), dtype=np.float32)
    for start in range(0, len(query_hv), 64):
        chunk = query_hv[start : start + 64]
        j = (chunk[:, None, :] == index_hv[None, :, :]).mean(
            axis=2, dtype=np.float32)
        q = sizes_query[start : start + 64, None]
        est[start : start + 64] = j * (q + sizes_index[None, :]) / (q * (1.0 + j))
    return est


def distribution_curves(dist, args):
    """Precision/recall/90pct-time vs num_part for one index-size
    distribution, plus the num_part-independent linear-scan reference."""
    size_dist, size_range, _ = DISTS[dist]
    index_sets, query_sets = synthetic_data.containment_corpus(
        shape="spread", query_size=args.query_size,
        index_size_range=size_range, size_dist=size_dist,
        n_background=args.n_background, n_queries=args.queries)
    sizes = [len(s) for s in index_sets]
    print(f"[{dist}] index={len(index_sets)} (sizes {min(sizes)}-{max(sizes)}, "
          f"median {int(np.median(sizes))}), queries={len(query_sets)}")

    c_rows = synthetic_data.exact_containment(index_sets, query_sets)
    gt = [set(np.flatnonzero(r >= args.threshold).tolist()) for r in c_rows]
    ev = [q for q, g in enumerate(gt) if g]

    index_mh = build_minhashes(index_sets, args.num_perm, args.scheme)
    query_mh = build_minhashes(query_sets, args.num_perm, args.scheme)
    index_hv = np.stack([m.hashvalues for m in index_mh])
    query_hv = np.stack([m.hashvalues for m in query_mh])
    si = np.array([len(s) for s in index_sets], dtype=np.float32)
    sq = np.array([len(s) for s in query_sets], dtype=np.float32)
    entries = [(k, m, int(si[k])) for k, m in enumerate(index_mh)]

    est = containment_estimates(index_hv, query_hv, si, sq)
    scan_res = [np.flatnonzero(r >= args.threshold).tolist() for r in est]
    scan_pr = np.mean([get_precision_recall(scan_res[q], gt[q]) for q in ev], axis=0)
    timing = random.Random(7).sample(
        range(len(query_sets)), min(args.timing_queries, len(query_sets)))
    scan_times = []
    for qi in timing:
        m, q_size = query_mh[qi], sq[qi]
        t0 = time.perf_counter()
        _ = [k for k, mx in enumerate(index_mh)
             for j in (m.jaccard(mx),)
             if j * (q_size + si[k]) / (q_size * (1.0 + j)) >= args.threshold]
        scan_times.append(time.perf_counter() - t0)

    curves = {"precision": [], "recall": [], "time": [],
              "scan_precision": float(scan_pr[0]), "scan_recall": float(scan_pr[1]),
              "scan_time": float(np.percentile(scan_times, 90) * 1000),
              "median_size": int(np.median(sizes))}
    for num_part in args.num_parts:
        ensemble = MinHashLSHEnsemble(
            threshold=args.threshold, num_perm=args.num_perm,
            num_part=num_part, m=args.m)
        ensemble.index(entries)
        results, times = [], []
        for qi, m in enumerate(query_mh):
            t0 = time.perf_counter()
            results.append(list(ensemble.query(m, int(sq[qi]))))
            times.append(time.perf_counter() - t0)
        pr = np.mean([get_precision_recall(results[q], gt[q]) for q in ev], axis=0)
        curves["precision"].append(float(pr[0]))
        curves["recall"].append(float(pr[1]))
        curves["time"].append(float(np.percentile(times, 90) * 1000))
        print(f"[{dist}] num_part={num_part:>2} P={pr[0]:.2f} R={pr[1]:.2f}")
    return curves


def plot_distribution(dist, curves, study):
    num_parts = study["num_parts"]
    label = DISTS[dist][2] if dist in DISTS else dist
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharex=True)
    panels = [("Average Precision", "precision", "scan_precision"),
              ("Average Recall", "recall", "scan_recall"),
              ("90 Percentile Query Time (ms)", "time", "scan_time")]
    for ax, (ylabel, key, scan_key) in zip(axes, panels):
        ax.plot(num_parts, curves[key], marker="o", color="C0", label="LSH Ensemble")
        ax.axhline(curves[scan_key], ls="--", color="C3", lw=1.5, label="Linear scan")
        ax.set_ylabel(ylabel)
        ax.set_xlabel("num_part")
        ax.set_xscale("log", base=2)
        ax.set_xticks(num_parts, labels=[str(p) for p in num_parts])
        ax.grid(alpha=0.4)
    axes[0].set_ylim(0.0, 1.02)
    axes[1].set_ylim(0.0, 1.02)
    axes[0].legend(loc="best", fontsize=9)
    fig.suptitle(f"{label}  (threshold={study['threshold']}, "
                 f"query size={QUERY_SIZE}, num_perm={study['num_perm']}, "
                 f"m={study['m']})", y=1.0)
    fig.tight_layout()
    out = f"{study['out_dir']}/lshensemble_{dist}_benchmark.png"
    fig.savefig(out, pad_inches=0.05, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("wrote", out)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--num-parts", type=int, nargs="+", default=NUM_PARTS)
    p.add_argument("--num-perm", type=int, default=NUM_PERM)
    p.add_argument("--threshold", type=float, default=THRESHOLD)
    p.add_argument("--query-size", type=int, default=QUERY_SIZE)
    p.add_argument("--n-background", type=int, default=5000)
    p.add_argument("--queries", type=int, default=500)
    p.add_argument("--m", type=int, default=M)
    p.add_argument("--dists", nargs="+", default=list(DISTS), choices=list(DISTS))
    p.add_argument("--scheme", default="affine32",
                   choices=["affine32", "affine64", "legacy"])
    p.add_argument("--timing-queries", type=int, default=200)
    p.add_argument("--plot-from", metavar="JSON")
    p.add_argument("--out-dir", default=".")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.plot_from:
        with open(args.plot_from) as f:
            study = json.load(f)
        study["out_dir"] = args.out_dir
        for dist in study["dists"]:
            plot_distribution(dist, study["curves"][dist], study)
        return

    study = {"threshold": args.threshold, "num_perm": args.num_perm, "m": args.m,
             "num_parts": args.num_parts, "dists": args.dists,
             "out_dir": args.out_dir, "curves": {}}
    for dist in args.dists:
        study["curves"][dist] = distribution_curves(dist, args)
    with open(f"{args.out_dir}/lshensemble_partitions_results.json", "w") as f:
        json.dump(study, f, indent=1)
    for dist in args.dists:
        plot_distribution(dist, study["curves"][dist], study)


if __name__ == "__main__":
    main()
