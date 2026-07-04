"""
Accuracy/speed benchmarks for MinHashLSH on two synthetic corpora.

Runs the same threshold-query experiment on the two corpus shapes from
``synthetic_data.py`` and writes one figure per shape:

- near-duplicate (``lsh_neardup_benchmark.png``): bimodal similarities;
  each query has 1-3 planted near-duplicates (Jaccard 0.6-0.95) among
  otherwise-disjoint sets, so nothing sits near the threshold.
- clustering (``lsh_clustering_benchmark.png``): borderline-heavy;
  overlapping sets put many pairs near the threshold.

Each figure shows average precision, average recall, and 90th
percentile query time versus ``num_perm`` for ``lsh.query`` and for a
linear scan that thresholds the MinHash estimate of every indexed set;
x-tick labels carry the ``(b, r)`` chosen by the optimizer at each
``num_perm``. Queries are held out of the index. Averages cover queries
with non-empty ground truth (exact Jaccard >= threshold). Linear-scan
result sets are computed vectorized (identical output); its query time
is measured with the plain per-query loop on a fixed-seed subsample.

Usage:
    python lsh_synthetic_benchmark.py --out-dir figures/
"""

from __future__ import annotations

import argparse
import json
import random
import time
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from datasketch import MinHash, MinHashLSH  # noqa: E402

import synthetic_data  # noqa: E402

NUM_PERMS = [32, 64, 96, 128, 160, 192, 224, 256]


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


def estimate_matrix(index_hv: np.ndarray, query_hv: np.ndarray) -> np.ndarray:
    est = np.empty((len(query_hv), len(index_hv)), dtype=np.float32)
    for start in range(0, len(query_hv), 64):
        chunk = query_hv[start : start + 64]
        est[start : start + 64] = (
            chunk[:, None, :] == index_hv[None, :, :]
        ).mean(axis=2, dtype=np.float32)
    return est


def run_case(case: str, index_sets, query_sets, args):
    print(f"[{case}] index={len(index_sets)} sets, queries={len(query_sets)}, "
          f"avg cardinality={np.mean([len(s) for s in index_sets]):.0f}")
    print(f"[{case}] computing exact Jaccard ground truth...")
    j_rows = synthetic_data.exact_jaccard(index_sets, query_sets)
    ground_truth = [
        set(np.flatnonzero(row >= args.threshold).tolist()) for row in j_rows
    ]
    eval_queries = [q for q, gt in enumerate(ground_truth) if gt]
    print(
        f"[{case}] queries with ground truth (J>={args.threshold}): "
        f"{len(eval_queries)}/{len(query_sets)}, median neighbours "
        f"{int(np.median([len(ground_truth[q]) for q in eval_queries]))}"
    )
    timing_sample = random.Random(7).sample(
        range(len(query_sets)), min(args.timing_queries, len(query_sets))
    )

    curves: Dict[str, List[float]] = {
        name: []
        for name in (
            "lsh_precision", "lsh_recall", "lsh_time",
            "scan_precision", "scan_recall", "scan_time",
        )
    }
    params: List[tuple] = []

    for num_perm in args.num_perms:
        print(f"[{case}] num_perm={num_perm}: building sketches...")
        index_mh = build_minhashes(index_sets, num_perm, args.scheme)
        query_mh = build_minhashes(query_sets, num_perm, args.scheme)
        index_hv = np.stack([m.hashvalues for m in index_mh])
        query_hv = np.stack([m.hashvalues for m in query_mh])

        lsh = MinHashLSH(threshold=args.threshold, num_perm=num_perm)
        params.append((lsh.b, lsh.r))
        for key, m in enumerate(index_mh):
            lsh.insert(key, m)
        lsh_results, lsh_times = [], []
        for m in query_mh:
            t0 = time.perf_counter()
            res = lsh.query(m)
            lsh_times.append(time.perf_counter() - t0)
            lsh_results.append(res)

        est = estimate_matrix(index_hv, query_hv)
        scan_results = [
            np.flatnonzero(row >= args.threshold).tolist() for row in est
        ]
        scan_times = []
        for q in timing_sample:
            m = query_mh[q]
            t0 = time.perf_counter()
            _ = [k for k, mt in enumerate(index_mh) if m.jaccard(mt) >= args.threshold]
            scan_times.append(time.perf_counter() - t0)

        pr = np.mean(
            [get_precision_recall(lsh_results[q], ground_truth[q]) for q in eval_queries],
            axis=0,
        )
        curves["lsh_precision"].append(pr[0])
        curves["lsh_recall"].append(pr[1])
        pr = np.mean(
            [get_precision_recall(scan_results[q], ground_truth[q]) for q in eval_queries],
            axis=0,
        )
        curves["scan_precision"].append(pr[0])
        curves["scan_recall"].append(pr[1])
        curves["lsh_time"].append(np.percentile(lsh_times, 90) * 1000)
        curves["scan_time"].append(np.percentile(scan_times, 90) * 1000)
        print(
            f"  (b, r)=({lsh.b}, {lsh.r}) "
            f"lsh P={curves['lsh_precision'][-1]:.3f} R={curves['lsh_recall'][-1]:.3f} "
            f"t90={curves['lsh_time'][-1]:.2f}ms | "
            f"scan P={curves['scan_precision'][-1]:.3f} R={curves['scan_recall'][-1]:.3f} "
            f"t90={curves['scan_time'][-1]:.1f}ms"
        )

    return curves, params


def plot_case(case: str, curves, params, args, acc_ylim, time_ylim):
    titles = {
        "neardup": "Near-duplicate corpus (bimodal similarities)",
        "clustering": "Clustering corpus (borderline-heavy similarities)",
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), sharex=True)
    axes[0].plot(args.num_perms, curves["scan_precision"], marker="+", label="Linearscan")
    axes[0].plot(args.num_perms, curves["lsh_precision"], marker="+", label="LSH")
    axes[0].set_ylabel("Average Precision")
    axes[1].plot(args.num_perms, curves["scan_recall"], marker="+", label="Linearscan")
    axes[1].plot(args.num_perms, curves["lsh_recall"], marker="+", label="LSH")
    axes[1].set_ylabel("Average Recall")
    axes[2].plot(args.num_perms, curves["scan_time"], marker="+", label="Linearscan")
    axes[2].plot(args.num_perms, curves["lsh_time"], marker="+", label="LSH")
    axes[2].set_ylabel("90 Percentile Query Time (ms)")
    axes[2].legend(loc="center right")
    ticks = [
        f"{np_}\n({b},{r})" for np_, (b, r) in zip(args.num_perms, params)
    ]
    axes[0].set_ylim(*acc_ylim)
    axes[1].set_ylim(*acc_ylim)
    axes[2].set_ylim(*time_ylim)
    for ax in axes:
        ax.set_xticks(args.num_perms, labels=ticks, fontsize=8)
        ax.set_xlabel("num_perm over chosen (b, r)")
        ax.grid()
    fig.suptitle(titles[case], y=1.0)
    fig.tight_layout()
    out = f"{args.out_dir}/lsh_{case}_benchmark.png"
    fig.savefig(out, pad_inches=0.05, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("wrote", out)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--num-perms", type=int, nargs="+", default=NUM_PERMS)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--cases", nargs="+", default=["neardup", "clustering"],
                   choices=["neardup", "clustering"])
    p.add_argument("--scheme", default="affine32",
                   choices=["affine32", "affine64", "legacy"])
    p.add_argument("--timing-queries", type=int, default=200)
    p.add_argument("--out-dir", default=".")
    args = p.parse_args()

    corpora = {
        "neardup": synthetic_data.neardup_corpus,
        "clustering": synthetic_data.clustering_corpus,
    }
    results = {
        case: run_case(case, *corpora[case](), args) for case in args.cases
    }

    # Shared axis limits so the per-case figures compare directly.
    acc_keys = ("lsh_precision", "lsh_recall", "scan_precision", "scan_recall")
    acc_min = min(min(c[k]) for c, _ in results.values() for k in acc_keys)
    t_max = max(
        max(max(c["lsh_time"]), max(c["scan_time"]))
        for c, _ in results.values()
    )
    acc_ylim = (max(0.0, acc_min - 0.03), 1.005)
    time_ylim = (0.0, t_max * 1.06)

    for case, (curves, params) in results.items():
        with open(f"{args.out_dir}/lsh_{case}_results.json", "w") as f:
            json.dump({"num_perms": args.num_perms, "params": params,
                       "curves": curves}, f, indent=1)
        plot_case(case, curves, params, args, acc_ylim, time_ylim)


if __name__ == "__main__":
    main()
