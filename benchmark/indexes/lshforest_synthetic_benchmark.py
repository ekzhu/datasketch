"""
Accuracy/speed benchmarks for MinHashLSHForest on two synthetic corpora.

Runs the same top-k experiment on the two corpus shapes from
``synthetic_data.py`` and writes one figure per shape:

- near-duplicate (``lshforest_neardup_benchmark.png``): bimodal
  similarities; each query has 1-3 planted near-duplicates (Jaccard
  0.6-0.95) among otherwise-disjoint sets.
- clustering (``lshforest_clustering_benchmark.png``):
  borderline-heavy; overlapping sets give every query a deep pool of
  moderately similar neighbours.

Each figure shows MAP@k and 90th percentile query time versus
``num_perm`` for ``forest.query`` -- one line per number of prefix
trees ``l``, each tree indexing prefixes of depth ``k = num_perm / l``
-- and for a linear scan that ranks all indexed sets by MinHash
estimate. Ground truth is the
exact-Jaccard top-k with ties on the similarity truncated to 2 decimals
(zero-similarity levels excluded; queries with no positive-similarity
neighbour are skipped). Retrieved candidates are re-ranked by exact
Jaccard before scoring, as in the original benchmark. Queries are held
out of the index. Linear-scan result sets are computed vectorized; its
query time is measured with the plain per-query loop on a fixed-seed
subsample.

Usage:
    python lshforest_synthetic_benchmark.py --out-dir figures/
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

from datasketch import MinHash, MinHashLSHForest  # noqa: E402

import synthetic_data  # noqa: E402

NUM_PERMS = [32, 64, 96, 128, 160, 192, 224, 256]


def apk(actual, predicted, k=10):
    if len(predicted) > k:
        predicted = predicted[:k]
    score, num_hits = 0.0, 0.0
    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    if not actual:
        return 0.0
    return score / min(len(actual), k)


def topk_with_ties(j_row: np.ndarray, k: int) -> List[int]:
    order = np.argsort(-j_row, kind="stable")
    result, curr_level, rank = [], None, 0
    for key in order:
        level = float(int(j_row[key] * 100)) / 100.0
        if level <= 0.0:
            break
        if level != curr_level:
            curr_level = level
            rank += 1
            if rank > k:
                break
        result.append(int(key))
    return result


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
    ground_truth = [topk_with_ties(row, args.k) for row in j_rows]
    eval_queries = [q for q, gt in enumerate(ground_truth) if gt]
    print(
        f"[{case}] queries with ground truth (top-{args.k}): "
        f"{len(eval_queries)}/{len(query_sets)}"
    )
    timing_sample = random.Random(7).sample(
        range(len(query_sets)), min(args.timing_queries, len(query_sets))
    )

    curves = {
        "scan_map": [], "scan_time": [],
        "forest_map": {l: [] for l in args.l_values},
        "forest_time": {l: [] for l in args.l_values},
    }

    for num_perm in args.num_perms:
        print(f"[{case}] num_perm={num_perm}: building sketches...")
        index_mh = build_minhashes(index_sets, num_perm, args.scheme)
        query_mh = build_minhashes(query_sets, num_perm, args.scheme)
        index_hv = np.stack([m.hashvalues for m in index_mh])
        query_hv = np.stack([m.hashvalues for m in query_mh])

        forest_maps, forest_t90s = {}, {}
        for l in args.l_values:
            forest = MinHashLSHForest(num_perm=num_perm, l=l)
            for key, m in enumerate(index_mh):
                forest.add(key, m)
            forest.index()
            forest_results, forest_times = [], []
            for q, m in enumerate(query_mh):
                t0 = time.perf_counter()
                res = forest.query(m, args.k)
                forest_times.append(time.perf_counter() - t0)
                forest_results.append(
                    sorted(res, key=lambda key: j_rows[q, key], reverse=True)
                )
            forest_maps[l] = np.mean(
                [apk(ground_truth[q], forest_results[q], args.k) for q in eval_queries]
            )
            forest_t90s[l] = np.percentile(forest_times, 90) * 1000

        est = estimate_matrix(index_hv, query_hv)
        order = np.argsort(-est, axis=1, kind="stable")[:, : args.k]
        scan_results = [
            sorted(order[q].tolist(), key=lambda key: j_rows[q, key], reverse=True)
            for q in range(len(query_sets))
        ]
        scan_times = []
        for q in timing_sample:
            m = query_mh[q]
            t0 = time.perf_counter()
            _ = sorted(
                range(len(index_mh)),
                key=lambda key: m.jaccard(index_mh[key]),
                reverse=True,
            )[: args.k]
            scan_times.append(time.perf_counter() - t0)

        for l in args.l_values:
            curves["forest_map"][l].append(forest_maps[l])
            curves["forest_time"][l].append(forest_t90s[l])
        curves["scan_map"].append(np.mean(
            [apk(ground_truth[q], scan_results[q], args.k) for q in eval_queries]
        ))
        curves["scan_time"].append(np.percentile(scan_times, 90) * 1000)
        maps = " ".join(
            f"l={l}:{forest_maps[l]:.3f}/{forest_t90s[l]:.2f}ms"
            for l in args.l_values
        )
        print(f"  forest {maps} | scan MAP={curves['scan_map'][-1]:.3f} "
              f"t90={curves['scan_time'][-1]:.1f}ms")

    return curves


def plot_case(case: str, curves, args, map_ylim, time_ylim):
    titles = {
        "neardup": "Near-duplicate corpus (bimodal similarities)",
        "clustering": "Clustering corpus (borderline-heavy similarities)",
    }
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), sharex=True)
    axes[0].plot(args.num_perms, curves["scan_map"], marker="+",
                 color="k", ls="--", label="Linearscan")
    axes[1].plot(args.num_perms, curves["scan_time"], marker="+",
                 color="k", ls="--", label="Linearscan")
    for i, l in enumerate(args.l_values):
        axes[0].plot(args.num_perms, curves["forest_map"][l], marker="+",
                     color=f"C{i}", label=f"LSH Forest l={l}")
        axes[1].plot(args.num_perms, curves["forest_time"][l], marker="+",
                     color=f"C{i}", label=f"LSH Forest l={l}")
    axes[0].set_ylabel(f"MAP (k = {args.k})")
    axes[1].set_ylabel("90 Percentile Query Time (ms)")
    axes[1].legend(loc="center right", fontsize=9)
    axes[0].set_ylim(*map_ylim)
    axes[1].set_ylim(*time_ylim)
    for ax in axes:
        ax.set_xticks(args.num_perms)
        ax.set_xlabel("num_perm  (each tree indexes prefixes of depth num_perm / l)")
        ax.grid()
    fig.suptitle(titles[case], y=1.0)
    fig.tight_layout()
    out = f"{args.out_dir}/lshforest_{case}_benchmark.png"
    fig.savefig(out, pad_inches=0.05, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("wrote", out)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--num-perms", type=int, nargs="+", default=NUM_PERMS)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--l-values", type=int, nargs="+", default=[2, 4, 8, 16])
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
    map_min = min(
        min(min(v) for v in c["forest_map"].values()) if k == "forest"
        else min(c["scan_map"])
        for c in results.values() for k in ("forest", "scan")
    )
    t_max = max(
        max(max(v) for v in c["forest_time"].values()) if k == "forest"
        else max(c["scan_time"])
        for c in results.values() for k in ("forest", "scan")
    )
    map_ylim = (max(0.0, map_min - 0.03), 1.005)
    time_ylim = (0.0, t_max * 1.06)

    for case, curves in results.items():
        dump = {
            "num_perms": args.num_perms,
            "l_values": args.l_values,
            "curves": {
                "scan_map": curves["scan_map"],
                "scan_time": curves["scan_time"],
                "forest_map": {str(l): v for l, v in curves["forest_map"].items()},
                "forest_time": {str(l): v for l, v in curves["forest_time"].items()},
            },
        }
        with open(f"{args.out_dir}/lshforest_{case}_results.json", "w") as f:
            json.dump(dump, f, indent=1)
        plot_case(case, curves, args, map_ylim, time_ylim)


if __name__ == "__main__":
    main()
