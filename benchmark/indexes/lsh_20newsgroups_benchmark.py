"""
Accuracy benchmark for MinHashLSH on 20 newsgroups.

Recreates the experiment behind ``docs/_static/lsh_benchmark.png``
(originally produced by a pre-repository script; see commit 5513004):

- Corpus: scikit-learn 20 newsgroups, headers/footers/quotes removed,
  word 3-shingles, empty documents dropped. The train split is indexed,
  the test split queries.
- MinHashLSH: threshold 0.5. Reports average precision/recall of
  ``lsh.query`` against exact-Jaccard ground truth (>= 0.5), versus a
  linear scan that thresholds the MinHash estimate. Queries whose ground
  truth is empty are excluded from the averages, as in the original.
- Query times are 90th-percentile wall-clock of the real per-query API
  calls. Linear-scan result sets are computed vectorized (identical
  output, much faster); linear-scan times come from timing the plain
  per-query loop on a fixed-seed subsample of queries.

The companion top-k experiment for MinHashLSHForest lives in
``lshforest_synthetic_benchmark.py``.

Usage:
    python lsh_20newsgroups_benchmark.py --out-dir figures/
"""

from __future__ import annotations

import argparse
import random
import time
from typing import Dict, List, Set

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.sparse import csr_matrix  # noqa: E402

from datasketch import MinHash, MinHashLSH  # noqa: E402

NUM_PERMS = [32, 64, 96, 128, 160, 192, 224, 256]


def get_ngrams(text: str, n: int = 3) -> List[str]:
    words = text.split()
    if len(words) < n:
        return [" ".join(words)] if words else []
    return [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]


def load_corpus():
    from sklearn.datasets import fetch_20newsgroups

    sets = {}
    for subset in ("train", "test"):
        news = fetch_20newsgroups(
            subset=subset, remove=("headers", "footers", "quotes")
        )
        shingled = [set(get_ngrams(text)) for text in news.data]
        sets[subset] = [s for s in shingled if s]
    return sets["train"], sets["test"]


def exact_jaccard_matrix(train: List[Set[str]], test: List[Set[str]]):
    """Dense exact-Jaccard rows (test x train), computed chunk-wise from a
    sparse one-hot shingle matrix."""
    vocab: Dict[str, int] = {}
    for s in train:
        for w in s:
            vocab.setdefault(w, len(vocab))

    def one_hot(sets_):
        indptr, indices = [0], []
        for s in sets_:
            cols = [vocab[w] for w in s if w in vocab]
            indices.extend(cols)
            indptr.append(len(indices))
        return csr_matrix(
            (np.ones(len(indices), dtype=np.float32), indices, indptr),
            shape=(len(sets_), len(vocab)),
        )

    a_train = one_hot(train).T.tocsc()
    a_test = one_hot(test)
    sizes_train = np.array([len(s) for s in train], dtype=np.float32)
    sizes_test = np.array([len(s) for s in test], dtype=np.float32)
    for start in range(0, len(test), 512):
        chunk = a_test[start : start + 512]
        inter = np.asarray((chunk @ a_train).todense(), dtype=np.float32)
        union = sizes_test[start : start + 512, None] + sizes_train[None, :] - inter
        yield start, inter / np.maximum(union, 1.0)


# --- metrics ported from the original benchmark scripts ---------------


def get_precision_recall(found, reference):
    reference = set(reference)
    intersect = sum(1 for i in found if i in reference)
    precision = float(intersect) / float(len(found)) if found else 0.0
    recall = float(intersect) / float(len(reference)) if reference else 1.0
    if len(found) == len(reference) == 0:
        precision = recall = 1.0
    return precision, recall


# --- benchmark ---------------------------------------------------------


def build_minhashes(sets_, num_perm, scheme):
    ms = []
    for s in sets_:
        m = MinHash(num_perm=num_perm, scheme=scheme)
        m.update_batch([w.encode("utf-8") for w in s])
        ms.append(m)
    return ms


def estimate_matrix(train_hv: np.ndarray, test_hv: np.ndarray):
    """MinHash Jaccard estimates (test x train), chunked."""
    est = np.empty((len(test_hv), len(train_hv)), dtype=np.float32)
    for start in range(0, len(test_hv), 128):
        chunk = test_hv[start : start + 128]
        est[start : start + 128] = (
            chunk[:, None, :] == train_hv[None, :, :]
        ).mean(axis=2, dtype=np.float32)
    return est


def run(args):
    print("loading 20 newsgroups...")
    train, test = load_corpus()
    print(
        f"index={len(train)} docs, queries={len(test)} docs, "
        f"avg cardinality={np.mean([len(s) for s in train]):.0f} shingles"
    )

    print("computing exact Jaccard ground truth...")
    lsh_gt: List[Set[int]] = [set() for _ in test]
    for start, j_chunk in exact_jaccard_matrix(train, test):
        for i, row in enumerate(j_chunk):
            lsh_gt[start + i] = set(
                np.flatnonzero(row >= args.threshold).tolist()
            )

    lsh_eval = [q for q in range(len(test)) if lsh_gt[q]]
    print(f"queries with ground truth (J>={args.threshold}): {len(lsh_eval)}")
    timing_queries = random.Random(42).sample(
        range(len(test)), min(args.timing_queries, len(test))
    )

    curves: Dict[str, List[float]] = {
        name: []
        for name in (
            "lsh_precision", "lsh_recall", "lsh_time",
            "scan_precision", "scan_recall", "scan_time",
        )
    }

    for num_perm in args.num_perms:
        print(f"num_perm={num_perm}: building sketches...")
        train_mh = build_minhashes(train, num_perm, args.scheme)
        test_mh = build_minhashes(test, num_perm, args.scheme)
        train_hv = np.stack([m.hashvalues for m in train_mh])
        test_hv = np.stack([m.hashvalues for m in test_mh])

        # MinHashLSH: real queries (results + times in one pass).
        lsh = MinHashLSH(threshold=args.threshold, num_perm=num_perm)
        for key, m in enumerate(train_mh):
            lsh.insert(key, m)
        lsh_results, lsh_times = [], []
        for m in test_mh:
            t0 = time.perf_counter()
            res = lsh.query(m)
            lsh_times.append(time.perf_counter() - t0)
            lsh_results.append(res)

        # Linear scan, thresholded estimate: vectorized results.
        est = estimate_matrix(train_hv, test_hv)
        scan_results = [
            np.flatnonzero(row >= args.threshold).tolist() for row in est
        ]
        # ...and timing of the plain per-query loop on a subsample.
        scan_times = []
        for q in timing_queries:
            m = test_mh[q]
            t0 = time.perf_counter()
            _ = [key for key, mt in enumerate(train_mh) if m.jaccard(mt) >= args.threshold]
            scan_times.append(time.perf_counter() - t0)

        pr = np.mean(
            [get_precision_recall(lsh_results[q], lsh_gt[q]) for q in lsh_eval],
            axis=0,
        )
        curves["lsh_precision"].append(pr[0])
        curves["lsh_recall"].append(pr[1])
        pr = np.mean(
            [get_precision_recall(scan_results[q], lsh_gt[q]) for q in lsh_eval],
            axis=0,
        )
        curves["scan_precision"].append(pr[0])
        curves["scan_recall"].append(pr[1])
        curves["lsh_time"].append(np.percentile(lsh_times, 90) * 1000)
        curves["scan_time"].append(np.percentile(scan_times, 90) * 1000)

        print(
            f"  lsh P={curves['lsh_precision'][-1]:.3f} R={curves['lsh_recall'][-1]:.3f} "
            f"scan P={curves['scan_precision'][-1]:.3f} R={curves['scan_recall'][-1]:.3f}"
        )

    plot_lsh(args, curves)


def plot_lsh(args, curves):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True)
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
    for ax in axes:
        ax.set_xlabel("# of Permutation Functions")
        ax.grid()
    out = f"{args.out_dir}/lsh_benchmark.png"
    fig.savefig(out, pad_inches=0.05, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("wrote", out)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--num-perms", type=int, nargs="+", default=NUM_PERMS)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--scheme", default="affine32",
                   choices=["affine32", "affine64", "legacy"])
    p.add_argument("--timing-queries", type=int, default=400)
    p.add_argument("--out-dir", default=".")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
