"""
Characteristics of MinHashLSHForest top-k retrieval vs. true Jaccard.

Where ``lshforest_synthetic_benchmark.py`` reports one aggregate MAP per
``num_perm``, this benchmark resolves the forest's behaviour *by the true
Jaccard similarity of a neighbour*, producing the characteristic LSH
retrieval curve: the probability that a set at true similarity ``s`` to
the query is returned by ``forest.query(query, k)``.

Design (planted neighbours at controlled similarity)
----------------------------------------------------
For each of ``--queries`` query sets (size ``--set-size``, tokens drawn
from ``[0, population)``) we plant exactly one neighbour at each target
similarity level. A neighbour at Jaccard ``s`` to a query ``A`` of size
``n`` shares ``i = round(2ns / (1 + s))`` of ``A``'s tokens and takes its
remaining ``n - i`` tokens from a disjoint id range, so both sets have
size ``n``, ``|A n B| = i`` and ``|A u B| = 2n - i`` exactly -- the
realised Jaccard ``i / (2n - i)`` is used for the x-axis. ``--background``
random size-``n`` sets are indexed as distractors. Queries are not indexed.

The retrieval probability at level ``s`` is the fraction of queries whose
planted-``s`` neighbour appears in the returned set. The forest returns
the ``k`` indexed sets with the longest shared hash prefix across its
``l`` trees (i.e. the highest *estimated* similarity), so a neighbour is
returned roughly when its similarity ranks in the top ``k`` -- a soft step
whose location and height are set by the two tunable knobs below.

Two panels (both at fixed ``num_perm``)
---------------------------------------
- l sweep (fixed query k): each tree indexes a prefix of length
  ``num_perm / l``. More, shorter trees (larger ``l``) make more sets
  collide, lifting recall of weaker neighbours (curve shifts left) at the
  cost of more candidates to sift; fewer, deeper trees are precise but
  only surface very similar sets (curve shifts right).
- k sweep (fixed l): larger query ``k`` gathers more candidates, so the
  curve shifts to lower similarity. This is the knob behind the
  "over-retrieve then re-rank" accuracy tip in ``MinHashLSHForest.query``.

``num_perm`` itself (swept with ``--experiment num_perm``) mostly sets the
estimation resolution and, on this retrieval metric, has a much weaker
effect than ``l`` or ``k`` -- included for completeness.

Usage:
    python lshforest_similarity_benchmark.py --out-dir figures/
"""

from __future__ import annotations

import argparse
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from datasketch import MinHash, MinHashLSHForest  # noqa: E402

# Ids for the planted neighbours' fresh tokens come from here and up, a
# range disjoint from the [0, population) base tokens, so a neighbour's
# non-shared part collides with nothing.
_FRESH_BASE = 1 << 34


def shared_count(s: float, n: int) -> int:
    """Tokens a size-n neighbour must share with a size-n query to have
    Jaccard similarity s: i = 2ns / (1 + s)."""
    return int(round(2 * n * s / (1.0 + s)))


def realised_jaccard(levels: np.ndarray, n: int) -> np.ndarray:
    i = np.array([shared_count(s, n) for s in levels], dtype=np.float64)
    return i / (2 * n - i)


def build_corpus(rng: np.random.RandomState, args, levels: np.ndarray):
    """Return (query_sets, neighbour_sets, background_sets).

    neighbour_sets[q] is a list aligned with `levels`: the planted
    neighbour of query q at each similarity level.
    """
    n, pop = args.set_size, args.population
    counts = [shared_count(s, n) for s in levels]
    fresh = _FRESH_BASE

    query_sets, neighbour_sets = [], []
    for _ in range(args.queries):
        a = rng.randint(0, pop, size=n)
        query_sets.append(a)
        neighbours = []
        for i in counts:
            shared = rng.choice(a, size=i, replace=False) if i > 0 else np.empty(0, np.int64)
            n_fresh = n - i
            fresh_tokens = np.arange(fresh, fresh + n_fresh)
            fresh += n_fresh
            neighbours.append(np.concatenate([shared, fresh_tokens]))
        neighbour_sets.append(neighbours)

    background = [rng.randint(0, pop, size=n) for _ in range(args.background)]
    return query_sets, neighbour_sets, background


def to_minhash(tokens: np.ndarray, num_perm: int, scheme: str) -> MinHash:
    m = MinHash(num_perm=num_perm, scheme=scheme)
    m.update_batch([int(t).to_bytes(8, "little") for t in tokens])
    return m


def build_forest(neighbour_mh, background_mh, l):
    """A forest with the given l, indexing precomputed MinHashes. Neighbour
    keys are ("n", query_index, level_index); background keys ("b", j)."""
    num_perm = len(neighbour_mh[0][0])
    forest = MinHashLSHForest(num_perm=num_perm, l=l)
    for q, neighbours in enumerate(neighbour_mh):
        for li, m in enumerate(neighbours):
            forest.add(("n", q, li), m)
    for j, m in enumerate(background_mh):
        forest.add(("b", j), m)
    forest.index()
    return forest


def retrieval_probability(forest, query_mh, num_levels, k) -> np.ndarray:
    """Fraction of queries whose planted neighbour at each level is
    returned by forest.query(query, k)."""
    hits = np.zeros(num_levels, dtype=np.float64)
    for q, qmh in enumerate(query_mh):
        results = set(forest.query(qmh, k))
        for li in range(num_levels):
            if ("n", q, li) in results:
                hits[li] += 1.0
    return hits / len(query_mh)


def run(args):
    levels = np.round(np.arange(args.min_sim, args.max_sim + 1e-9, args.step), 3)
    j = realised_jaccard(levels, args.set_size)
    rng = np.random.RandomState(args.seed)

    print(
        f"planting {args.queries} queries x {len(levels)} levels "
        f"+ {args.background} background sets (set size {args.set_size})..."
    )
    query_sets, neighbour_sets, background = build_corpus(rng, args, levels)

    print(f"hashing MinHashes (num_perm={args.num_perm})...")
    neighbour_mh = [
        [to_minhash(t, args.num_perm, args.scheme) for t in neighbours]
        for neighbours in neighbour_sets
    ]
    background_mh = [to_minhash(t, args.num_perm, args.scheme) for t in background]
    query_mh = [to_minhash(t, args.num_perm, args.scheme) for t in query_sets]

    # Panel A: l sweep at fixed query k. A forest per l, reusing MinHashes.
    print(f"panel A: l sweep at k={args.k}, num_perm={args.num_perm} ...")
    curves_l: Dict[int, np.ndarray] = {}
    for l in args.l_sweep:
        forest = build_forest(neighbour_mh, background_mh, l)
        curves_l[l] = retrieval_probability(forest, query_mh, len(levels), args.k)
        print(f"  l={l} (depth {args.num_perm // l}): {np.round(curves_l[l], 2)}")

    # Panel B: k sweep at fixed l (reuse one index).
    print(f"panel B: k sweep at l={args.l}, num_perm={args.num_perm} ...")
    forest = build_forest(neighbour_mh, background_mh, args.l)
    curves_k: Dict[int, np.ndarray] = {}
    for k in args.k_sweep:
        curves_k[k] = retrieval_probability(forest, query_mh, len(levels), k)
        print(f"  k={k}: {np.round(curves_k[k], 2)}")

    plot(args, j, curves_l, curves_k)


def plot(args, j, curves_l, curves_k):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)

    ax = axes[0]
    for l in args.l_sweep:
        ax.plot(j, curves_l[l], marker=".", label=f"l={l} (depth {args.num_perm // l})")
    ax.set_title(f"Effect of l  (num_perm={args.num_perm}, k={args.k})")
    ax.set_xlabel("True Jaccard similarity of neighbour")
    ax.set_ylabel(f"P(returned in top-{args.k})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)

    ax = axes[1]
    for k in args.k_sweep:
        ax.plot(j, curves_k[k], marker=".", label=f"k={k}")
    ax.set_title(f"Effect of query k  (num_perm={args.num_perm}, l={args.l})")
    ax.set_xlabel("True Jaccard similarity of neighbour")
    ax.set_ylabel("P(returned)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)

    fig.tight_layout()
    out = f"{args.out_dir}/lshforest_characteristics.png"
    fig.savefig(out, pad_inches=0.05, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("wrote", out)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--num-perm", type=int, default=128,
                   help="fixed num_perm for both panels")
    p.add_argument("--l-sweep", type=int, nargs="+", default=[2, 4, 8, 32],
                   help="l values for panel A")
    p.add_argument("--k", type=int, default=10,
                   help="query k for the l sweep (panel A)")
    p.add_argument("--l", type=int, default=8,
                   help="fixed l for the k sweep (panel B)")
    p.add_argument("--k-sweep", type=int, nargs="+", default=[5, 10, 20],
                   help="query k values for panel B")
    p.add_argument("--queries", type=int, default=400)
    p.add_argument("--background", type=int, default=2000)
    p.add_argument("--set-size", type=int, default=200)
    p.add_argument("--population", type=int, default=5_000_000)
    p.add_argument("--min-sim", type=float, default=0.05)
    p.add_argument("--max-sim", type=float, default=0.95)
    p.add_argument("--step", type=float, default=0.05)
    p.add_argument("--scheme", default="affine32",
                   choices=["affine32", "affine64", "legacy"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default=".")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
