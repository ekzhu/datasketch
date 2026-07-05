"""
Characteristics of MinHashLSH retrieval vs. true Jaccard similarity.

Complements ``lsh_20newsgroups_benchmark.py`` (aggregate precision/recall
on a real corpus) by resolving retrieval *by the true Jaccard similarity
of a neighbour*: the probability that a set at similarity ``s`` to the
query is returned by ``lsh.query``. For a banding index with ``b`` bands
of ``r`` rows this probability is, in theory,

    P(s) = 1 - (1 - s**r) ** b

and the constructor's ``threshold``/``weights`` optimizer chooses
``(b, r)`` to place this S-curve: measured curves are plotted against the
theory (dashed) for the ``(b, r)`` actually chosen.

Design (planted neighbours at controlled similarity)
----------------------------------------------------
Same as ``lshforest_similarity_benchmark.py``: for each of ``--queries``
query sets (size ``--set-size``) one neighbour is planted at each target
similarity level -- a neighbour at Jaccard ``s`` shares
``i = round(2ns / (1 + s))`` tokens with the query and takes the rest
from a disjoint id range, so the realised similarity ``i / (2n - i)`` is
exact. ``--background`` random sets are indexed as distractors; queries
are not indexed. The retrieval probability at a level is the fraction of
queries whose planted neighbour appears in ``lsh.query``'s result.

Three panels
------------
- threshold sweep (fixed num_perm, default weights): the optimizer moves
  the S-curve's transition to the requested threshold.
- num_perm sweep (fixed threshold): more permutations allow finer
  ``(b, r)`` choices -- a sharper transition (fewer false
  positives/negatives near the threshold).
- weights sweep (fixed threshold and num_perm): ``weights =
  (false_positive_weight, false_negative_weight)`` trades the two error
  areas -- penalizing false positives shifts the curve right of the
  threshold, penalizing false negatives shifts it left.

Usage:
    python lsh_similarity_benchmark.py --out-dir figures/
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from datasketch import MinHash, MinHashLSH  # noqa: E402

# Ids for the planted neighbours' fresh tokens come from here and up, a
# range disjoint from the [0, population) base tokens.
_FRESH_BASE = 1 << 34


def shared_count(s: float, n: int) -> int:
    """Tokens a size-n neighbour must share with a size-n query to have
    Jaccard similarity s: i = 2ns / (1 + s)."""
    return int(round(2 * n * s / (1.0 + s)))


def realised_jaccard(levels: np.ndarray, n: int) -> np.ndarray:
    i = np.array([shared_count(s, n) for s in levels], dtype=np.float64)
    return i / (2 * n - i)


def build_corpus(rng: np.random.RandomState, args, levels: np.ndarray):
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


def hash_corpus(query_sets, neighbour_sets, background, num_perm, scheme):
    neighbour_mh = [
        [to_minhash(t, num_perm, scheme) for t in neighbours]
        for neighbours in neighbour_sets
    ]
    background_mh = [to_minhash(t, num_perm, scheme) for t in background]
    query_mh = [to_minhash(t, num_perm, scheme) for t in query_sets]
    return query_mh, neighbour_mh, background_mh


def measure(lsh_factory, query_mh, neighbour_mh, background_mh, num_levels):
    """Build an index with `lsh_factory`, insert everything, and return
    (retrieval probability per level, (b, r) of the index)."""
    lsh = lsh_factory()
    for q, neighbours in enumerate(neighbour_mh):
        for li, m in enumerate(neighbours):
            lsh.insert(("n", q, li), m)
    for j, m in enumerate(background_mh):
        lsh.insert(("b", j), m)
    hits = np.zeros(num_levels, dtype=np.float64)
    for q, qmh in enumerate(query_mh):
        results = set(lsh.query(qmh))
        for li in range(num_levels):
            if ("n", q, li) in results:
                hits[li] += 1.0
    return hits / len(query_mh), (lsh.b, lsh.r)


def theory(b: int, r: int, s: np.ndarray) -> np.ndarray:
    return 1.0 - (1.0 - s ** float(r)) ** float(b)


def run(args):
    levels = np.round(np.arange(args.min_sim, args.max_sim + 1e-9, args.step), 3)
    j = realised_jaccard(levels, args.set_size)
    rng = np.random.RandomState(args.seed)

    print(
        f"planting {args.queries} queries x {len(levels)} levels "
        f"+ {args.background} background sets (set size {args.set_size})..."
    )
    query_sets, neighbour_sets, background = build_corpus(rng, args, levels)

    hashed: Dict[int, Tuple] = {}
    for num_perm in sorted({args.num_perm, *args.num_perm_sweep}):
        print(f"hashing MinHashes (num_perm={num_perm})...")
        hashed[num_perm] = hash_corpus(
            query_sets, neighbour_sets, background, num_perm, args.scheme
        )

    def run_config(threshold, num_perm, weights):
        qmh, nmh, bmh = hashed[num_perm]
        prob, br = measure(
            lambda: MinHashLSH(threshold=threshold, num_perm=num_perm, weights=weights),
            qmh, nmh, bmh, len(levels),
        )
        print(
            f"  threshold={threshold} num_perm={num_perm} weights={weights}"
            f" -> (b, r) = {br}"
        )
        return prob, br

    print("panel A: threshold sweep ...")
    panel_a = {t: run_config(t, args.num_perm, (0.5, 0.5)) for t in args.thresholds}
    print("panel B: num_perm sweep ...")
    panel_b = {
        np_: (panel_a[args.threshold] if np_ == args.num_perm
              else run_config(args.threshold, np_, (0.5, 0.5)))
        for np_ in args.num_perm_sweep
    }
    print("panel C: weights sweep ...")
    panel_c = {}
    for w in args.fp_weights:
        weights = (w, round(1.0 - w, 3))
        panel_c[weights] = (
            panel_a[args.threshold] if weights == (0.5, 0.5)
            else run_config(args.threshold, args.num_perm, weights)
        )

    plot(args, j, panel_a, panel_b, panel_c)


def plot(args, j, panel_a, panel_b, panel_c):
    s_dense = np.linspace(0.01, 0.99, 200)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), sharey=True)

    def draw(ax, curves, labeler):
        for i, (key, (prob, (b, r))) in enumerate(curves.items()):
            color = f"C{i}"
            ax.plot(j, prob, marker=".", color=color, label=labeler(key, b, r))
            ax.plot(s_dense, theory(b, r, s_dense), "--", color=color, alpha=0.55, lw=1)
        ax.set_xlabel("True Jaccard similarity of neighbour")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=9)

    ax = axes[0]
    draw(ax, panel_a, lambda t, b, r: f"threshold={t} (b={b}, r={r})")
    for t in args.thresholds:
        ax.axvline(t, color="grey", lw=0.6, alpha=0.4)
    ax.set_title(f"Effect of threshold  (num_perm={args.num_perm})")
    ax.set_ylabel("P(returned by query)")

    ax = axes[1]
    draw(ax, panel_b, lambda np_, b, r: f"num_perm={np_} (b={b}, r={r})")
    ax.axvline(args.threshold, color="grey", lw=0.6, alpha=0.4)
    ax.set_title(f"Effect of num_perm  (threshold={args.threshold})")

    ax = axes[2]
    draw(ax, panel_c, lambda w, b, r: f"weights={w} (b={b}, r={r})")
    ax.axvline(args.threshold, color="grey", lw=0.6, alpha=0.4)
    ax.set_title(
        f"Effect of weights (fp, fn)  "
        f"(threshold={args.threshold}, num_perm={args.num_perm})"
    )

    fig.tight_layout()
    out = f"{args.out_dir}/lsh_characteristics.png"
    fig.savefig(out, pad_inches=0.05, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("wrote", out)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--thresholds", type=float, nargs="+",
                   default=[0.3, 0.5, 0.7, 0.9], help="panel A")
    p.add_argument("--num-perm-sweep", type=int, nargs="+",
                   default=[32, 128, 512], help="panel B")
    p.add_argument("--fp-weights", type=float, nargs="+",
                   default=[0.9, 0.5, 0.1],
                   help="panel C false-positive weights; fn weight = 1 - fp")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="fixed threshold for panels B and C")
    p.add_argument("--num-perm", type=int, default=128,
                   help="fixed num_perm for panels A and C")
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
