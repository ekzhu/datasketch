"""Shared synthetic corpora for the LSH / LSH Forest benchmarks.

Two corpus shapes with opposite similarity distributions around a
threshold, so index behaviour can be shown at both extremes:

- ``neardup_corpus``: bimodal, the near-duplicate detection shape. Each
  query has 1..max_dups planted near-duplicates (Jaccard in
  ``dup_range``); every other pair is essentially disjoint (tokens
  drawn from a large id space). Nothing sits near a mid threshold.
- ``clustering_corpus``: borderline-heavy, the clustering shape. Sets
  drawn from a small vocabulary with sizes up to the vocabulary size
  overlap heavily, so pair similarities spread across the whole range
  and many sit near a mid threshold.

Queries are held out (not indexed) in both corpora. All tokens are
ints; hash them as fixed-width bytes.
"""

from __future__ import annotations

import random
from typing import List, Set, Tuple

import numpy as np
from scipy.sparse import csr_matrix

# Fresh ids for planted near-duplicates' non-shared tokens come from
# here and up, disjoint from any corpus id space.
_FRESH_BASE = 1 << 34


def shared_count(s: float, n: int) -> int:
    """Tokens a size-n set must share with a size-n set to have Jaccard
    similarity s: i = 2ns / (1 + s)."""
    return int(round(2 * n * s / (1.0 + s)))


def neardup_corpus(
    n_background: int = 5000,
    n_queries: int = 500,
    set_size: int = 200,
    dup_range: Tuple[float, float] = (0.6, 0.95),
    max_dups: int = 3,
    population: int = 5_000_000,
    seed: int = 42,
) -> Tuple[List[Set[int]], List[Set[int]]]:
    rng = np.random.RandomState(seed)
    fresh = _FRESH_BASE
    index_sets: List[Set[int]] = [
        set(rng.randint(0, population, size=set_size).tolist())
        for _ in range(n_background)
    ]
    query_sets: List[Set[int]] = []
    for _ in range(n_queries):
        q = rng.randint(0, population, size=set_size)
        query_sets.append(set(q.tolist()))
        for _ in range(rng.randint(1, max_dups + 1)):
            s = rng.uniform(*dup_range)
            i = shared_count(s, set_size)
            shared = rng.choice(q, size=i, replace=False)
            n_fresh = set_size - i
            dup = set(shared.tolist()) | set(range(fresh, fresh + n_fresh))
            fresh += n_fresh
            index_sets.append(dup)
    return index_sets, query_sets


def clustering_corpus(
    n_sets: int = 10000,
    population: int = 5000,
    min_size: int = 10,
    max_size: int = 5000,
    query_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Set[int]], List[Set[int]]]:
    rnd = random.Random(seed)
    sizes = np.random.RandomState(seed).randint(min_size, max_size, size=n_sets)
    tokens = list(range(population))
    sets = [set(rnd.sample(tokens, int(size))) for size in sizes]
    query_indices = set(rnd.sample(range(n_sets), int(n_sets * query_ratio)))
    queries = [sets[i] for i in sorted(query_indices)]
    index = [s for i, s in enumerate(sets) if i not in query_indices]
    return index, queries


def containment_corpus(
    shape: str = "near_containment",
    n_background: int = 5000,
    n_queries: int = 500,
    query_size: int = 200,
    index_size_range: Tuple[int, int] = (100, 10_000),
    planted_range: Tuple[float, float] = (0.85, 1.0),
    max_planted: int = 3,
    spread_planted: int = 12,
    population: int = 5_000_000,
    size_dist: str = "loguniform",
    zipf_shape: float = 1.5,
    seed: int = 42,
) -> Tuple[List[Set[int]], List[Set[int]]]:
    """Corpora for containment (domain) search: index sets are drawn from
    ``index_size_range`` and every query is a fixed ``query_size`` tokens,
    smaller than the typical index set -- the domain-search regime, where
    containment differs from Jaccard. A planted neighbour at containment
    ``c`` shares ``round(c * query_size)`` of the query's tokens and takes
    the rest from a disjoint id range, so its containment is (nearly)
    exact by construction and independent of its own size.
    ``shape="near_containment"`` plants 1..max_planted neighbours at very
    high containment (bimodal: everything is either nearly contained or
    unrelated -- the domain-search shape); ``shape="spread"`` plants
    ``spread_planted`` neighbours at containments uniform in (0.05, 0.95)
    (borderline-heavy).

    ``size_dist`` selects the index-set cardinality distribution over
    ``index_size_range``: ``"loguniform"``; ``"uniform"`` (linear, use a
    narrow range for a nearly-constant cardinality); or ``"zipf"`` -- a
    truncated Pareto with tail exponent ``zipf_shape``, the power-law
    shape reported for real open-data corpora in the LSH Ensemble paper
    (most sets small, a heavy tail of very large ones). Partitioning
    helps in proportion to this skew.
    """
    rng = np.random.RandomState(seed)
    fresh = _FRESH_BASE

    def draw_size(lo: int, hi: int) -> int:
        """Index-set size from the selected distribution."""
        if size_dist == "zipf":
            return min(int(lo * (1.0 + rng.pareto(zipf_shape))), hi)
        if size_dist == "loguniform":
            return int(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        if size_dist == "uniform":
            return int(rng.uniform(lo, hi))
        raise ValueError(size_dist)

    index_sets: List[Set[int]] = [
        set(rng.randint(0, population, size=draw_size(*index_size_range)).tolist())
        for _ in range(n_background)
    ]
    query_sets: List[Set[int]] = []
    for _ in range(n_queries):
        q = rng.randint(0, population, size=query_size)
        query_sets.append(set(q.tolist()))
        if shape == "near_containment":
            cs = rng.uniform(*planted_range, size=rng.randint(1, max_planted + 1))
        elif shape == "spread":
            cs = rng.uniform(0.05, 0.95, size=spread_planted)
        else:
            raise ValueError(shape)
        for c in cs:
            i = int(round(c * len(q)))
            shared = rng.choice(q, size=i, replace=False)
            size = max(draw_size(*index_size_range), i)
            n_fresh = size - i
            x = set(shared.tolist()) | set(range(fresh, fresh + n_fresh))
            fresh += n_fresh
            index_sets.append(x)
    return index_sets, query_sets


def _intersections(index_sets, query_sets, chunk):
    """Yield (start, intersection-count matrix chunk) via sparse one-hot."""
    vocab = {}
    for s in index_sets:
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

    a_index = one_hot(index_sets).T.tocsc()
    a_query = one_hot(query_sets)
    for start in range(0, len(query_sets), chunk):
        q = a_query[start : start + chunk]
        yield start, np.asarray((q @ a_index).todense(), dtype=np.float32)


def exact_jaccard(
    index_sets: List[Set[int]], query_sets: List[Set[int]], chunk: int = 256
) -> np.ndarray:
    """Exact Jaccard matrix (queries x index sets)."""
    sizes_index = np.array([len(s) for s in index_sets], dtype=np.float32)
    sizes_query = np.array([len(s) for s in query_sets], dtype=np.float32)
    out = np.empty((len(query_sets), len(index_sets)), dtype=np.float32)
    for start, inter in _intersections(index_sets, query_sets, chunk):
        union = (
            sizes_query[start : start + chunk, None]
            + sizes_index[None, :]
            - inter
        )
        out[start : start + chunk] = inter / np.maximum(union, 1.0)
    return out


def exact_containment(
    index_sets: List[Set[int]], query_sets: List[Set[int]], chunk: int = 256
) -> np.ndarray:
    """Exact containment |Q n X| / |Q| matrix (queries x index sets)."""
    sizes_query = np.array([len(s) for s in query_sets], dtype=np.float32)
    out = np.empty((len(query_sets), len(index_sets)), dtype=np.float32)
    for start, inter in _intersections(index_sets, query_sets, chunk):
        out[start : start + chunk] = (
            inter / sizes_query[start : start + chunk, None]
        )
    return out
