"""
Billion-scale permutation-scheme bias experiment (issue #212).

Two MinHash signatures sharing the same permutations are built from two
disjoint streams of ``n`` simulated hash values each: distinct 64-bit
element ids mapped through the MurmurHash3 finalizer (``fmix64``, a
bijection), truncated to 32 bits for the 32-bit schemes. This models an
ideal hash function applied to two disjoint sets, so the true Jaccard
similarity is 0 and an unbiased scheme's estimate equals the input-hash
saturation floor

    E(w, n) = p / (2 - p),   p = 1 - exp(-n / 2**w)

which is ~0.116 for w=32 at n=1e9 and ~0 for w=64 everywhere. Estimates
above the floor are bias added by the permutation step itself: the
pre-2.0.0 "legacy" scheme lands far above the floor, the affine schemes
on it.

The signature arithmetic mirrors ``MinHash.update_batch`` exactly but
streams vectorized chunks (CuPy when a CUDA device is available, NumPy
otherwise); ``--self-check`` verifies the mirror bit-for-bit against
``MinHash.update_batch``. Streams are deterministic: a cell is
identified by (scheme, n, trial) and reproduces exactly anywhere.

Usage
-----
    # verify the bulk kernel against the library first
    python minhash_scheme_bias_benchmark.py --self-check

    # small CPU run
    python minhash_scheme_bias_benchmark.py --sizes 1e6 --trials 4

    # full experiment (CUDA machine, ~1 GPU-hour on an L4)
    python minhash_scheme_bias_benchmark.py \
        --sizes 1e8 3.16e8 1e9 3.16e9 1e10 --trials 16 16 16 8 6 \
        --json results.json

    # render the docs figure from saved results
    python minhash_scheme_bias_benchmark.py \
        --plot-from results.json --out-fig scheme_bias.png
"""

from __future__ import annotations

import argparse
import json
import math
import time
from typing import Dict, List, Optional

import numpy as np

from datasketch.minhash import (
    MinHash,
    _fmix,
    _max_hash,
    _mersenne_prime,
)

try:
    import cupy as cp

    try:
        GPU_AVAILABLE = cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        GPU_AVAILABLE = False
except Exception:
    cp = None
    GPU_AVAILABLE = False

SCHEMES = ("affine32", "affine64", "legacy")
NUM_PERM = 128

# Element-id stride between streams; must exceed any n (2**35 > 1e10).
# Stream 2*trial feeds set A, stream 2*trial + 1 feeds set B, so the two
# sets of a trial are disjoint by construction. The same streams are
# reused across schemes and sizes, pairing the comparison: scheme
# differences within a trial cannot come from input noise.
_STREAM_STRIDE = 1 << 35


def stream_hashes(xp, stream: int, offset: int, count: int, width: int):
    """Simulated hash values for elements [offset, offset+count) of a stream.

    fmix64 maps the distinct element ids to uniform 64-bit hash values
    without collisions; the low 32 bits behave like an ideal 32-bit hash,
    including its birthday collisions (the saturation the floor accounts
    for).
    """
    base = xp.uint64(stream) * xp.uint64(_STREAM_STRIDE) + xp.uint64(offset)
    ids = xp.arange(count, dtype=xp.uint64) + base
    h64 = _fmix(ids, 64)
    if width == 64:
        return h64
    return h64.astype(xp.uint32)  # low 32 bits


def _chunk_mins(xp, hv, a, b, scheme: str, group: int = 16):
    """Per-permutation minima over one chunk, mirroring update_batch."""
    if scheme == "legacy":
        h = hv.astype(xp.uint64).reshape(-1, 1)
    else:
        h = _fmix(hv, 64 if scheme == "affine64" else 32).reshape(-1, 1)
    mins = []
    # Permutations are broadcast in groups to bound the (chunk, group)
    # intermediate arrays.
    for i in range(0, len(a), group):
        if scheme == "legacy":
            phv = (h * a[i : i + group] + b[i : i + group]) % _mersenne_prime
            phv = xp.bitwise_and(phv, _max_hash)
        else:
            phv = h * a[i : i + group] + b[i : i + group]
        mins.append(phv.min(axis=0))
    return xp.concatenate(mins)


def build_signature(
    stream: int,
    n: int,
    minhash: MinHash,
    use_gpu: Optional[bool] = None,
    chunk: int = 1 << 22,
) -> np.ndarray:
    """Signature of `minhash`'s permutations over `n` stream elements."""
    if use_gpu is None:
        use_gpu = GPU_AVAILABLE
    xp = cp if use_gpu else np
    scheme = minhash.scheme
    a = xp.asarray(minhash.permutations[0])
    b = xp.asarray(minhash.permutations[1])
    sig = xp.asarray(minhash.hashvalues)  # "empty" sentinels, scheme dtype
    width = 64 if scheme == "affine64" else 32
    for offset in range(0, n, chunk):
        hv = stream_hashes(xp, stream, offset, min(chunk, n - offset), width)
        sig = xp.minimum(sig, _chunk_mins(xp, hv, a, b, scheme))
    return sig if xp is np else cp.asnumpy(sig)


def run_trial(
    scheme: str,
    n: int,
    trial: int,
    num_perm: int = NUM_PERM,
    use_gpu: Optional[bool] = None,
    chunk: int = 1 << 22,
) -> Dict:
    """Estimated Jaccard of two disjoint n-element sets (true J = 0)."""
    t0 = time.perf_counter()
    sigs = []
    for side in (0, 1):
        m = MinHash(num_perm=num_perm, seed=trial, scheme=scheme)
        sigs.append(build_signature(2 * trial + side, n, m, use_gpu, chunk))
    ma, mb = (
        MinHash(num_perm=num_perm, seed=trial, scheme=scheme, hashvalues=s)
        for s in sigs
    )
    return {
        "scheme": scheme,
        "n": n,
        "trial": trial,
        "estimate": float(ma.jaccard(mb)),
        "seconds": time.perf_counter() - t0,
    }


def saturation_floor(width: int, n: int) -> float:
    """Expected estimate of an unbiased scheme: hash-image overlap of two
    disjoint n-element sets under an ideal width-bit hash."""
    p = -math.expm1(-n / 2.0**width)
    return p / (2.0 - p)


def self_check(use_gpu: Optional[bool] = None) -> None:
    """The bulk kernel must be bit-identical to MinHash.update_batch."""
    n, num_perm, stream = 100_000, 32, 3
    for scheme in SCHEMES:
        width = 64 if scheme == "affine64" else 32
        hv = stream_hashes(np, stream, 0, n, width)
        m = MinHash(
            num_perm=num_perm, seed=5, scheme=scheme, hashfunc=lambda x: x
        )
        m.update_batch(hv.tolist())
        ref = MinHash(num_perm=num_perm, seed=5, scheme=scheme)
        # small chunk also exercises the chunk-boundary reduction
        sig = build_signature(stream, n, ref, use_gpu, chunk=1 << 14)
        assert np.array_equal(m.hashvalues, sig), scheme
        assert m.hashvalues.dtype == sig.dtype, scheme
        print(f"self-check {scheme}: OK ({'GPU' if (GPU_AVAILABLE if use_gpu is None else use_gpu) else 'CPU'})")


def summarize(results: List[Dict]) -> List[Dict]:
    cells: Dict = {}
    for r in results:
        cells.setdefault((r["scheme"], r["n"]), []).append(r["estimate"])
    rows = []
    for (scheme, n), ests in sorted(cells.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        arr = np.asarray(ests)
        width = 64 if scheme == "affine64" else 32
        rows.append(
            {
                "scheme": scheme,
                "n": n,
                "trials": len(ests),
                "mean": float(arr.mean()),
                "ci95": float(1.96 * arr.std(ddof=1) / math.sqrt(len(arr)))
                if len(ests) > 1
                else 0.0,
                "floor": saturation_floor(width, n),
            }
        )
    return rows


def plot(results: List[Dict], out_fig: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = summarize(results)
    sizes = sorted({r["n"] for r in rows})
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    styles = {
        "legacy": dict(color="tab:red", marker="o"),
        "affine32": dict(color="tab:blue", marker="s"),
        "affine64": dict(color="tab:green", marker="^"),
    }
    for scheme in ("legacy", "affine32", "affine64"):
        pts = [r for r in rows if r["scheme"] == scheme]
        if not pts:
            continue
        ax.errorbar(
            [r["n"] for r in pts],
            [r["mean"] for r in pts],
            yerr=[r["ci95"] for r in pts],
            label=scheme,
            capsize=3,
            **styles[scheme],
        )
    xs = np.logspace(math.log10(min(sizes)), math.log10(max(sizes)), 200)
    ax.plot(
        xs,
        [saturation_floor(32, x) for x in xs],
        "k--",
        lw=1,
        label="32-bit hash saturation floor",
    )
    ax.axhline(0.0, color="grey", lw=0.8, alpha=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("Unique elements per set")
    ax.set_ylabel("Estimated Jaccard similarity")
    ax.set_title(
        f"Two disjoint sets (true Jaccard = 0), num_perm={NUM_PERM}"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_fig, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--sizes", type=float, nargs="+", default=[1e6])
    p.add_argument(
        "--trials",
        type=int,
        nargs="+",
        default=[4],
        help="trial count per size (single value broadcasts)",
    )
    p.add_argument("--schemes", nargs="+", default=list(SCHEMES))
    p.add_argument("--num-perm", type=int, default=NUM_PERM)
    p.add_argument("--chunk-log2", type=int, default=22)
    p.add_argument("--self-check", action="store_true")
    p.add_argument("--json", help="write per-trial results to this file")
    p.add_argument("--plot-from", help="render a figure from a results JSON")
    p.add_argument("--out-fig", default="scheme_bias.png")
    args = p.parse_args()

    if args.plot_from:
        with open(args.plot_from) as f:
            plot(json.load(f), args.out_fig)
        print("wrote", args.out_fig)
        return

    self_check()
    if args.self_check:
        return

    trials = args.trials
    if len(trials) == 1:
        trials = trials * len(args.sizes)
    if len(trials) != len(args.sizes):
        p.error("--trials must match --sizes (or be a single value)")

    results = []
    print("scheme,n,trial,estimate,seconds")
    for n, t in zip(args.sizes, trials):
        for scheme in args.schemes:
            for trial in range(t):
                r = run_trial(
                    scheme,
                    int(n),
                    trial,
                    num_perm=args.num_perm,
                    chunk=1 << args.chunk_log2,
                )
                results.append(r)
                print(
                    f"{r['scheme']},{r['n']},{r['trial']},"
                    f"{r['estimate']:.6f},{r['seconds']:.2f}"
                )

    for row in summarize(results):
        print(
            f"{row['scheme']:9s} n={row['n']:.2e} trials={row['trials']:3d} "
            f"mean={row['mean']:.4f} +/-{row['ci95']:.4f} "
            f"floor={row['floor']:.4f}"
        )
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f)
        print("wrote", args.json)


if __name__ == "__main__":
    main()
