"""
Benchmark MinHash.update_batch across CPU / GPU / Auto backends.

Outputs
-------
- CSV rows to stdout: n,num_perm,backend,ms
- Figures (if --plot):
    <outdir>/minhash_gpu_overview.png           # speedup+runtime vs num_perm at max(n)
    <outdir>/minhash_gpu_vs_size_k<num_perm>.png# runtime vs n at fixed num_perm
    <outdir>/minhash_gpu_size_<n>.png           # runtime vs num_perm for each n
"""

from __future__ import annotations

import argparse
import collections
import math
import os
import time
from typing import Dict, List, Optional, Tuple

import matplotlib

# non-interactive backend for CI/docs builds
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from datasketch.minhash import MinHash  # noqa: E402

MODE_LABEL = {
    "disable": "CPU (disable)",
    "always": "GPU (always)",
    "detect": "Auto (detect)",
}


def make_data(n: int) -> List[bytes]:
    # deterministic bytes for a fair compare across backends
    return [f"token-{i}".encode("utf-8") for i in range(n)]


def bench_once(n: int, num_perm: int, mode: str) -> float:
    m = MinHash(num_perm=num_perm, seed=7, gpu_mode=mode)
    data = make_data(n)
    t0 = time.perf_counter()
    m.update_batch(data)
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0  # ms


def bench_avg(n: int, num_perm: int, mode: str, repeats: int, warmup: bool) -> Optional[float]:
    if warmup:
        try:
            bench_once(n, num_perm, mode)
        except Exception:
            # GPU may be unavailable; ignore warmup failures
            pass
    times: List[float] = []
    for _ in range(repeats):
        try:
            times.append(bench_once(n, num_perm, mode))
        except Exception:
            # Treat GPU-unavailable as None so plots show NaN
            return None
    return sum(times) / len(times)


def plot_overview_vs_numperm(
    results: Dict[Tuple[int, int], Dict[str, Optional[float]]],
    sizes: List[int],
    num_perms: List[int],
    out_path: str,
) -> None:
    if not sizes or not num_perms:
        return
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    n = max(sizes)
    xs: List[int] = []
    cpu_ms: List[float] = []
    det_ms: List[float] = []
    always_ms: List[float] = []
    sp_det: List[float] = []
    sp_alw: List[float] = []

    for k in num_perms:
        rec = results.get((n, k), {})
        c, d, a = rec.get("disable"), rec.get("detect"), rec.get("always")
        if c is None:
            continue
        xs.append(k)
        cpu_ms.append(c)
        det_ms.append(d if d is not None else math.nan)
        always_ms.append(a if a is not None else math.nan)
        sp_det.append((c / d) if (d and d > 0) else math.nan)
        sp_alw.append((c / a) if (a and a > 0) else math.nan)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)

    # Speedup panel
    ax = axes[0]
    ax.plot(xs, sp_det, marker="+", label="CPU/Auto speedup")
    ax.plot(xs, sp_alw, marker="+", label="CPU/GPU speedup")
    ax.axhline(1.0, ls="--", lw=1, c="grey", alpha=0.6)
    ax.set_xlabel("num_perm")
    ax.set_ylabel("Speedup (×)")
    ax.set_title(f"Speedup at n={n}")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Runtime panel
    ax = axes[1]
    ax.plot(xs, cpu_ms, marker="+", label=MODE_LABEL["disable"])
    ax.plot(xs, det_ms, marker="+", label=MODE_LABEL["detect"])
    ax.plot(xs, always_ms, marker="+", label=MODE_LABEL["always"])
    ax.set_xlabel("num_perm")
    ax.set_ylabel("Runtime (ms)")
    ax.set_title(f"update_batch runtime at n={n} (CPU vs GPU vs Auto)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_runtime_vs_size(
    results: Dict[Tuple[int, int], Dict[str, Optional[float]]],
    sizes: List[int],
    fixed_num_perm: int,
    out_path: str,
) -> None:
    """Runtime vs n for a single num_perm across CPU/GPU/Auto."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    xs: List[int] = []
    cpu_ms: List[float] = []
    det_ms: List[float] = []
    always_ms: List[float] = []

    for n in sorted(sizes):
        rec = results.get((n, fixed_num_perm), {})
        c, d, a = rec.get("disable"), rec.get("detect"), rec.get("always")
        if c is None:
            continue
        xs.append(n)
        cpu_ms.append(c)
        det_ms.append(d if d is not None else math.nan)
        always_ms.append(a if a is not None else math.nan)

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.4))
    ax.plot(xs, cpu_ms, marker="+", label=MODE_LABEL["disable"])
    ax.plot(xs, det_ms, marker="+", label=MODE_LABEL["detect"])
    ax.plot(xs, always_ms, marker="+", label=MODE_LABEL["always"])
    ax.set_xlabel("n (batch size)")
    ax.set_ylabel("Runtime (ms)")
    ax.set_title(f"update_batch runtime vs n (num_perm={fixed_num_perm})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_per_size_panels(
    results: Dict[Tuple[int, int], Dict[str, Optional[float]]],
    sizes: List[int],
    num_perms: List[int],
    out_dir: str,
) -> None:
    """One figure per n: runtime vs num_perm."""
    os.makedirs(out_dir, exist_ok=True)
    for n in sizes:
        xs: List[int] = []
        cpu_ms: List[float] = []
        det_ms: List[float] = []
        always_ms: List[float] = []
        for k in num_perms:
            rec = results.get((n, k), {})
            c, d, a = rec.get("disable"), rec.get("detect"), rec.get("always")
            if c is None:
                continue
            xs.append(k)
            cpu_ms.append(c)
            det_ms.append(d if d is not None else math.nan)
            always_ms.append(a if a is not None else math.nan)
        fig, ax = plt.subplots(1, 1, figsize=(6.6, 4.4))
        ax.plot(xs, cpu_ms, marker="+", label=MODE_LABEL["disable"])
        ax.plot(xs, det_ms, marker="+", label=MODE_LABEL["detect"])
        ax.plot(xs, always_ms, marker="+", label=MODE_LABEL["always"])
        ax.set_xlabel("num_perm")
        ax.set_ylabel("Runtime (ms)")
        ax.set_title(f"update_batch performance (n={n}) — CPU / GPU / Auto")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(
            out_dir, f"minhash_gpu_size_{n}.png"), dpi=150)
        plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Benchmark MinHash.update_batch across CPU / GPU / Auto backends."
    )
    p.add_argument("--sizes", type=int, nargs="+",
                   default=[1_000, 10_000, 50_000])
    p.add_argument("--num-perm", type=int, nargs="+", default=[128, 256, 512])
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--warmup", action="store_true")
    p.add_argument("--out-dir", default="benchmark_out")
    p.add_argument("--plot", action="store_true")
    p.add_argument(
        "--fixed-num-perm",
        type=int,
        default=256,
        help="num_perm used for the runtime-vs-size figure",
    )
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    modes = ["disable", "detect", "always"]
    results: Dict[Tuple[int, int], Dict[str, Optional[float]]
                  ] = collections.OrderedDict()

    print("n,num_perm,backend,ms")
    for n in args.sizes:
        for k in args.num_perm:
            results[(n, k)] = {}
            for m in modes:
                ms = bench_avg(n, k, m, repeats=args.repeats,
                               warmup=args.warmup)
                results[(n, k)][m] = ms
                label = MODE_LABEL[m]
                print(f"{n},{k},{label},{'NaN' if ms is None else f'{ms:.2f}'}")

    if args.plot:
        plot_overview_vs_numperm(
            results, args.sizes, args.num_perm, os.path.join(
                args.out_dir, "minhash_gpu_overview.png")
        )
        plot_runtime_vs_size(
            results,
            args.sizes,
            args.fixed_num_perm,
            os.path.join(
                args.out_dir, f"minhash_gpu_vs_size_k{args.fixed_num_perm}.png"),
        )
        plot_per_size_panels(results, args.sizes, args.num_perm, args.out_dir)


if __name__ == "__main__":
    main()