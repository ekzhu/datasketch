import collections
import math
import time
import os
import matplotlib.pyplot as plt
from networkx import constraint
from datasketch.minhash import MinHash
from typing import Dict, Tuple, Optional, List
import matplotlib
matplotlib.use("Agg")

# Map API modes -> human-friendly labels
MODE_LABEL = {
    "disable": "CPU (disable)",
    "always":  "GPU (always)",
    "detect":  "Auto (detect)",
}


def make_data(n: int) -> List[bytes]:
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
    times = []
    for _ in range(repeats):
        try:
            times.append(bench_once(n, num_perm, mode))
        except Exception:
            # Treat GPU-unavailable as None (so plots show NaN)
            return None
    return sum(times) / len(times)


def plot_overview(results: Dict[Tuple[int, int], Dict[str, Optional[float]]],
                  sizes: List[int], num_perms: List[int], out_path: str) -> None:
    if not sizes or not num_perms:
        return
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    n = max(sizes)
    xs, cpu_ms, det_ms, always_ms, sp_det, sp_alw = [], [], [], [], [], []
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

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True, constrained_layout=True)

    ax = axes[0]
    ax.plot(xs, sp_det, marker="+", label="CPU/Auto speedup")
    ax.plot(xs, sp_alw, marker="+", label="CPU/GPU speedup")
    ax.axhline(1.0, ls="--", lw=1, c="grey", alpha=0.6)
    ax.set_xlabel("num_perm")
    ax.set_ylabel("Speedup (×)")
    ax.set_title(f"Speedup at n={n}")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1]
    ax.plot(xs, cpu_ms, marker="+", label=MODE_LABEL["disable"])
    ax.plot(xs, det_ms, marker="+", label=MODE_LABEL["detect"])
    ax.plot(xs, always_ms, marker="+", label=MODE_LABEL["always"])
    ax.set_xlabel("num_perm")
    ax.set_ylabel("Runtime (ms)")
    ax.set_title(f"update_batch runtime at n={n} (CPU vs GPU vs Auto)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.suptitle(
        "Modes: CPU (disable), GPU (always), Auto (detect)", fontsize=10)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_per_size(results, sizes, num_perms, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for n in sizes:
        xs, cpu_ms, det_ms, always_ms = [], [], [], []
        for k in num_perms:
            rec = results.get((n, k), {})
            c, d, a = rec.get("disable"), rec.get("detect"), rec.get("always")
            if c is None:
                continue
            xs.append(k)
            cpu_ms.append(c)
            det_ms.append(d if d is not None else math.nan)
            always_ms.append(a if a is not None else math.nan)
        fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.4))
        ax.plot(xs, cpu_ms, marker="+", label=MODE_LABEL["disable"])
        ax.plot(xs, det_ms, marker="+", label=MODE_LABEL["detect"])
        ax.plot(xs, always_ms, marker="+", label=MODE_LABEL["always"])
        ax.set_xlabel("num_perm")
        ax.set_ylabel("Runtime (ms)")
        ax.set_title(f"update_batch performance (n={n}) — CPU / GPU / Auto")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.savefig(os.path.join(
            out_dir, f"minhash_gpu_size_{n}.png"), dpi=150)
        plt.close(fig)


def main():
    import argparse
    p = argparse.ArgumentParser(
        description="Benchmark MinHash.update_batch CPU vs GPU vs Auto with plotting.")
    p.add_argument("--sizes", type=int, nargs="+",
                   default=[1_000, 10_000, 50_000])
    p.add_argument("--num-perm", type=int, nargs="+", default=[128, 256, 512])
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--warmup", action="store_true")
    p.add_argument("--out-dir", default="benchmark_out")
    p.add_argument("--plot", action="store_true")
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
                ms = bench_avg(n, k, m, args.repeats, args.warmup)
                results[(n, k)][m] = ms
                label = MODE_LABEL[m]
                print(f"{n},{k},{label},{'NaN' if ms is None else f'{ms:.2f}'}")

    if args.plot:
        plot_overview(results, args.sizes, args.num_perm, os.path.join(
            args.out_dir, "minhash_gpu_overview.png"))
        plot_per_size(results, args.sizes, args.num_perm, args.out_dir)


if __name__ == "__main__":
    main()
