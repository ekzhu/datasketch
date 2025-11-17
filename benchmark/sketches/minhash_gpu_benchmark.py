import argparse
import os
import sys
import time
from collections import defaultdict

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None  # still allow CSV-only runs

from datasketch.minhash import MinHash

# ---- GPU availability helper (mirrors library behavior) ----


def _gpu_available() -> bool:
    try:
        import cupy as cp  # noqa: F401
        try:
            return cp.cuda.runtime.getDeviceCount() > 0
        except Exception:
            return False
    except Exception:
        return False


def make_data(n):
    # bytes are closer to real MinHash usage
    return [f"token-{i}".encode("utf-8") for i in range(n)]


def bench_once(n, num_perm, use_gpu):
    m = MinHash(num_perm=num_perm, seed=7, use_gpu=use_gpu)
    data = make_data(n)
    t0 = time.perf_counter()
    m.update_batch(data)
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0  # milliseconds


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", type=int, nargs="+",
                   default=[1_000, 5_000, 10_000, 50_000, 100_000])
    p.add_argument("--num-perm", type=int, nargs="+",
                   default=[64, 128, 256, 512])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--warmup", action="store_true",
                   help="Do a warm-up run for each backend/setting.")
    p.add_argument("--plot", action="store_true",
                   help="Generate runtime and speedup PNGs in --outdir.")
    p.add_argument("--outdir", default="benchmark",
                   help="Directory to write figures if --plot is set.")
    args = p.parse_args()

    gpu_ok = _gpu_available()

    # results[(n, k)][backend] -> list[ms]
    results = defaultdict(lambda: {"cpu": [], "gpu": []})

    print("n,num_perm,backend,ms")
    for n in args.sizes:
        for k in args.num_perm:
            # CPU
            if args.warmup:
                try:
                    bench_once(n, k, use_gpu=False)
                except Exception:
                    pass
            for _ in range(args.repeats):
                ms = bench_once(n, k, use_gpu=False)
                results[(n, k)]["cpu"].append(ms)
                print(f"{n},{k},cpu,{ms:.2f}")

            # GPU (skip gracefully if unavailable)
            if gpu_ok:
                if args.warmup:
                    try:
                        bench_once(n, k, use_gpu=True)
                    except Exception:
                        # if something went wrong mid-run, treat as unavailable
                        gpu_ok = False
                if gpu_ok:
                    for _ in range(args.repeats):
                        ms = bench_once(n, k, use_gpu=True)
                        results[(n, k)]["gpu"].append(ms)
                        print(f"{n},{k},gpu,{ms:.2f}")
            else:
                # keep the CSV tidy with a comment
                print(f"# gpu_unavailable_for_n_{n}_k_{k}")

    if args.plot:
        if plt is None:
            print("# matplotlib not available; cannot plot.", file=sys.stderr)
            return

        os.makedirs(args.outdir, exist_ok=True)

        # Aggregate means
        means = {}
        for (n, k), vals in results.items():
            cpu_mean = sum(vals["cpu"]) / len(vals["cpu"]
                                              ) if vals["cpu"] else None
            gpu_mean = sum(vals["gpu"]) / len(vals["gpu"]
                                              ) if vals["gpu"] else None
            means[(n, k)] = {"cpu": cpu_mean, "gpu": gpu_mean}

        # ---- Figure 1: Runtime (ms) vs num_perm, separate lines per size, CPU vs GPU ----
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        sizes_sorted = sorted(set(n for (n, _) in means))
        perms_sorted = sorted(set(k for (_, k) in means))

        for n in sizes_sorted:
            y_cpu = []
            y_gpu = []
            for k in perms_sorted:
                m = means[(n, k)]
                y_cpu.append(m["cpu"])
                y_gpu.append(m["gpu"])
            ax.plot(perms_sorted, y_cpu, marker="o", label=f"CPU n={n}")
            if any(v is not None for v in y_gpu):
                ax.plot(perms_sorted, y_gpu, marker="x",
                        linestyle="--", label=f"GPU n={n}")

        ax.set_xlabel("num_perm")
        ax.set_ylabel("Runtime (ms)")
        ax.set_title("MinHash update_batch runtime (CPU vs GPU)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        path_runtime = os.path.join(args.outdir, "minhash_gpu_runtime.png")
        fig.savefig(path_runtime, dpi=150)

        # ---- Figure 2: Speedup (CPU / GPU) vs num_perm, lines per size ----
        # Only plot points where both cpu and gpu exist
        fig2, ax2 = plt.subplots(1, 1, figsize=(8, 4))
        for n in sizes_sorted:
            x = []
            y = []
            for k in perms_sorted:
                m = means[(n, k)]
                if m["cpu"] is not None and m["gpu"] is not None and m["gpu"] > 0:
                    x.append(k)
                    y.append(m["cpu"] / m["gpu"])
            if x:
                ax2.plot(x, y, marker="o", label=f"speedup n={n}")

        ax2.axhline(1.0, color="gray", linewidth=1)
        ax2.set_xlabel("num_perm")
        ax2.set_ylabel("Speedup (CPU ms / GPU ms)")
        ax2.set_title("GPU speedup (>1 = GPU faster)")
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=8, ncol=2)
        fig2.tight_layout()
        path_speed = os.path.join(args.outdir, "minhash_gpu_speedup.png")
        fig2.savefig(path_speed, dpi=150)

        print(f"# Wrote {path_runtime}")
        print(f"# Wrote {path_speed}")


if __name__ == "__main__":
    main()
