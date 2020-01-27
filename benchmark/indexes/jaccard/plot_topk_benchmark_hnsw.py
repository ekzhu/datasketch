import argparse
import sys
import json
import sqlite3
import collections

import matplotlib.pyplot as plt
import numpy as np

from utils import load_results, compute_recalls

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Plot HNSW topk benchmark results.")
    parser.add_argument("--benchmark-result", required=True)
    parser.add_argument("--plot-filename-prefix", 
            default="jaccard_topk_hnsw")
    args = parser.parse_args(sys.argv[1:])    

    conn = sqlite3.connect(args.benchmark_result)

    # Get runs.
    cursor = conn.cursor()
    cursor.execute("""SELECT key, name, k, params
            FROM runs WHERE name in ('hnsw', 'ground_truth') 
            ORDER BY k, name""")
    runs = cursor.fetchall()
    cursor.close()

    # Group runs by k and then by names.
    groups = collections.defaultdict(lambda: collections.defaultdict(list))
    for run in runs:
        (key, name, k, params) = run
        groups[k][name].append([key, params])

    # Compute the mean recall and query time of every run.
    points = collections.defaultdict(lambda: collections.defaultdict(list))
    for k in groups:
        ground_run_key, _ = groups[k]["ground_truth"][0]
        ground_results, ground_times = load_results(ground_run_key, conn)
        for run_key, params in groups[k]['hnsw']:
            results, times = load_results(run_key, conn)
            recalls = compute_recalls(results, ground_results)
            mean_recall = np.mean([x[1] for x in recalls])
            mean_time = np.mean(times * 1000)
            # Extract HNSW parameters.
            params = json.loads(params)
            M = params["M"]
            ef = params["efConstruction"]
            points[k]['hnsw'].append([mean_recall, mean_time, M, ef])
        points[k]['hnsw'].sort(key=lambda x : x[2])
    
    for k in points:
        efs = list(np.unique([p[3] for p in points[k]['hnsw']]))
        # Plot recall vs. M.
        plt.figure()
        for ef in efs:
            # Get points given ef.
            ps = np.array([p for p in points[k]['hnsw'] if p[3] == ef])
            recalls, _, Ms, _ = ps.T
            plt.plot(Ms, recalls, "-x", label=f"ef = {ef}")
        plt.xlabel("M")
        plt.ylabel(f"Recall @ {k}")
        plt.grid()
        plt.legend()
        plt.savefig(f"{args.plot_filename_prefix}_{k}_recalls.png")
        plt.close()
        # Plot time vs. M. 
        plt.figure()
        for ef in efs:
            # Get points given ef.
            ps = np.array([p for p in points[k]['hnsw'] if p[3] == ef])
            _, times, Ms, _ = ps.T
            plt.plot(Ms, times, "-x", label=f"ef = {ef}")
        plt.xlabel("Ms")
        plt.ylabel("Time (ms)")
        plt.grid()
        plt.legend()
        plt.savefig(f"{args.plot_filename_prefix}_{k}_times.png")
        plt.close()
