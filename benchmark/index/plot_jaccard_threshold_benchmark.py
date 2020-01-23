import argparse
import sys
import json
import sqlite3
import collections

import matplotlib.pyplot as plt
import numpy as np

from utils import load_results, compute_recalls

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-output", required=True)
    parser.add_argument("--plot-filename-prefix", 
            default="jaccard_threshold_benchmark")
    args = parser.parse_args(sys.argv[1:])    

    conn = sqlite3.connect(args.benchmark_output)

    # Get all runs.
    cursor = conn.cursor()
    cursor.execute("""SELECT key, name, threshold, params
            FROM runs ORDER BY threshold, name""")
    runs = cursor.fetchall()
    cursor.close()

    # Group runs by threshold and then by names.
    groups = collections.defaultdict(lambda: collections.defaultdict(list))
    for run in runs:
        (key, name, threshold, params) = run
        groups[threshold][name].append([key, params])

    # Compute the mean recall and query time of every run.
    points = collections.defaultdict(lambda: collections.defaultdict(list))
    for threshold in groups:
        ground_run_key, _ = groups[threshold]["ground_truth"][0]
        ground_results, ground_times = load_results(ground_run_key, conn)
        for name in groups[threshold]:
            if name == "groud_truth":
                continue
            for run_key, _ in groups[threshold][name]:
                results, times = load_results(run_key, conn)
                recalls = compute_recalls(results, ground_results)
                mean_recall = np.mean([x[1] for x in recalls])
                mean_time = np.mean(times)
                points[threshold][name].append([mean_recall, mean_time])
            points[threshold][name].sort(key=lambda x : x[0])
    
    # Plot.
    for threshold in points:
        plt.figure()
        for name in points[threshold]:
            recalls, times = np.array(points[threshold][name]).T
            plt.plot(recalls, times, "-x", label=name)
        plt.xlabel(f"Recall @ {threshold:.2f}")
        plt.ylabel("Time (sec)")
        plt.grid()
        plt.legend()
        plt.savefig(f"{args.plot_filename_prefix}_{threshold:.2f}.png")
        plt.close()
