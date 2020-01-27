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
            description="Plot MinHash LSH threshold benchmark results.")
    parser.add_argument("--benchmark-result", required=True)
    parser.add_argument("--plot-filename-prefix", 
            default="jaccard_threshold_lsh")
    args = parser.parse_args(sys.argv[1:])    

    conn = sqlite3.connect(args.benchmark_result)

    # Get runs.
    cursor = conn.cursor()
    cursor.execute("""SELECT key, name, threshold, params
            FROM runs WHERE name in ('lsh', 'ground_truth') 
            ORDER BY threshold, name""")
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
        for run_key, params in groups[threshold]['lsh']:
            results, times = load_results(run_key, conn)
            recalls = compute_recalls(results, ground_results)
            mean_recall = np.mean([x[1] for x in recalls])
            mean_time = np.mean(times * 1000)
            num_perm = json.loads(params)["num_perm"]
            points[threshold]['lsh'].append([mean_recall, mean_time, num_perm])
        points[threshold]['lsh'].sort(key=lambda x : x[2])
    
    for threshold in points:
        # Plot recall vs. num perm.
        plt.figure()
        recalls, _, num_perms = np.array(points[threshold]['lsh']).T
        plt.plot(num_perms, recalls, "-x")
        plt.xlabel(f"num_perm")
        plt.ylabel(f"Recall @ {threshold:.2f}")
        plt.grid()
        plt.legend()
        plt.savefig(f"{args.plot_filename_prefix}_{threshold:.2f}_recalls.png")
        plt.close()
        # Plot time vs. num perm. 
        plt.figure()
        _, times, num_perms = np.array(points[threshold]['lsh']).T
        plt.plot(num_perms, times, "-x")
        plt.xlabel(f"num_perm")
        plt.ylabel("Time (ms)")
        plt.grid()
        plt.legend()
        plt.savefig(f"{args.plot_filename_prefix}_{threshold:.2f}_times.png")
        plt.close()
