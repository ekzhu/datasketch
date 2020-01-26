import argparse
import sys
import json
import sqlite3
import collections

import matplotlib.pyplot as plt
import numpy as np

from utils import load_results, compute_recalls, compute_relevances


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-result", required=True)
    parser.add_argument("--plot-filename-prefix", 
            default="jaccard_topk_benchmark")
    args = parser.parse_args(sys.argv[1:])    

    conn = sqlite3.connect(args.benchmark_result)

    # Get all runs.
    cursor = conn.cursor()
    cursor.execute("""SELECT key, name, k, params
            FROM runs ORDER BY k, name""")
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
        for name in groups[k]:
            if name == "ground_truth":
                continue
            for run_key, _ in groups[k][name]:
                results, times = load_results(run_key, conn)
                recalls = compute_recalls(results, ground_results)
                relevances = compute_relevances(results, ground_results)
                mean_recall = np.mean([x[1] for x in recalls])
                mean_relevance = np.mean([x[1] for x in relevances])
                mean_time = np.mean(times * 1000)
                points[k][name].append([mean_recall, mean_relevance, mean_time])
            points[k][name].sort(key=lambda x : x[0])
    
    # Plot.
    for k in points:
        plt.figure()
        for name in points[k]:
            recalls, _, times = np.array(points[k][name]).T
            plt.plot(recalls, times, "-x", label=name)
        plt.xlabel(f"Recall @ {k}")
        plt.ylabel("Time (ms)")
        plt.grid()
        plt.legend()
        plt.savefig(f"{args.plot_filename_prefix}_recall_{k}.png")
        plt.close()
        
        plt.figure()
        for name in points[k]:
            _, relevances, times = np.array(points[k][name]).T
            plt.plot(relevances, times, "-x", label=name)
        plt.xlabel(f"Relevance @ {k}")
        plt.ylabel("Time (ms)")
        plt.grid()
        plt.legend()
        plt.savefig(f"{args.plot_filename_prefix}_relevance_{k}.png")
        plt.close()
