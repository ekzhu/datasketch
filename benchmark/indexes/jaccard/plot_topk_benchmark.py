import argparse
import sys
import json
import sqlite3
import collections

import matplotlib.pyplot as plt
import numpy as np

from utils import evaluate_runs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark_result")
    parser.add_argument("--plot-filename-prefix", 
            default="jaccard_topk_benchmark")
    args = parser.parse_args(sys.argv[1:])    
    runs = evaluate_runs(args.benchmark_result)
    ks = np.sort(np.unique([run["k"] for run in runs]))
    # Plot.
    for k in ks:
        # Get algorithms.
        names = np.sort(np.unique(
                [run["name"] for run in runs if run["k"] == k and
                run["name"] != "ground_truth"]))
        # Plot recall vs. time. 
        plt.figure()
        for name in names:
            # Get recalls and times of this algorithm.
            selected = [run for run in runs
                    if run["k"] == k and run["name"] == name]
            recalls = [run["mean_recall"] for run in selected]
            times = [run["mean_time"] for run in selected]
            plt.plot(recalls, times, "*", label=name)
        plt.xlabel(f"Recall @ {k}")
        plt.ylabel("Time (ms)")
        plt.grid()
        plt.legend()
        plt.savefig(f"{args.plot_filename_prefix}_recall_{k}.png")
        plt.close()
        # Plot relevance vs. time. 
        plt.figure()
        for name in names:
            # Get recalls and times of this algorithm.
            selected = [run for run in runs
                    if run["k"] == k and run["name"] == name]
            relevances = [run["mean_relevance"] for run in selected]
            times = [run["mean_time"] for run in selected]
            plt.plot(relevances, times, "*", label=name)
        plt.xlabel(f"Relevance @ {k}")
        plt.ylabel("Time (ms)")
        plt.grid()
        plt.legend()
        plt.savefig(f"{args.plot_filename_prefix}_relevance_{k}.png")
        plt.close()
