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
    parser.add_argument("--benchmark-result", required=True)
    parser.add_argument("--plot-filename-prefix", 
            default="jaccard_topk_synthetic_benchmark")
    args = parser.parse_args(sys.argv[1:])    

    runs = evaluate_runs(args.benchmark_results)
    ks = np.sort(np.unique([run["k"] for run in runs]))
    for k in ks:
        plt.figure()
        # Plot recall vs. vocabulary size.
        # Get vocabulary sizes.
        vs = np.sort(np.unique(
                [run["benchmark"]["vocabulary_size"] 
                for run in runs if run["k"] == k]))
        # Get algorithms.
        names = np.sort(np.unique(
                [run["name"] for run in runs if run["k"] == k and
                run["name"] != "ground_truth"]))
        # Get the recall for every vocabulary size.
        for name in names:
            recalls = [[run["mean_recall"] for run in runs 
                    if run["k"] == k and
                    run["benchmark"]["vocabulary_size"] == v and 
                    run["name"] == name][0]
                    for v in vs]
            plt.plot(vs, recalls, '-x', label=name)
        plt.xlabel("Vocabulary Size")
        plt.ylabel(f"Recall @ {k}")
        plt.legend()
        plt.grid()
        plt.savefig(f"{args.plot_filename_prefix}_recall_vs_vocabulary_size_{k}.png")
