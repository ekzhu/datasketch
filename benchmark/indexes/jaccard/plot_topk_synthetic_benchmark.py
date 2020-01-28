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
            default="jaccard_topk_synthetic_benchmark")
    args = parser.parse_args(sys.argv[1:])    

    conn = sqlite3.connect(args.benchmark_result)

    # Get all runs.
    cursor = conn.cursor()
    cursor.execute("""SELECT key, name, k, params
            FROM runs ORDER BY k, name""")
    runs = [{
                "key": key,
                "name": name, 
                "k": k, 
                **json.loads(params),
            }
            for (key, name, k, params) in cursor.fetchall()]
    cursor.close()

    # Get ground truth results first.
    for i, run in enumerate([run for run in runs 
            if run["name"] == "ground_truth"]):
        results, times = load_results(run["key"], conn)
        run.update({"results": results, "times": times})
    
    # Compute mean recall and query time of every run.
    for i, run in enumerate([run for run in runs 
            if run["name"] != "ground_truth"]):
        # Load results of this run.
        results, times = load_results(run["key"], conn)
        # Find the corresponding ground truth run.
        ground_truth = [x for x in runs if x["name"] == "ground_truth" and
                x["vocabulary_size"] == run["vocabulary_size"]][0]
        # Compute metrics.
        recalls = compute_recalls(results, ground_truth["results"])
        relevances = compute_relevances(results, ground_truth["results"])
        mean_recall = np.mean([x[1] for x in recalls])
        mean_relevance = np.mean([x[1] for x in relevances])
        mean_time = np.mean(times * 1000)
        # Update run with computed metrics.
        run.update({
            "mean_recall": mean_recall, 
            "mean_relevance": mean_relevance, 
            "mean_time": mean_time,
        })
        
    # Plot.
    ks = np.sort(np.unique([run["k"] for run in runs]))
    for k in ks:
        plt.figure()
        # Plot recall vs. vocabulary size.
        # Get vocabulary sizes.
        vs = np.sort(np.unique(
                [run["vocabulary_size"] for run in runs if run["k"] == k]))
        # Get algorithms.
        names = np.sort(np.unique(
                [run["name"] for run in runs if run["k"] == k and
                run["name"] != "ground_truth"]))
        # Get the recall for every vocabulary size.
        for name in names:
            recalls = [[run["mean_recall"] for run in runs if run["k"] == k and
                    run["vocabulary_size"] == v and run["name"] == name][0]
                    for v in vs]
            plt.plot(vs, recalls, '-x', label=name)
        plt.xlabel("Vocabulary Size")
        plt.ylabel(f"Recall @ {k}")
        plt.legend()
        plt.grid()
        plt.savefig(f"{args.plot_filename_prefix}_recall_vs_vocabulary_size_{k}.png")
