import json, sys, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lsh_benchmark_plot import get_precision_recall, fscore, average_fscore

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark_output")
    args = parser.parse_args(sys.argv[1:])

    with open(args.benchmark_output) as f:
        benchmark = json.load(f) 

    num_perms = benchmark["num_perms"]
    lsh_times = benchmark["lsh_times"]
    ground_truth_results = [[x[0] for x in r] for r in benchmark["ground_truth_results"]]
    lsh_fscores = []
    for results in benchmark["lsh_results"]:
        query_results = [[x[0] for x in r] for r in results]
        lsh_fscores.append(average_fscore(query_results, ground_truth_results))

    lsh_times = np.array([np.percentile(ts, 90) 
        for ts in lsh_times])*1000

    fig, axes = plt.subplots(1, 2, figsize=(5*2, 4.5), sharex=True)
    # Plot query fscore vs. num perm
    axes[0].plot(num_perms, lsh_fscores, marker="+", label="LSH Ensemble")
    axes[0].set_ylabel("Average F-Score")
    axes[0].set_xlabel("# of Permmutation Functions")
    axes[0].grid()
    # Plot query time vs. num perm
    axes[1].plot(num_perms, lsh_times, marker="+", label="LSH Ensemble")
    axes[1].set_xlabel("# of Permutation Functions")
    axes[1].set_ylabel("90 Percentile Query Time (ms)")
    axes[1].grid()
    axes[1].legend(loc="lower right")
    plt.tight_layout()
    fig.savefig("lshensemble_benchmark.png", pad_inches=0.05, bbox_inches="tight")
