import json, sys, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from average_precision import mapk 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark_output")
    args = parser.parse_args(sys.argv[1:])

    with open(args.benchmark_output) as f:
        benchmark = json.load(f) 

    num_perms = benchmark["num_perms"]
    lsh_times = benchmark["lsh_times"]
    linearscan_times = benchmark["linearscan_times"]
    ground_truth_results = [[x[0] for x in r] for r in benchmark["ground_truth_results"]]
    k = 10
    lsh_maps = []
    for results in benchmark["lsh_results"]:
        query_results = [[x[0] for x in r] for r in results]
        lsh_maps.append(mapk(ground_truth_results, query_results, k))
    linearscan_maps = []
    for results in benchmark["linearscan_results"]:
        query_results = [[x[0] for x in r] for r in results]
        linearscan_maps.append(mapk(ground_truth_results, query_results, k))

    lsh_times = np.array([np.percentile(ts, 90) 
        for ts in lsh_times])*1000
    linearscan_times = np.array([np.percentile(ts, 90) 
        for ts in linearscan_times])*1000

    fig, axes = plt.subplots(1, 2, figsize=(5*2, 4.5), sharex=True)
    # Plot query average MAP vs. num perm
    axes[0].plot(num_perms, linearscan_maps, marker="+", label="Linearscan")
    axes[0].plot(num_perms, lsh_maps, marker="+", label="LSH Forest")
    axes[0].set_ylabel("MAP (k = %d)" % k)
    axes[0].set_xlabel("# of Permmutation Functions")
    axes[0].grid()
    # Plot query time vs. num perm
    axes[1].plot(num_perms, linearscan_times, marker="+", label="Linearscan")
    axes[1].plot(num_perms, lsh_times, marker="+", label="LSH Forest")
    axes[1].set_xlabel("# of Permutation Functions")
    axes[1].set_ylabel("90 Percentile Query Time (ms)")
    axes[1].grid()
    axes[1].legend(loc="center right")
    fig.savefig("lshforest_benchmark.png", pad_inches=0.05, bbox_inches="tight")
