import json, sys, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def get_precision_recall(found, reference):
    reference = set(reference)
    intersect = sum(1 for i in found if i in reference)
    if len(found) == 0:
        precision = 0.0
    else:
        precision = float(intersect) / float(len(found))
    if len(reference) == 0:
        recall = 1.0
    else:
        recall = float(intersect) / float(len(reference))
    if len(found) == len(reference) == 0:
        precision = 1.0
        recall = 1.0
    return [precision, recall]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark_output")
    args = parser.parse_args(sys.argv[1:])

    with open(args.benchmark_output) as f:
        benchmark = json.load(f) 

    num_perms = benchmark["num_perms"]
    lsh_times = benchmark["lsh_times"]
    linearscan_times = benchmark["linearscan_times"]
    lsh_results = [np.array(r) for r in benchmark["lsh_results"]]
    linearscan_results = [np.array(r) for r in benchmark["linearscan_results"]]
    ground_truth_results = np.array(benchmark["ground_truth_results"])

    # Slice out the empty results
    indices = np.array([i for i, r in enumerate(ground_truth_results) if len(r) > 0])
    ground_truth_results = ground_truth_results[indices]
    linearscan_results = [r[indices] for r in linearscan_results]
    lsh_results = [r[indices] for r in lsh_results]

    lsh_times = np.array([np.percentile(ts, 90) 
        for ts in lsh_times])*1000
    linearscan_times = np.array([np.percentile(ts, 90) 
        for ts in linearscan_times])*1000
    lsh_precisions, lsh_recalls = \
            np.array([np.mean([get_precision_recall(f, r) 
                for f, r in zip(l, ground_truth_results)], axis=0) 
                for l in lsh_results]).T
    linearscan_precisions, linearscan_recalls = \
            np.array([np.mean([get_precision_recall(f, r) 
                for f, r in zip(l, ground_truth_results)], axis=0) 
                for l in linearscan_results]).T

    fig, axes = plt.subplots(1, 3, figsize=(5*3, 4.5), sharex=True)
    # Plot query precision vs. num perm
    axes[0].plot(num_perms, linearscan_precisions, marker="+", label="Linearscan")
    axes[0].plot(num_perms, lsh_precisions, marker="+", label="LSH")
    axes[0].set_ylabel("Average Precision")
    axes[0].set_xlabel("# of Permmutation Functions")
    axes[0].grid()
    # Plot query recall vs. num perm
    axes[1].plot(num_perms, linearscan_precisions, marker="+", label="Linearscan")
    axes[1].plot(num_perms, lsh_recalls, marker="+", label="LSH")
    axes[1].set_ylabel("Average Recall")
    axes[1].set_xlabel("# of Permmutation Functions")
    axes[1].grid()
    # Plot query time vs. num perm
    axes[2].plot(num_perms, linearscan_times, marker="+", label="Linearscan")
    axes[2].plot(num_perms, lsh_times, marker="+", label="LSH")
    axes[2].set_xlabel("# of Permutation Functions")
    axes[2].set_ylabel("90 Percentile Query Time (ms)")
    axes[2].grid()
    axes[2].legend(loc="center right")
    fig.savefig("lsh_benchmark.png", pad_inches=0.05, bbox_inches="tight")
