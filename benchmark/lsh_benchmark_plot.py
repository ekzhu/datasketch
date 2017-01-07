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

def fscore(precision, recall):
    if precision == 0.0 and recall == 0.0:
        return 0.0
    return 2.0 / (1.0 / precision + 1.0 / recall)

def average_fscore(founds, references):
    return np.mean([fscore(*get_precision_recall(found, reference))
                    for found, reference in zip(founds, references)])

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
    lsh_fscores = []
    for results in benchmark["lsh_results"]:
        query_results = [[x[0] for x in r] for r in results]
        lsh_fscores.append(average_fscore(query_results, ground_truth_results))
    linearscan_fscores = []
    for results in benchmark["linearscan_results"]:
        query_results = [[x[0] for x in r] for r in results]
        linearscan_fscores.append(average_fscore(query_results, ground_truth_results))

    lsh_times = np.array([np.percentile(ts, 90) 
        for ts in lsh_times])*1000
    linearscan_times = np.array([np.percentile(ts, 90) 
        for ts in linearscan_times])*1000

    fig, axes = plt.subplots(1, 2, figsize=(5*2, 4.5), sharex=True)
    # Plot query fscore vs. num perm
    axes[0].plot(num_perms, linearscan_fscores, marker="+", label="Linearscan")
    axes[0].plot(num_perms, lsh_fscores, marker="+", label="LSH")
    axes[0].set_ylabel("Average F-Score")
    axes[0].set_xlabel("# of Permmutation Functions")
    axes[0].grid()
    # Plot query time vs. num perm
    axes[1].plot(num_perms, linearscan_times, marker="+", label="Linearscan")
    axes[1].plot(num_perms, lsh_times, marker="+", label="LSH")
    axes[1].set_xlabel("# of Permutation Functions")
    axes[1].set_ylabel("90 Percentile Query Time (ms)")
    axes[1].grid()
    axes[1].legend(loc="center right")
    fig.savefig("lsh_benchmark.png", pad_inches=0.05, bbox_inches="tight")
