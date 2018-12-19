import json, sys, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lsh_benchmark_plot import get_precision_recall, fscore, average_fscore

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query_results")
    parser.add_argument("ground_truth_results")
    args = parser.parse_args(sys.argv[1:])
    df = pd.read_csv(args.query_results,
            converters={"results": lambda x: x.split(",")})
    df_groundtruth = pd.read_csv(args.ground_truth_results,
            converters={"results": lambda x: x.split(",")})
    df = pd.merge(df, df_groundtruth, on=["query_key", "threshold"],
            suffixes=("", "_ground_truth"))
    prs = [get_precision_recall(result, ground_truth)
            for result, ground_truth in \
                    zip(df["results"], df["results_ground_truth"])]
    df["precision"] = [p for p, _ in prs]
    df["recall"] = [r for _, r in prs]
    df["fscore"] = [fscore(*pr) for pr in prs]

    thresholds = sorted(list(set(df["threshold"])))
    num_perms = sorted(list(set(df["num_perm"])))
    num_parts = sorted(list(set(df["num_part"])))

    for i, num_perm in enumerate(num_perms):
        for j, num_part in enumerate(num_parts):
            sub = df[(df["num_part"] == num_part) & (df["num_perm"] == num_perm)].\
                    groupby("threshold")
            fscores = sub["fscore"].mean()
            recalls = sub["recall"].mean()
            precisions = sub["precision"].mean()
            plt.plot(thresholds, precisions, label="num_part = {}".format(num_part))
        plt.ylim(0.0, 1.0)
        plt.xlabel("Thresholds")
        plt.ylabel("Average Precisions")
        plt.grid()
        plt.legend()
        plt.savefig("lshensemble_num_perm_{}.png".format(num_perm))

