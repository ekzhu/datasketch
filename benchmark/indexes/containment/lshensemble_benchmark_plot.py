import json, sys, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from utils import get_precision_recall, fscore, average_fscore

def _parse_results(r):
    r = r.strip().split(",")
    return [x for x in r if len(x) > 0]

def _label(num_part):
    if num_part == 1:
        label = "MinHash LSH"
    else:
        label = "LSH Ensemble ({})".format(num_part)
    return label

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query_results")
    parser.add_argument("ground_truth_results")
    parser.add_argument("--asym-query-results")
    args = parser.parse_args(sys.argv[1:])
    df = pd.read_csv(args.query_results,
            converters={"results": _parse_results})
    df_groundtruth = pd.read_csv(args.ground_truth_results,
            converters={"results": _parse_results})
    df_groundtruth["has_result"] = [len(r) > 0
            for r in df_groundtruth["results"]]
    df_groundtruth = df_groundtruth[df_groundtruth["has_result"]]
    df = pd.merge(df, df_groundtruth, on=["query_key", "threshold"],
            suffixes=("", "_ground_truth"))
    prs = [get_precision_recall(result, ground_truth)
            for result, ground_truth in \
                    zip(df["results"], df["results_ground_truth"])]
    df["precision"] = [p for p, _ in prs]
    df["recall"] = [r for _, r in prs]
    df["fscore"] = [fscore(*pr) for pr in prs]
    #df["query_time_lshensemble"] = df["probe_time"] + df["process_time"]
    df["query_time_lshensemble"] = df["probe_time"]

    if args.asym_query_results is not None:
        df_asym = pd.read_csv(args.asym_query_results,
                converters={"results": _parse_results})
        df = pd.merge(df, df_asym, on=["query_key", "threshold"],
                suffixes=("", "_asym"))
        prs = [get_precision_recall(result, ground_truth)
                for result, ground_truth in \
                        zip(df["results_asym"], df["results_ground_truth"])]
        df["precision_asym"] = [p for p, _ in prs]
        df["recall_asym"] = [r for _, r in prs]
        df["fscore_asym"] = [fscore(*pr) for pr in prs]
        #df["query_time_asym"] = df["probe_time_asym"] + df["process_time_asym"]
        df["query_time_asym"] = df["probe_time_asym"]

    thresholds = sorted(list(set(df["threshold"])))
    num_perms = sorted(list(set(df["num_perm"])))
    num_parts = sorted(list(set(df["num_part"])))

    for i, num_perm in enumerate(num_perms):
        # Plot precisions
        for j, num_part in enumerate(num_parts):
            sub = df[(df["num_part"] == num_part) & (df["num_perm"] == num_perm)].\
                    groupby("threshold")
            precisions = sub["precision"].mean()
            stds = sub["precision"].std()
            plt.plot(thresholds, precisions, "^-", label=_label(num_part))
            #plt.fill_between(thresholds, precisions-stds, precisions+stds,
            #        alpha=0.2)
        if "precision_asym" in df:
            sub = df[(df["num_part"] == 1) & (df["num_perm"] == num_perm)].\
                    groupby("threshold")
            precisions = sub["precision_asym"].mean()
            stds = sub["precision_asym"].std()
            plt.plot(thresholds, precisions, "s-", label="Asym Minhash LSH")
            #plt.fill_between(thresholds, precisions-stds, precisions+stds,
            #        alpha=0.2)
        plt.ylim(0.0, 1.0)
        plt.xlabel("Thresholds")
        plt.ylabel("Average Precisions")
        plt.grid()
        plt.legend()
        plt.savefig("lshensemble_num_perm_{}_precision.png".format(num_perm))
        plt.close()
        # Plot recalls
        for j, num_part in enumerate(num_parts):
            sub = df[(df["num_part"] == num_part) & (df["num_perm"] == num_perm)].\
                    groupby("threshold")
            recalls = sub["recall"].mean()
            stds = sub["recall"].std()
            plt.plot(thresholds, recalls, "^-", label=_label(num_part))
            #plt.fill_between(thresholds, recalls-stds, recalls+stds, alpha=0.2)
        if "recall_asym" in df:
            sub = df[(df["num_part"] == 1) & (df["num_perm"] == num_perm)].\
                    groupby("threshold")
            recalls = sub["recall_asym"].mean()
            stds = sub["recall_asym"].std()
            plt.plot(thresholds, recalls, "s-", label="Asym Minhash LSH")
            #plt.fill_between(thresholds, recalls-stds, recalls+stds, alpha=0.2)
        plt.ylim(0.0, 1.0)
        plt.xlabel("Thresholds")
        plt.ylabel("Average Recalls")
        plt.grid()
        plt.legend()
        plt.savefig("lshensemble_num_perm_{}_recall.png".format(num_perm))
        plt.close()
        # Plot fscores.
        for j, num_part in enumerate(num_parts):
            sub = df[(df["num_part"] == num_part) & (df["num_perm"] == num_perm)].\
                    groupby("threshold")
            fscores = sub["fscore"].mean()
            stds = sub["fscore"].std()
            plt.plot(thresholds, fscores, "^-", label=_label(num_part))
            #plt.fill_between(thresholds, fscores-stds, fscores+stds, alpha=0.2)
        if "fscore_asym" in df:
            sub = df[(df["num_part"] == 1) & (df["num_perm"] == num_perm)].\
                    groupby("threshold")
            fscores = sub["fscore_asym"].mean()
            stds = sub["fscore_asym"].std()
            plt.plot(thresholds, fscores, "s-", label="Asym Minhash LSH")
            #plt.fill_between(thresholds, fscores-stds, fscores+stds, alpha=0.2)
        plt.ylim(0.0, 1.0)
        plt.xlabel("Thresholds")
        plt.ylabel("Average F-Scores")
        plt.grid()
        plt.legend()
        plt.savefig("lshensemble_num_perm_{}_fscore.png".format(num_perm))
        plt.close()
        # Plot query time.
        for num_part in num_parts:
            sub = df[(df["num_part"] == num_part) & (df["num_perm"] == num_perm)].\
                    groupby("threshold")
            t = sub["query_time_lshensemble"].quantile(0.9)
            plt.plot(thresholds, t, "^-", label=_label(num_part))
        t = df_groundtruth.groupby("threshold")["query_time"].quantile(0.9)
        plt.plot(thresholds, t, "o-", label="Exact")
        plt.xlabel("Thresholds")
        plt.ylabel("90 Percentile Query Time (ms)")
        plt.legend()
        plt.grid()
        plt.savefig("lshensemble_num_perm_{}_query_time.png".format(num_perm))
        plt.close()

    # Output results
    # df = df.drop(columns=["results", "results_ground_truth", "results_asym"])
    # df.to_csv("out.csv")
