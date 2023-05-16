import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from utils import evaluate_runs


def _annotate_points(plt, name, xs, ys, runs):
    for x, y, run in zip(xs, ys, runs):
        if name == "lsh" or name == "lshforest":
            text = f"({run['b']}, {run['r']})"
        elif name == "hnsw":
            text = f"({run['m']}, {run['efConstruction']})"
        else:
            break
        plt.annotate(
            text,
            xy=(x, y),
            # ha="center",
            # va="center",
            # fontsize=8,
            # arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.1"),
        )


def _frontier_points(xs, ys, metadata, y_dir="increasing"):
    assert len(xs) == len(ys) and len(xs) == len(metadata)
    frontier_xs = []
    frontier_ys = []
    frontier_metadata = []
    ys, xs, metadata = zip(
        *sorted(zip(ys, xs, metadata), reverse=y_dir != "increasing")
    )
    for i in range(len(xs)):
        if i == 0 or xs[i] >= frontier_xs[-1]:
            frontier_xs.append(xs[i])
            frontier_ys.append(ys[i])
            frontier_metadata.append(metadata[i])
    return frontier_xs, frontier_ys, frontier_metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark_result")
    parser.add_argument("--output-dir", default="plots")
    parser.add_argument("--output-prefix", default="jaccard")
    parser.add_argument("--min-similarity-at-k", nargs="+", type=float)
    parser.add_argument("--ignore-query", action="store_true")
    parser.add_argument("--annotate-points", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    runs = evaluate_runs(args.benchmark_result, ignore_query=args.ignore_query)
    ks = np.sort(np.unique([run["k"] for run in runs]))
    # Replaces floats with more than 2 decimals, deduplicate and sort.
    min_similarities_at_k = np.sort(
        np.unique([np.round(x, 2) for x in args.min_similarity_at_k])
    )
    # Plot.
    for k in ks:
        # Get algorithms.
        names = np.sort(
            np.unique(
                [
                    run["name"]
                    for run in runs
                    if run["k"] == k and run["name"] != "ground_truth"
                ]
            )
        )
        plot_settings = {}
        for min_similarity in min_similarities_at_k:
            # Compute mean recall, similarities, and times for each run filtered by min_similarity_at_K.
            for run in [
                run for run in runs if run["k"] == k and run["name"] != "ground_truth"
            ]:
                query_keys = set(
                    [
                        query_key
                        for query_key, similarity_at_k in run["similarities_at_k"]
                        if similarity_at_k >= min_similarity
                    ]
                )
                mean_recall = np.mean(
                    [
                        recall
                        for query_key, recall in run["recalls"]
                        if query_key in query_keys and recall is not None
                    ]
                )
                mean_time = np.mean(
                    [
                        time
                        for query_key, time in run["times"]
                        if query_key in query_keys and time is not None
                    ]
                )
                mean_qps = 1.0 / mean_time
                mean_similarity = np.mean(
                    [
                        similarity
                        for query_key, similarity in run["mean_similarities"]
                        if query_key in query_keys and similarity is not None
                    ]
                )
                indexing_time = run["indexing_time"] / 60.0
                run.update(
                    {
                        "mean_recall": mean_recall,
                        "mean_time": mean_time,
                        "mean_similarity": mean_similarity,
                        "mean_qps": mean_qps,
                    }
                )

            # Plot qps vs. recall.
            plt.figure()
            for name in names:
                selected = [
                    run for run in runs if run["k"] == k and run["name"] == name
                ]
                mean_recalls, mean_qps, selected = _frontier_points(
                    [run["mean_recall"] for run in selected],
                    [run["mean_qps"] for run in selected],
                    selected,
                    y_dir="decreasing",
                )
                plt.plot(mean_recalls, mean_qps, "-*", label=name)
                plt.yscale("log")
                if args.annotate_points:
                    _annotate_points(
                        plt,
                        name,
                        mean_recalls,
                        mean_qps,
                        selected,
                    )
            plt.xlim(0.0, 1.0)
            if "qps_vs_recall" in plot_settings:
                plt.xticks(*plot_settings["qps_vs_recall"]["xticks"])
                plt.yticks(*plot_settings["qps_vs_recall"]["yticks"])
                plt.ylim(*plot_settings["qps_vs_recall"]["ylim"])
            else:
                plot_settings["qps_vs_recall"] = {
                    "xticks": plt.xticks(),
                    "yticks": plt.yticks(),
                    "ylim": plt.ylim(),
                }
            plt.xlabel(f"Mean Recall@{k}")
            plt.ylabel("Mean Query Per Second (QPS)")
            plt.title(f"QPS vs. Recall@K (Min Similarity@K = {min_similarity:.2f})")
            plt.grid()
            plt.legend()
            plt.savefig(
                os.path.join(
                    args.output_dir,
                    f"{args.output_prefix}_mean_recall_{min_similarity:.2f}_at_{k}.png",
                ),
                bbox_inches="tight",
            )
            plt.close()

            # Plot similarity vs. time.
            plt.figure()
            for name in names:
                # Get similarities and times of this algorithm.
                selected = [
                    run for run in runs if run["k"] == k and run["name"] == name
                ]
                plt.plot(
                    [run["mean_similarity"] for run in selected],
                    [run["mean_qps"] for run in selected],
                    "*",
                    label=name,
                )
            plt.xlim(0.0, 1.0)
            plt.yscale("log")
            plt.xlabel(f"Mean Similarity@{k}")
            plt.ylabel("Mean Query Per Second (QPS)")
            plt.title(f"QPS vs. Similarity@K (Min Similarity@K = {min_similarity:.2f})")
            plt.grid()
            plt.legend()
            plt.savefig(
                os.path.join(
                    args.output_dir,
                    f"{args.output_prefix}_mean_similarity_{min_similarity:.2f}_at_{k}.png",
                ),
                bbox_inches="tight",
            )
            plt.close()

            # Plot recall vs. indexing time.
            plt.figure()
            for name in names:
                # Get recalls and indexing times of this algorithm.
                selected = [
                    run for run in runs if run["k"] == k and run["name"] == name
                ]
                mean_recalls, indexing_times, selected = _frontier_points(
                    [run["mean_recall"] for run in selected],
                    [run["indexing_time"] for run in selected],
                    selected,
                    y_dir="increasing",
                )
                plt.plot(mean_recalls, indexing_times, "-*", label=name)
                if args.annotate_points:
                    _annotate_points(plt, name, mean_recalls, indexing_times, selected)
            if "indexing_time_vs_recall" in plot_settings:
                plt.xticks(*plot_settings["indexing_time_vs_recall"]["xticks"])
                plt.yticks(*plot_settings["indexing_time_vs_recall"]["yticks"])
                plt.xlim(*plot_settings["indexing_time_vs_recall"]["xlim"])
                plt.ylim(*plot_settings["indexing_time_vs_recall"]["ylim"])
            else:
                plot_settings["indexing_time_vs_recall"] = {
                    "xticks": plt.xticks(),
                    "yticks": plt.yticks(),
                    "ylim": plt.ylim(),
                    "xlim": plt.xlim(),
                }
            plt.xlabel(f"Mean Recall@{k}")
            plt.ylabel("Indexing time (min)")
            plt.title(
                f"Indexing Time vs. Recall@K (Min Similarity@K = {min_similarity:.2f})"
            )
            plt.grid()
            plt.legend()
            plt.savefig(
                os.path.join(
                    args.output_dir,
                    f"{args.output_prefix}_indexing_mean_recall_{min_similarity:.2f}_at_{k}.png",
                ),
                bbox_inches="tight",
            )
            plt.close()
