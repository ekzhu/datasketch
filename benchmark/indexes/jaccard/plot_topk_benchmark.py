import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from utils import evaluate_runs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark_result")
    parser.add_argument("--output-dir", default="plots")
    parser.add_argument("--output-prefix", default="jaccard")
    parser.add_argument("--min-similarity-at-k", nargs="+", type=float)
    parser.add_argument("--ignore-query", action="store_true")
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
                mean_time = (
                    np.mean(
                        [
                            time
                            for query_key, time in run["times"]
                            if query_key in query_keys and time is not None
                        ]
                    )
                    * 1000.0
                )
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
                    }
                )

            # Plot query time vs. recall.
            plt.figure()
            for name in names:
                # Get recalls and times of this algorithm.
                selected = [
                    run for run in runs if run["k"] == k and run["name"] == name
                ]
                mean_recalls = [run["mean_recall"] for run in selected]
                mean_times = [run["mean_time"] for run in selected]
                plt.scatter(mean_recalls, mean_times, label=name)
                if name == "lsh":
                    for run in selected:
                        x = run["mean_recall"]
                        y = run["mean_time"]
                        b = run["index"]["b"]
                        r = run["index"]["r"]
                        plt.annotate(f"({b}, {r})", (x, y))
            if "query_time_vs_recall" in plot_settings:
                plt.xticks(*plot_settings["query_time_vs_recall"]["xticks"])
                plt.yticks(*plot_settings["query_time_vs_recall"]["yticks"])
                plt.ylim(*plot_settings["query_time_vs_recall"]["ylim"])
                plt.xlim(*plot_settings["query_time_vs_recall"]["xlim"])
            else:
                plot_settings["query_time_vs_recall"] = {
                    "xticks": plt.xticks(),
                    "yticks": plt.yticks(),
                    "ylim": plt.ylim(),
                    "xlim": plt.xlim(),
                }
            plt.xlabel(f"Mean Recall@{k}")
            plt.ylabel("Mean Query Time (ms)")
            plt.title(
                f"Query Time vs. Recall@K (Min Similarity@K = {min_similarity:.2f})"
            )
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
                mean_similarity = [run["mean_similarity"] for run in selected]
                times = [run["mean_time"] for run in selected]
                plt.plot(mean_similarity, times, "*", label=name)
                if name == "lsh":
                    for run in selected:
                        x = run["mean_similarity"]
                        y = run["mean_time"]
                        b = run["index"]["b"]
                        r = run["index"]["r"]
                        plt.annotate(f"({b}, {r})", (x, y))
            plt.xlim(0.0, 1.0)
            if "query_time_vs_similarity" in plot_settings:
                plt.xticks(*plot_settings["query_time_vs_similarity"]["xticks"])
                plt.yticks(*plot_settings["query_time_vs_similarity"]["yticks"])
                plt.ylim(*plot_settings["query_time_vs_similarity"]["ylim"])
            else:
                plot_settings["query_time_vs_similarity"] = {
                    "xticks": plt.xticks(),
                    "yticks": plt.yticks(),
                    "ylim": plt.ylim(),
                }
            plt.xlabel(f"Mean Similarity@{k}")
            plt.ylabel("Mean Query Time (ms)")
            plt.title(
                f"Similarity vs. Recall@K (Min Similarity@K = {min_similarity:.2f})"
            )
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
                mean_recalls = [run["mean_recall"] for run in selected]
                indexing_times = [run["indexing_time"] for run in selected]
                plt.plot(mean_recalls, indexing_times, "o", label=name)
                if name == "lsh":
                    for run in selected:
                        x = run["mean_recall"]
                        y = run["indexing_time"]
                        b = run["index"]["b"]
                        r = run["index"]["r"]
                        plt.annotate(f"({b}, {r})", (x, y))
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
