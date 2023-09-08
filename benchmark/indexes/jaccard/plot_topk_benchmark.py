import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from utils import evaluate_runs


def _annotate_points(plt, name, xs, ys, runs):
    for x, y, run in zip(xs, ys, runs):
        if name == "lsh" or name == "lshforest":
            text = f"({run['index']['b']}, {run['index']['r']})"
        # elif name == "hnsw":
        #     text = f"({run['index']['M']}, {run['index']['efConstruction']})"
        elif name == "exact":
            text = "Inverted Index"
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
    parser.add_argument("--max-distance-at-k", nargs="+", type=float)
    parser.add_argument("--annotate-points", action="store_true")
    parser.add_argument("--names", nargs="+", type=str)
    args = parser.parse_args(sys.argv[1:])
    runs = evaluate_runs(args.benchmark_result, ignore_query=True)
    ks = np.sort(np.unique([run["query"]["k"] for run in runs]))
    # Replaces floats with more than 2 decimals, deduplicate and sort.
    max_distances_at_k = np.sort(
        np.unique([np.round(x, 2) for x in args.max_distance_at_k])
    )
    result_name = os.path.splitext(os.path.basename(args.benchmark_result))[0]
    for k in ks:
        output_dir = os.path.join(args.output_dir, f"k{k}")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # Get algorithms.
        if args.names is None or len(args.names) == 0:
            names = np.sort(
                np.unique([run["name"] for run in runs if run["query"]["k"] == k])
            )
        else:
            names = np.sort(args.names)
        plot_settings = {}
        for max_distance in max_distances_at_k:
            for run in [run for run in runs if run["query"]["k"] == k]:
                query_keys = set(
                    [
                        query_key
                        for query_key, distance_at_k in run["distances_at_k"]
                        if distance_at_k <= max_distance
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
                mean_distance = np.mean(
                    [
                        distance
                        for query_key, distance in run["mean_distances"]
                        if query_key in query_keys and distance is not None
                    ]
                )
                mean_count = np.mean(
                    [
                        count
                        for query_key, count in run["counts"]
                        if query_key in query_keys and count is not None
                    ]
                )
                run.update(
                    {
                        "mean_recall": mean_recall,
                        "mean_time": mean_time,
                        "mean_distance": mean_distance,
                        "mean_qps": mean_qps,
                        "mean_count": mean_count,
                    }
                )

            plt.figure()
            for name in names:
                selected = [
                    run
                    for run in runs
                    if run["query"]["k"] == k and run["name"] == name
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
            plt.title(
                f"QPS vs. Recall@K (MaxDistance@K = {max_distance:.2f}, K = {k}, {result_name})"
            )
            plt.grid()
            plt.legend()
            plt.savefig(
                os.path.join(
                    output_dir,
                    f"{result_name}_qps_recall_{max_distance:.2f}.png",
                ),
                bbox_inches="tight",
            )
            plt.close()

            plt.figure()
            for name in names:
                selected = [
                    run
                    for run in runs
                    if run["query"]["k"] == k and run["name"] == name
                ]
                plt.plot(
                    [run["mean_distance"] for run in selected],
                    [run["mean_qps"] for run in selected],
                    "*",
                    label=name,
                )
            plt.xlim(0.0, 1.0)
            plt.yscale("log")
            plt.xlabel(f"Mean Distance@{k}")
            plt.ylabel("Mean Query Per Second (QPS)")
            plt.title(
                f"QPS vs. Distance@K (MaxDistance@K = {max_distance:.2f}, K = {k}, {result_name})"
            )
            plt.grid()
            plt.legend()
            plt.savefig(
                os.path.join(
                    output_dir,
                    f"{result_name}_qps_distance_{max_distance:.2f}.png",
                ),
                bbox_inches="tight",
            )
            plt.close()

            plt.figure()
            for name in names:
                # Get recalls and indexing times of this algorithm.
                selected = [
                    run
                    for run in runs
                    if run["query"]["k"] == k and run["name"] == name
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
            plt.yscale("log")
            plt.xlabel(f"Mean Recall@{k}")
            plt.ylabel("Indexing time (s)")
            plt.title(
                f"Indexing Time vs. Recall@K (MaxDistance@K = {max_distance:.2f}, K = {k}, {result_name})"
            )
            plt.grid()
            plt.legend()
            plt.savefig(
                os.path.join(
                    output_dir,
                    f"{result_name}_indexing_recall_{max_distance:.2f}.png",
                ),
                bbox_inches="tight",
            )
            plt.close()

            plt.figure()
            all_xs = []
            all_ys = []
            for name in names:
                # Get recalls, qps, and distances of this algorithm.
                selected = [
                    run
                    for run in runs
                    if run["query"]["k"] == k and run["name"] == name
                ]
                mean_recalls = np.array([run["mean_recall"] for run in selected])
                mean_qps = np.array([run["mean_qps"] for run in selected])
                mean_distances = np.array([run["mean_distance"] for run in selected])
                xs = (1.0 - mean_distances) - mean_recalls
                ys = mean_qps
                all_xs.extend(xs)
                all_ys.extend(ys)
                plt.scatter(xs, ys, label=name)
                if args.annotate_points:
                    _annotate_points(plt, name, xs, ys, selected)
            plt.yscale("log")
            plt.axhline(np.median(all_ys), color="black")
            plt.axvline(np.median(all_xs), color="black")
            plt.grid()
            plt.legend()
            plt.ylabel("Mean Query Per Second (QPS)")
            plt.xlabel("(1 - Mean Distance@K) - Mean Recall@K")
            plt.savefig(
                os.path.join(
                    output_dir,
                    f"{result_name}_qps_recall_relevance_{max_distance:.2f}.png",
                ),
                bbox_inches="tight",
            )
            plt.close()

            plt.figure()
            all_ys = []
            all_xs = []
            for name in names:
                # Get recalls, indexing time, and distances of this algorithm.
                selected = [
                    run
                    for run in runs
                    if run["query"]["k"] == k and run["name"] == name
                ]
                mean_recalls = np.array([run["mean_recall"] for run in selected])
                indexing_times = np.array([run["indexing_time"] for run in selected])
                mean_distances = np.array([run["mean_distance"] for run in selected])
                xs = (1.0 - mean_distances) - mean_recalls
                ys = indexing_times
                all_ys.extend(ys)
                all_xs.extend(xs)
                plt.scatter(xs, ys, label=name)
                if args.annotate_points:
                    _annotate_points(plt, name, xs, ys, selected)
            plt.yscale("log")
            plt.axhline(np.median(all_ys), color="black")
            plt.axvline(np.median(all_xs), color="black")
            plt.grid()
            plt.legend()
            plt.xlabel("(1 - Mean Distance@K) - Mean Recall@K")
            plt.ylabel("Indexing time (s)")
            plt.savefig(
                os.path.join(
                    output_dir,
                    f"{result_name}_indexing_recall_relevance_{max_distance:.2f}.png",
                ),
                bbox_inches="tight",
            )
            plt.close()

            plt.figure()
            for name in names:
                selected = [
                    run
                    for run in runs
                    if run["query"]["k"] == k and run["name"] == name
                ]
                mean_recalls = [run["mean_recall"] for run in selected]
                mean_counts = [run["mean_count"] for run in selected]
                plt.plot(mean_counts, mean_recalls, "*", label=name)
                if args.annotate_points:
                    _annotate_points(plt, name, mean_counts, mean_recalls, selected)
            plt.xlabel(f"Mean ResultSize@{k}")
            plt.ylabel(f"Mean Recall@{k}")
            plt.grid()
            plt.legend()
            plt.title(
                f"ResultSize@K vs. Recall@K (MaxDistance@K = {max_distance:.2f}, K = {k}, {result_name})"
            )
            plt.savefig(
                os.path.join(
                    output_dir,
                    f"{result_name}_count_recall_{max_distance:.2f}.png",
                ),
                bbox_inches="tight",
            )
            plt.close()

            plt.figure()
            for name in names:
                selected = [
                    run
                    for run in runs
                    if run["query"]["k"] == k and run["name"] == name
                ]
                mean_distances = [run["mean_distance"] for run in selected]
                mean_counts = [run["mean_count"] for run in selected]
                plt.plot(mean_counts, mean_distances, "*", label=name)
                if args.annotate_points:
                    _annotate_points(plt, name, mean_counts, mean_distances, selected)
            plt.xlabel(f"Mean ResultSize@{k}")
            plt.ylabel(f"Mean Distance@{k}")
            plt.grid()
            plt.legend()
            plt.title(
                f"ResultSize@K vs. Distance@K (MaxDistance@K = {max_distance:.2f}, K = {k}, {result_name})"
            )
            plt.savefig(
                os.path.join(
                    output_dir,
                    f"{result_name}_count_distance_{max_distance:.2f}.png",
                ),
                bbox_inches="tight",
            )
            plt.close()

            plt.figure()
            for name in names:
                selected = [
                    run
                    for run in runs
                    if run["query"]["k"] == k and run["name"] == name
                ]
                mean_qps = [run["mean_qps"] for run in selected]
                mean_counts = [run["mean_count"] for run in selected]
                plt.plot(mean_counts, mean_qps, "*", label=name)
                if args.annotate_points:
                    _annotate_points(plt, name, mean_counts, mean_qps, selected)
            plt.xlabel(f"Mean ResultSize@{k}")
            plt.ylabel(f"Mean QPS")
            plt.grid()
            plt.legend()
            plt.title(
                f"ResultSize@K vs. QPS (MaxDistance@K = {max_distance:.2f}, K = {k}, {result_name})"
            )
            plt.savefig(
                os.path.join(
                    output_dir,
                    f"{result_name}_count_qps_{max_distance:.2f}.png",
                ),
                bbox_inches="tight",
            )
            plt.close()

            plt.figure()
            for name in names:
                selected = [
                    run
                    for run in runs
                    if run["query"]["k"] == k and run["name"] == name
                ]
                indexing_times = [run["indexing_time"] for run in selected]
                mean_counts = [run["mean_count"] for run in selected]
                plt.plot(mean_counts, indexing_times, "*", label=name)
                if args.annotate_points:
                    _annotate_points(plt, name, mean_counts, indexing_times, selected)
            plt.xlabel(f"Mean ResultSize@{k}")
            plt.ylabel(f"Indexing Time (s)")
            plt.grid()
            plt.legend()
            plt.title(
                f"ResultSize@K vs. Indexing Time (MaxDistance@K = {max_distance:.2f}, K = {k}, {result_name})"
            )
            plt.savefig(
                os.path.join(
                    output_dir,
                    f"{result_name}_count_indexing_{max_distance:.2f}.png",
                ),
                bbox_inches="tight",
            )
            plt.close()
