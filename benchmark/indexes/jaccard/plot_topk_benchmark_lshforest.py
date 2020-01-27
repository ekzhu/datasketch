import argparse
import sys
import json
import sqlite3
import collections

import matplotlib.pyplot as plt
import numpy as np

from utils import load_results, compute_recalls

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Plot MinHash LSH Forest topk benchmark results.")
    parser.add_argument("--benchmark-result", required=True)
    parser.add_argument("--plot-filename-prefix", 
            default="jaccard_topk_lshforest")
    args = parser.parse_args(sys.argv[1:])    

    conn = sqlite3.connect(args.benchmark_result)

    # Get runs.
    cursor = conn.cursor()
    cursor.execute("""SELECT key, name, k, params
            FROM runs WHERE name in ('lshforest', 'ground_truth') 
            ORDER BY k, name""")
    runs = cursor.fetchall()
    cursor.close()

    # Group runs by k and then by names.
    groups = collections.defaultdict(lambda: collections.defaultdict(list))
    for run in runs:
        (key, name, k, params) = run
        groups[k][name].append([key, params])

    # Compute the mean recall and query time of every run, and correspond those
    # with the index parameters.
    points = collections.defaultdict(lambda: collections.defaultdict(list))
    for k in groups:
        ground_run_key, _ = groups[k]["ground_truth"][0]
        ground_results, ground_times = load_results(ground_run_key, conn)
        for run_key, params in groups[k]['lshforest']:
            results, times = load_results(run_key, conn)
            recalls = compute_recalls(results, ground_results)
            mean_recall = np.mean([x[1] for x in recalls])
            mean_time = np.mean(times * 1000)
            # Extract LSH Forest parameters.
            params = json.loads(params)
            num_perm = params["num_perm"]
            l = params["l"]
            r = num_perm / l
            points[k]['lshforest'].append(
                    [mean_recall, mean_time, num_perm, l, r])
    
    for k in points:
        ls = sorted(list(np.unique([p[3] for p in points[k]['lshforest']])))
        rs = sorted(list(np.unique([p[4] for p in points[k]['lshforest']])))
        
        # Plot time vs. num perm. 
        plt.figure()
        for l in ls:
            # Get points given l sorted by num_perm.
            ps = np.array(sorted([[p[2], p[1]]
                    for p in points[k]['lshforest'] if p[3] == l]))
            plt.plot(*ps.T, "-x", label=f"l = {l}")
        plt.xlabel("num_perm")
        plt.ylabel("Time (ms)")
        plt.grid()
        plt.legend()
        plt.savefig(f"{args.plot_filename_prefix}_{k}_times_vs_num_perm.png")
        plt.close()

        # Plot recall vs. num perm.
        plt.figure()
        for l in ls:
            # Get points given l sorted by num_perm.
            ps = np.array(sorted([[p[2], p[0]]
                    for p in points[k]['lshforest'] if p[3] == l]))
            plt.plot(*ps.T, "-x", label=f"l = {l}")
        plt.xlabel("num_perm")
        plt.ylabel(f"Recall @ {k}")
        plt.grid()
        plt.legend()
        plt.savefig(f"{args.plot_filename_prefix}_{k}_recall_vs_num_perm.png")
        plt.close()

        # Plot recall vs. r.
        plt.figure()
        for l in ls:
            # Get points given l sorted by r.
            ps = np.array(sorted([[p[4], p[0]]
                    for p in points[k]['lshforest'] if p[3] == l]))
            plt.plot(*ps.T, "-x", label=f"l = {l}")
        plt.xlabel("r")
        plt.ylabel(f"Recall @ {k}")
        plt.grid()
        plt.legend()
        plt.savefig(f"{args.plot_filename_prefix}_{k}_recall_vs_r.png")
        plt.close()

        # Plot recall vs. l.
        plt.figure()
        for r in rs:
            # Get points given r sorted by l.
            ps = np.array(sorted([[p[3], p[0]]
                    for p in points[k]['lshforest'] if p[4] == r]))
            plt.plot(*ps.T, "-x", label=f"r = {r}")
        plt.xlabel("l")
        plt.ylabel(f"Recall @ {k}")
        plt.grid()
        plt.legend()
        plt.savefig(f"{args.plot_filename_prefix}_{k}_recall_vs_l.png")
        plt.close()
