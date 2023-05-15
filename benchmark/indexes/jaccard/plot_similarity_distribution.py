import json
import os
import argparse
import sqlite3

import matplotlib.pyplot as plt
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("benchmark_result", nargs="+", type=str)
parser.add_argument("--output-dir", default="plots")
args = parser.parse_args()

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

# Obtain ground truth runs from the benchmark results.
median_similarities_at_k = {}
for benchmark_result in args.benchmark_result:
    conn = sqlite3.connect(benchmark_result)
    cursor = conn.cursor()
    cursor.execute(
        """SELECT key
            FROM runs 
            WHERE name == 'ground_truth'""",
    )
    run_keys = [row[0] for row in cursor]

    # Load results for the first run.
    similarities = []
    cursor.execute("""SELECT result FROM results WHERE run_key = ?""", (run_keys[0],))
    for row in cursor:
        result = json.loads(row[0])
        similarities.append([similarity for _, similarity in result])

    # Pad similarities with zeros.
    max_length = max([len(x) for x in similarities])
    similarities = np.array(
        [np.pad(x, (0, max_length - len(x)), "constant") for x in similarities]
    )

    # Compute median similarities at k.
    name = os.path.basename(benchmark_result).rstrip(".sqlite")
    median_similarities_at_k[name] = np.median(similarities, axis=0)

# Plot.
plt.figure()
for name, ys in median_similarities_at_k.items():
    xs = np.arange(1, len(ys) + 1)
    plt.plot(xs, ys, label=name)
plt.xlabel("K")
plt.ylabel("Median Jaccard Similarity at K")
plt.title("Median Jaccard Similarities at K")
plt.legend()
plt.savefig(os.path.join(args.output_dir, "median_jaccard_similarities_at_k.png"))
plt.close()
