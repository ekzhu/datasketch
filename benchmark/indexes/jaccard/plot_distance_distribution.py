import json
import os
import argparse
import sqlite3

import matplotlib.pyplot as plt
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("benchmark_result", nargs="+", type=str)
parser.add_argument("--k", type=int, required=False)
parser.add_argument("--output-dir", default="plots")
args = parser.parse_args()

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

# Obtain ground truth runs from the benchmark results.
quartiles_at_k = {}
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
    distances = []
    cursor.execute("""SELECT result FROM results WHERE run_key = ?""", (run_keys[0],))
    for row in cursor:
        result = json.loads(row[0])
        if args.k is not None:
            result = result[: args.k]
        distances.append([1.0 - similarity for _, similarity in result])

    # Pad similarities with zeros.
    max_length = max([len(x) for x in distances])
    distances = np.array(
        [
            np.pad(x, (0, max_length - len(x)), "constant", constant_values=1.0)
            for x in distances
        ]
    )

    # Compute quartiles at k.
    name = os.path.splitext(os.path.basename(benchmark_result))[0]
    pcts = [25, 50, 75]
    quartiles_at_k[name] = (pcts, np.percentile(distances, pcts, axis=0))


# Plot.
if args.k is not None:
    filename = f"jaccard_distances_at_k_k={args.k}.png"
else:
    filename = "jaccard_distances_at_k.png"
plt.figure()
for name, (pcts, sims) in quartiles_at_k.items():
    lower, upper = sims[0], sims[-1]
    xs = np.arange(1, len(lower) + 1)
    plt.fill_between(xs, lower, upper, alpha=0.2, label=f"{name} (25-75%)")
    plt.plot(xs, sims[1], label=f"{name} (50%)")
plt.xlabel("K")
plt.ylabel("Median Jaccard Distance at K")
plt.title("Jaccard Distances at K")
plt.legend()
plt.savefig(os.path.join(args.output_dir, filename))
plt.close()
