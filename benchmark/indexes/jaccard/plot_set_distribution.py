import argparse
import os
from matplotlib import pyplot as plt
import numpy as np

import numpy as np
from utils import read_set_sizes_from_file

parser = argparse.ArgumentParser()
parser.add_argument(
    "sets_files", nargs="+", help="Ihe input files for reading sets from."
)
parser.add_argument("--output-dir", default=".", help="The output directory.")
args = parser.parse_args()

plt.figure()
for sets_file in args.sets_files:
    name = os.path.splitext(os.path.basename(sets_file))[0]
    print(f"Processing {sets_file}...")
    set_sizes = read_set_sizes_from_file(sets_file)
    # Plot the distribution of set sizes histogram.
    h, bins, patches = plt.hist(
        set_sizes,
        bins=np.logspace(np.log10(np.min(set_sizes)), np.log10(np.max(set_sizes)), 30),
        alpha=0.5,
        label=name,
    )
    color = patches[0].get_facecolor()
    mean = np.mean(set_sizes)
    plt.axvline(mean, color=color, linestyle="dashed", linewidth=1)
    plt.text(
        mean,
        0.5 * np.max(h),
        f"Mean Size = {mean:.2f}",
        color="k",
    )
plt.xscale("log")
# plt.yscale("log")
plt.legend()
plt.xlabel("Set size")
plt.ylabel("Number of sets")
plt.title("Distribution of Set Sizes")
plt.savefig(
    os.path.join(args.output_dir, "set_size_distribution.png"), bbox_inches="tight"
)
