'''
Benchmarking the performance and accuracy of WeightedMinHash.
'''
import time, logging, random
from hashlib import sha1
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from datasketch import WeightedMinHashGenerator

logging.basicConfig(level=logging.INFO)

def run_perf(dim, num_rep, sample_size):
    wmg = WeightedMinHashGenerator(dim, sample_size=sample_size)
    logging.info("WeightedMinHash using %d samples" % sample_size)
    data = np.random.uniform(0, dim, (num_rep, dim)) 
    durs = []
    for i in range(num_rep):
        start = time.clock()
        wmg.minhash(data[i])
        duration = (time.clock() - start) * 1000
        durs.append(duration)
    ave = np.mean(durs)
    logging.info("Generated %d minhashes, average time %.4f ms" % (num_rep, ave))
    return ave

def jaccard(v1, v2):
    min_sum = np.sum(np.minimum(v1, v2))
    max_sum = np.sum(np.maximum(v1, v2))
    return float(min_sum) / float(max_sum)

def run_acc(dim, num_rep, sample_size):
    logging.info("WeightedMinHash using %d samples" % sample_size)
    wmg = WeightedMinHashGenerator(dim, sample_size=sample_size)
    data1 = np.random.uniform(0, dim, (num_rep, dim)) 
    data2 = np.random.uniform(0, dim, (num_rep, dim)) 
    errs = []
    for i in range(num_rep):
        wm1 = wmg.minhash(data1[i])
        wm2 = wmg.minhash(data2[i])
        j_e = wm1.jaccard(wm2)
        j = jaccard(data1[i], data2[i])
        errs.append(abs(j - j_e))
    ave = np.mean(errs)
    logging.info("%d runs, mean error %.4f" % (num_rep, ave))
    return ave

sample_sizes = range(10, 160, 10)
num_rep = 100
dim = 5000
output = "weighted_minhash_benchmark.png"

logging.info("> Running performance tests")
run_times = [run_perf(dim, num_rep, n) for n in sample_sizes]

logging.info("> Running accuracy tests")
errs = [run_acc(dim, num_rep, n) for n in sample_sizes]

logging.info("> Plotting result")
fig, axe = plt.subplots(1, 2, sharex=True, figsize=(10, 4))
ax = axe[1]
ax.plot(sample_sizes, run_times, marker='+')
ax.set_xlabel("Number of samples")
ax.set_ylabel("Running time (ms)")
ax.set_title("WeightedMinHash performance")
ax.grid()
ax = axe[0]
ax.plot(sample_sizes, errs, marker='+')
ax.set_xlabel("Number of samples")
ax.set_ylabel("Absolute error in Jaccard estimation")
ax.set_title("WeightedMinHash accuracy")
ax.grid()

fig.savefig(output, bbox_inches="tight")
logging.info("Plot saved to %s" % output)


