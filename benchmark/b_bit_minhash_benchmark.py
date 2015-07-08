'''
Benchmarking the performance and accuracy of b-bi MinHash.
'''
import time, logging, random
from hashlib import sha1
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datasketch.minhash import MinHash, jaccard
from datasketch.b_bit_minhash import bBitMinHash

logging.basicConfig(level=logging.INFO)

# Produce some bytes
int_bytes = lambda x : ("a-%d-%d" % (x, x)).encode('utf-8')

def _run_acc(size, seed, num_perm):
    m = MinHash(num_perm=num_perm)
    s = set()
    random.seed(seed)
    for i in range(size):
        v = int_bytes(random.randint(1, size))
        m.digest(sha1(v))
        s.add(v)
    return (m, s)

def run_acc(size, num_perm, bs):
    logging.info("MinHash using %d permutation functions" % num_perm)
    m1, s1 = _run_acc(size, 1, num_perm)
    m2, s2 = _run_acc(size, 4, num_perm)
    j = float(len(s1.intersection(s2)))/float(len(s1.union(s2)))
    j_e = np.array([bBitMinHash(m1, b).jaccard(bBitMinHash(m2, b))
        for b in bs])
    err = np.abs(j_e - j)
    return err

num_perms = range(10, 1024, 10)
bs = [1, 2, 4]
output = "b_bit_minhash_benchmark.png"

logging.info("> Running accuracy tests")
size = 5000
errs = np.array([run_acc(size, n, bs) for n in num_perms]).T

logging.info("> Plotting result")
fig, axe = plt.subplots(1, 1)
ax = axe
for b, err in zip(bs, errs):
    ax.plot(num_perms, err, label='b = %d' % b)
ax.set_xlabel("Number of permutation functions")
ax.set_ylabel("Absolute error in Jaccard estimation")
ax.set_title("b-bit MinHash accuracy")
ax.legend()
ax.grid()

fig.savefig(output)
logging.info("Plot saved to %s" % output)
