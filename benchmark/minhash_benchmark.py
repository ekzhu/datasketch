'''
Benchmarking the performance and accuracy of MinHash.
'''
import time, logging, random
from hashlib import sha1
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datasketch.minhash import MinHash

logging.basicConfig(level=logging.INFO)

# Produce some bytes
int_bytes = lambda x : ("a-%d-%d" % (x, x)).encode('utf-8')

def run_perf(card, num_perm):
    m = MinHash(num_perm=num_perm)
    logging.info("MinHash using %d permutation functions" % num_perm)
    start = time.clock()
    for i in range(card):
        m.update(int_bytes(i))
    duration = time.clock() - start
    logging.info("Digested %d hashes in %.4f sec" % (card, duration))
    return duration


def _run_acc(size, seed, num_perm):
    m = MinHash(num_perm=num_perm)
    s = set()
    random.seed(seed)
    for i in range(size):
        v = int_bytes(random.randint(1, size))
        m.update(v)
        s.add(v)
    return (m, s)

def run_acc(size, num_perm):
    logging.info("MinHash using %d permutation functions" % num_perm)
    m1, s1 = _run_acc(size, 1, num_perm)
    m2, s2 = _run_acc(size, 4, num_perm)
    j = float(len(s1.intersection(s2)))/float(len(s1.union(s2)))
    j_e = m1.jaccard(m2)
    err = abs(j - j_e)
    return err

num_perms = range(10, 256, 20)
output = "minhash_benchmark.png"

logging.info("> Running performance tests")
card = 5000
run_times = [run_perf(card, n) for n in num_perms]

logging.info("> Running accuracy tests")
size = 5000
errs = [run_acc(size, n) for n in num_perms]

logging.info("> Plotting result")
fig, axe = plt.subplots(1, 2, sharex=True, figsize=(10, 4))
ax = axe[1]
ax.plot(num_perms, run_times, marker='+')
ax.set_xlabel("Number of permutation functions")
ax.set_ylabel("Running time (sec)")
ax.set_title("MinHash performance")
ax.grid()
ax = axe[0]
ax.plot(num_perms, errs, marker='+')
ax.set_xlabel("Number of permutation functions")
ax.set_ylabel("Absolute error in Jaccard estimation")
ax.set_title("MinHash accuracy")
ax.grid()

fig.savefig(output)
logging.info("Plot saved to %s" % output)
