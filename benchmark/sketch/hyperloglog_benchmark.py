'''
Performance and accuracy of HyperLogLog
'''
import time, logging, random
from datasketch.hyperloglog import HyperLogLog

logging.basicConfig(level=logging.INFO)

# Produce some bytes
int_bytes = lambda x : ("a-%d-%d" % (x, x)).encode('utf-8')

def run_perf(card, p):
    h = HyperLogLog(p=p)
    logging.info("HyperLogLog using p = %d " % p)
    start = time.clock()
    for i in range(card):
        h.update(int_bytes(i))
    duration = time.clock() - start
    logging.info("Digested %d hashes in %.4f sec" % (card, duration))
    return duration


def run_acc(size, seed, p):
    logging.info("HyperLogLog using p = %d " % p)
    h = HyperLogLog(p=p)
    s = set()
    random.seed(seed)
    for i in range(size):
        v = int_bytes(random.randint(1, size))
        h.update(v)
        s.add(v)
    perr = abs(float(len(s)) - h.count()) / float(len(s))
    return perr

ps = range(4, 17)
output = "hyperloglog_benchmark.png"

logging.info("> Running performance tests")
card = 5000
run_times = [run_perf(card, p) for p in ps]

logging.info("> Running accuracy tests")
size = 5000
errs = [run_acc(size, 1, p) for p in ps]

logging.info("> Plotting result")
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, axe = plt.subplots(1, 2, sharex=True, figsize=(10, 4))
ax = axe[1]
ax.plot(ps, run_times, marker='+')
ax.set_xlabel("P values")
ax.set_ylabel("Running time (sec)")
ax.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
ax.set_title("HyperLogLog performance")
ax.grid()
ax = axe[0]
ax.plot(ps, errs, marker='+')
ax.set_xlabel("P values")
ax.set_ylabel("Error rate in cardinality estimation")
ax.set_title("HyperLogLog accuracy")
ax.grid()

fig.savefig(output)
logging.info("Plot saved to %s" % output)
