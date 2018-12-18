import time, argparse, sys, json
import numpy as np
import scipy.stats
import random
import collections
import gzip
import random
from datasketch import MinHashLSHEnsemble, MinHash


def bootstrap_sets(sets_file, sample_ratio, num_perms):
    print("Creating sets...")
    sets = collections.deque([])
    with gzip.open(sets_file, "rt") as f:
        for i, line in enumerate(f):
            if i == 0:
                # Skip first line
                continue
            if random.random() > sample_ratio:
                continue
            s = np.array(line.strip().split("\t")[1].split(","))
            sets.append(s)
            sys.stdout.write("\rRead {} sets".format(len(sets)))
        sys.stdout.write("\n")
    keys = list(range(len(sets)))
    print("Creating MinHash...")
    minhashes = dict()
    for num_perm in num_perms:
        print("Using num_parm = {}".format(num_perm))
        ms = []
        for s in sets:
            m = MinHash(num_perm)
            for word in s:
                m.update(word.encode("utf8"))
            ms.append(m)
            sys.stdout.write("\rMinhashed {} sets".format(len(ms)))
        sys.stdout.write("\n")
        minhashes[num_perm] = ms
    Data = collections.namedtuple('Data', ['minhashes', 'sets', 'keys'])
    index_data = Data(minhashes, sets, keys)
    query_indices = random.sample(list(range(len(sets))), int(len(sets)*0.1))
    if len(query_indices) == 0:
        raise RuntimeError("Empty query sets")
    query_data = Data(dict((num_perm, [minhashes[num_perm][i] for i in query_indices])
                           for num_perm in num_perms),
                      [sets[i] for i in query_indices],
                      [keys[i] for i in query_indices])
    return index_data, query_data


def benchmark_lshensemble(threshold, num_perm, num_part, m, index_data,
        query_data):
    print("Building LSH Ensemble index")
    lsh = MinHashLSHEnsemble(threshold=threshold, num_perm=num_perm,
            num_part=num_part, m=m)
    lsh.index((key, minhash, len(set))
                  for key, minhash, set in \
                          zip(index_data.keys, index_data.minhashes[num_perm],
                              index_data.sets))
    print("Querying")
    times = []
    results = []
    for qs, minhash in zip(query_data.sets, query_data.minhashes[num_perm]):
        start = time.perf_counter()
        result = list(lsh.query(minhash, len(qs)))
        duration = time.perf_counter() - start
        times.append(duration)
        results.append(sorted([[key, _compute_containment(qs, index_data.sets[key])]
                               for key in result],
                              key=lambda x : x[1], reverse=True))
    return times, results


def benchmark_ground_truth(threshold, index_data, query_data):
    times = []
    results = []
    for q in query_data.sets:
        start = time.perf_counter()
        result = [key for key, a in zip(index_data.keys, index_data.sets)
                  if _compute_containment(q, a) >= threshold]
        duration = time.perf_counter() - start
        times.append(duration)
        results.append(sorted([[key, _compute_containment(q, index_data.sets[key])]
                               for key in result],
                              key=lambda x : x[1], reverse=True))
    return times, results


def _compute_containment(x, y):
    if len(x) == 0 or len(y) == 0:
        return 0.0
    intersection = len(np.intersect1d(x, y, assume_unique=True))
    return float(intersection) / float(len(x))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True,
            help="Input set file (gzipped), each line is a set: "
            "<set_size> <1>,<2>,<3>..., where each <?> is an element.")
    parser.add_argument("--output", type=str, default="lshensemble_benchmark.json")
    args = parser.parse_args(sys.argv[1:])

    threshold = 0.5
    num_parts = [8, 12, 16]
    #num_perms = [32, 64, 96, 128, 160, 192, 224, 256]
    num_perms = [32, 64, 128, 256]
    m = 8
    output = {"threshold" : threshold,
              "num_parts" : num_parts,
              "num_perms" : num_perms,
              "m" : m,
              "lsh_times" : [], "lsh_results" : [],
              "ground_truth_times" : None, "ground_truth_results" : None}

    class zipfian:
        def __init__(self):
            self.rv = scipy.stats.zipf(1.25)
        def rvs(self):
            x = int(self.rv.rvs())
            if x > population_size:
                return population_size
            return x

    index_data, query_data = bootstrap_sets(args.input, 0.01, num_perms)

    for num_part in num_parts:
        print("Use num_part = {}".format(num_part))
        times = []
        results = []
        for num_perm in num_perms:
            print("Use num_perm = %d" % num_perm)
            print("Running LSH Ensemble benchmark")
            lsh_times, lsh_results = benchmark_lshensemble(
                    threshold, num_perm, num_part, m, index_data, query_data)
            times.append(lsh_times)
            results.append(lsh_results)
        output["lsh_times"].append(times)
        output["lsh_results"].append(results)

    print("Running ground truth benchmark")
    output["ground_truth_times"], output["ground_truth_results"] =\
            benchmark_ground_truth(threshold, index_data, query_data)

    average_cardinality = np.mean([len(s) for s in
        index_data.sets + query_data.sets])
    print("Average cardinality is", average_cardinality)

    with open(args.output, 'w') as f:
        json.dump(output, f)
