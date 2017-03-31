import time, argparse, sys, json
from hashlib import sha1
import numpy as np
import nltk
import scipy.stats
import random
import collections
from datasketch import MinHashLSHEnsemble, MinHash
from lshforest_benchmark import bootstrap_data

def benchmark_lshensemble(threshold, num_perm, num_part, l, index_data, query_data):
    print("Building LSH Ensemble index")
    lsh = MinHashLSHEnsemble(threshold=threshold, num_perm=num_perm, num_part=num_part, l=l)
    lsh.index((key, minhash, len(set)) 
                  for key, minhash, set in \
                          zip(index_data.keys, index_data.minhashes[num_perm], index_data.sets))
    print("Querying")
    times = []
    results = []
    for qs, minhash in zip(query_data.sets, query_data.minhashes[num_perm]):
        start = time.clock()
        result = list(lsh.query(minhash, len(qs)))
        duration = time.clock() - start
        times.append(duration)
        results.append(sorted([[key, _compute_containment(qs, index_data.sets[key])]
                               for key in result], 
                              key=lambda x : x[1], reverse=True))
    return times, results


def benchmark_ground_truth(threshold, index_data, query_data):
    times = []
    results = []
    for q in query_data.sets:
        start = time.clock()
        result = [key for key, a in zip(index_data.keys, index_data.sets)
                  if _compute_containment(q, a) >= threshold]
        duration = time.clock() - start
        times.append(duration)
        results.append(sorted([[key, _compute_containment(q, index_data.sets[key])]
                               for key in result], 
                              key=lambda x : x[1], reverse=True))
    return times, results


def _compute_containment(x, y):
    if len(x) == 0 or len(y) == 0:
        return 0.0
    intersection = 0
    for w in x:
        if w in y:
            intersection += 1
    return float(intersection) / float(len(x))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="lshensemble_benchmark.json")
    args = parser.parse_args(sys.argv[1:])

    threshold = 0.5
    num_perms = [32, 64, 96, 128, 160, 192, 224, 256]
    num_part = 16
    l = 8
    output = {"threshold" : threshold,
              "num_perms" : num_perms,
              "num_part" : 16,
              "l" : l,
              "lsh_times" : [], "lsh_results" : [],
              "ground_truth_times" : None, "ground_truth_results" : None}

    population_size = 500
    
    class zipfian:
        def __init__(self):
            self.rv = scipy.stats.zipf(1.25)
        def rvs(self):
            x = int(self.rv.rvs())
            if x > population_size:
                return population_size
            return x

    index_data, query_data = bootstrap_data(num_perms, 100, population_size, zipfian())

    for num_perm in num_perms:
        print("Use num_perm = %d" % num_perm)
        result = {}
        print("Running LSH Ensemble benchmark")
        lsh_times, lsh_results = benchmark_lshensemble(threshold, num_perm, num_part, l, index_data, query_data)
        output["lsh_times"].append(lsh_times)
        output["lsh_results"].append(lsh_results)
    
    print("Running ground truth benchmark")
    output["ground_truth_times"], output["ground_truth_results"] =\
            benchmark_ground_truth(threshold, index_data, query_data)

    average_cardinality = np.mean([len(s) for s in
        index_data.sets + query_data.sets])
    print("Average cardinality is", average_cardinality)

    with open(args.output, 'w') as f:
        json.dump(output, f)
