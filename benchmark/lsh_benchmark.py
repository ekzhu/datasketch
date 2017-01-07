import time, argparse, sys, json
from hashlib import sha1
from sklearn.datasets import fetch_20newsgroups
import numpy as np
import scipy.stats
from datasketch import MinHashLSH, MinHash
from lshforest_benchmark import bootstrap_data, _compute_jaccard


def benchmark_lsh(num_perm, threshold, index_data, query_data):
    print("Building LSH index")
    lsh = MinHashLSH(threshold, num_perm)
    for key, minhash in z:xip(index_data.keys, index_data.minhashes[num_perm]):
        lsh.insert(key, minhash)
    print("Querying")
    times = []
    results = []
    for qs, minhash in zip(query_data.sets, query_data.minhashes[num_perm]):
        start = time.clock()
        result = lsh.query(minhash)
        duration = time.clock() - start
        times.append(duration)
        results.append(sorted([[key, _compute_jaccard(qs, index_data.sets[key])]
                               for key in result], 
                              key=lambda x : x[1], reverse=True))
    return times, results


def benchmark_linearscan(num_perm, threshold, index_data, query_data):
    times = []
    results = []
    for qs, q in zip(query_data.sets, query_data.minhashes[num_perm]):
        start = time.clock()
        result = []
        for key, m in zip(index_data.keys, index_data.minhashes[num_perm]):
            j = q.jaccard(m)
            if j >= threshold:
                result.append(key)
        duration = time.clock() - start
        times.append(duration)
        results.append(sorted([[key, _compute_jaccard(qs, index_data.sets[key])]
                               for key in result], 
                              key=lambda x : x[1], reverse=True))
    return times, results


def benchmark_ground_truth(threshold, index_data, query_data):
    times = []
    results = []
    for q in query_data.sets:
        start = time.clock()
        result = []
        for key, a in zip(index_data.keys, index_data.sets):
            j = _compute_jaccard(q, a)
            if j >= threshold:
                result.append([key, j])
        duration = time.clock() - start
        results.append(sorted(result, key=lambda x : x[1], reverse=True))
        times.append(duration)
    return times, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="lsh_benchmark.json")
    args = parser.parse_args(sys.argv[1:])

    num_perms = [32, 64, 96, 128, 160, 192, 224, 256]
    output = {"num_perms" : num_perms,
              "lsh_times" : [], "lsh_results" : [],
              "linearscan_times" : [], "linearscan_results" : [],
              "ground_truth_times" : None, "ground_truth_results" : None}

    index_data, query_data = bootstrap_data(num_perms, 1000, 500, 
             scipy.stats.(10, 500))

    threshold = 0.9

    for num_perm in num_perms:
        print("Use num_perm = %d" % num_perm)
        result = {}
        print("Running linear scan benchmark")
        linearscan_times, linearscan_results = benchmark_linearscan(num_perm, threshold, index_data, query_data)
        print("Running LSH benchmark")
        lsh_times, lsh_results = benchmark_lsh(num_perm, threshold, index_data, query_data)
        output["lsh_times"].append(lsh_times)
        output["lsh_results"].append(lsh_results)
        output["linearscan_times"].append(linearscan_times)
        output["linearscan_results"].append(linearscan_results)

    print("Running ground truth benchmark")
    output["ground_truth_times"], output["ground_truth_results"] =\
            benchmark_ground_truth(threshold, index_data, query_data)

    average_cardinality = np.mean([len(s) for s in
        index_data.sets + query_data.sets])
    print("Average cardinality is", average_cardinality)

    with open(args.output, 'w') as f:
        json.dump(output, f)
