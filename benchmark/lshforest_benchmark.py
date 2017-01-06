import time, argparse, sys, json
from hashlib import sha1
from sklearn.datasets import fetch_20newsgroups
import numpy as np
from datasketch import MinHashLSHForest, MinHash
from lsh_benchmark import get_newsgroup_data 


def benchmark_lshforest(l, k, index_data, query_data):
    print("Building LSH Forest index")
    num_perm = len(index_data.minhashes[0].hashvalues)
    forest = MinHashLSHForest(num_perm=num_perm, l=l)
    for key, minhash in zip(index_data.filenames, index_data.minhashes):
        forest.add(key, minhash)
    forest.index()
    print("Querying")
    times = []
    results = []
    for minhash in query_data.minhashes:
        start = time.clock()
        result = forest.query(minhash, k)
        duration = time.clock() - start
        times.append(duration)
        results.append(result)
    return times, results


def benchmark_linearscan(k, index_data, query_data):
    times = []
    results = []
    for q in query_data.minhashes:
        start = time.clock()
        result = []
        result = [(key, q.jaccard(m))
                  for key, m in zip(index_data.filenames, index_data.minhashes)]
        result.sort(key=lambda x : x[1], reverse=True)
        result = [x[0] for x in result[:k]]
        duration = time.clock() - start
        times.append(duration)
        results.append(result)
    return times, results


def benchmark_ground_truth(k, index_data, query_data):
    times = []
    results = []
    def _compute_jaccard(x, y):
        intersection = 0
        for w in x:
            if w in y:
                intersection += 1
        return float(intersection) / float(len(x) + len(y) - intersection)
    for q in query_data.shingles:
        start = time.clock()
        result = [(key, _compute_jaccard(q, a)) 
                  for key, a in zip(index_data.filenames, index_data.shingles)]
        result.sort(key=lambda x : x[1], reverse=True)
        result = [x[0] for x in result[:k]]
        duration = time.clock() - start
        results.append(result)
        times.append(duration)
    return times, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="lshforest_benchmark.json")
    args = parser.parse_args(sys.argv[1:])

    num_perms = [32, 64, 96, 128, 160, 192, 224, 256]
    l = 8
    k = 5
    output = {"num_perms" : num_perms,
              "l" : l,
              "k" : k,
              "lsh_times" : [], "lsh_results" : [],
              "linearscan_times" : [], "linearscan_results" : [],
              "ground_truth_times" : None, "ground_truth_results" : None}

    for num_perm in num_perms:
        print("Use num_perm = %d" % num_perm)
        index_data = get_newsgroup_data(num_perm, "train")
        query_data = get_newsgroup_data(num_perm, "test")
        result = {}
        print("Running LSH benchmark l = %d" % l)
        lsh_times, lsh_results = benchmark_lshforest(l, k, index_data, query_data)
        print("Running linear scan benchmark")
        linearscan_times, linearscan_results = benchmark_linearscan(k, index_data, query_data)
        output["lsh_times"].append(lsh_times)
        output["lsh_results"].append(lsh_results)
        output["linearscan_times"].append(linearscan_times)
        output["linearscan_results"].append(linearscan_results)
    
    print("Running ground truth benchmark")
    output["ground_truth_times"], output["ground_truth_results"] =\
            benchmark_ground_truth(k, index_data, query_data)

    average_cardinality = np.mean([len(s) for s in
        index_data.shingles + query_data.shingles])
    print("Average cardinality is", average_cardinality)

    with open(args.output, 'w') as f:
        json.dump(output, f)
