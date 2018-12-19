import time, argparse, sys, json
import numpy as np
import scipy.stats
import random
import collections
import gzip
import random
import os
import pickle
import pandas as pd

from datasketch import MinHashLSHEnsemble, MinHash


def bootstrap_sets(sets_file, sample_ratio, num_perms, skip=1):
    print("Creating sets...")
    sets = collections.deque([])
    random.seed(41)
    with gzip.open(sets_file, "rt") as f:
        for i, line in enumerate(f):
            if i < skip:
                # Skip lines
                continue
            if random.random() > sample_ratio:
                continue
            s = np.array(line.strip().split("\t")[1].split(","))
            sets.append(s)
            sys.stdout.write("\rRead {} sets".format(len(sets)))
        sys.stdout.write("\n")
    sets = list(sets)
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
    return (minhashes, sets, keys)


def benchmark_lshensemble(threshold, num_perm, num_part, m, index_data,
        query_data):
    print("Building LSH Ensemble index")
    (minhashes, indexed_sets, keys) = index_data
    lsh = MinHashLSHEnsemble(threshold=threshold, num_perm=num_perm,
            num_part=num_part, m=m)
    lsh.index((key, minhash, len(set))
            for key, minhash, set in \
                    zip(keys, minhashes[num_perm], indexed_sets))
    print("Querying")
    (minhashes, sets, keys) = query_data
    times = []
    results = []
    for qs, minhash in zip(sets, minhashes[num_perm]):
        start = time.perf_counter()
        result = list(lsh.query(minhash, len(qs)))
        duration = time.perf_counter() - start
        times.append(duration)
        results.append(result)
        # results.append(sorted([[key, _compute_containment(qs, indexed_sets[key])]
        #                        for key in result],
        #                       key=lambda x : x[1], reverse=True))
        sys.stdout.write("\rQueried {} sets".format(len(results)))
    sys.stdout.write("\n")
    return times, results


def benchmark_ground_truth(threshold, index_data, query_data):
    (minhashes, indexed_sets, keys) = index_data
    (_, query_sets, _) = query_data
    times = []
    results = []
    for q in query_sets:
        start = time.perf_counter()
        result = [key for key, a in zip(keys, indexed_sets)
                  if _compute_containment(q, a) >= threshold]
        duration = time.perf_counter() - start
        times.append(duration)
        results.append(result)
        # results.append(sorted([[key, _compute_containment(q, indexed_sets[key])]
        #                        for key in result],
        #                       key=lambda x : x[1], reverse=True))
        sys.stdout.write("\rQueried {} sets".format(len(results)))
    sys.stdout.write("\n")
    return times, results


def _compute_containment(x, y):
    if len(x) == 0 or len(y) == 0:
        return 0.0
    intersection = len(np.intersect1d(x, y, assume_unique=True))
    return float(intersection) / float(len(x))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--indexed-sets", type=str, required=True,
            help="Input indexed set file (gzipped), each line is a set: "
            "<set_size> <1>,<2>,<3>..., where each <?> is an element.")
    parser.add_argument("--query-sets", type=str, required=True,
            help="Input query set file (gzipped), each line is a set: "
            "<set_size> <1>,<2>,<3>..., where each <?> is an element.")
    parser.add_argument("--query-results", type=str,
            default="lshensemble_benchmark_query_results.csv")
    parser.add_argument("--ground-truth-results", type=str,
            default="lshensemble_benchmark_ground_truth_results.csv")
    args = parser.parse_args(sys.argv[1:])

    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    num_parts = [8, 16, 32]
    #num_perms = [32, 64, 96, 128, 160, 192, 224, 256]
    num_perms = [256,]
    m = 8
    output = {"thresholds" : thresholds,
              "num_parts" : num_parts,
              "num_perms" : num_perms,
              "m" : m,
              "lsh_times" : [], "lsh_results" : [],
              "ground_truth_times" : None, "ground_truth_results" : None}

    index_data, query_data = None, None
    index_data_cache = "{}.pickle".format(args.indexed_sets)
    query_data_cache = "{}.pickle".format(args.query_sets)
    if os.path.exists(index_data_cache):
        print("Using cached indexed sets {}".format(index_data_cache))
        with open(index_data_cache, "rb") as d:
            index_data = pickle.load(d)
    else:
        print("Using indexed sets {}".format(args.indexed_sets))
        index_data = bootstrap_sets(args.indexed_sets, 0.001, num_perms)
        with open(index_data_cache, "wb") as d:
            pickle.dump(index_data, d)
    if os.path.exists(query_data_cache):
        print("Using cached query sets {}".format(query_data_cache))
        with open(query_data_cache, "rb") as d:
            query_data = pickle.load(d)
    else:
        print("Using query sets {}".format(args.query_sets))
        query_data = bootstrap_sets(args.query_sets, 1.0, num_perms, skip=0)
        with open(query_data_cache, "wb") as d:
            pickle.dump(query_data, d)

    rows = []
    for threshold in thresholds:
        for num_part in num_parts:
            for num_perm in num_perms:
                print("Running LSH Ensemble benchmark "
                        "threshold = {}, num_part = {}, num_perm = {}".format(
                            threshold, num_part, num_perm))
                lsh_times, lsh_results = benchmark_lshensemble(
                        threshold, num_perm, num_part, m, index_data, query_data)
                for t, r, query_set, query_key in zip(lsh_times, lsh_results,
                        query_data[1], query_data[2]):
                    rows.append((query_key, len(query_set), threshold,
                        num_part, num_perm, t, ",".join(str(k) for k in r)))
    df = pd.DataFrame.from_records(rows,
        columns=["query_key", "query_size", "threshold", "num_part",
            "num_perm", "query_time", "results"])
    df.to_csv(args.query_results)

    rows = []
    for threshold in thresholds:
        print("Running ground truth benchmark threshold = {}".format(threshold))
        ground_truth_times, ground_truth_results = \
                benchmark_ground_truth(threshold, index_data, query_data)
        for t, r, query_set, query_key in zip(ground_truth_times,
                ground_truth_results, query_data[1], query_data[2]):
            rows.append((query_key, len(query_set), threshold, t,
                ",".join(str(k) for k in r)))
    df_groundtruth = pd.DataFrame.from_records(rows,
        columns=["query_key", "query_size", "threshold",
            "query_time", "results"])
    df_groundtruth.to_csv(args.ground_truth_results)

