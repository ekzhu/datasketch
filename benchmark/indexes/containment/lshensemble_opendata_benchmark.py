"""
Benchmark dataset from:
https://github.com/ekzhu/set-similarity-search-benchmark.
Use "Canada US and UK Open Data":

    Indexed sets: canada_us_uk_opendata.inp.gz
    Query sets (10 stratified samples from 10 percentile intervals):
        Size from 10 - 1k: canada_us_uk_opendata_queries_1k.inp.gz
        Size from 10 - 10k: canada_us_uk_opendata_queries_10k.inp.gz
        Size from 10 - 100k: canada_us_uk_opendata_queries_100k.inp.gz
"""
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
from SetSimilaritySearch import SearchIndex
import farmhash

from datasketch import MinHashLSHEnsemble, MinHash


def _hash_32(d):
    return farmhash.hash32(d)


def bootstrap_sets(sets_file, sample_ratio, num_perms, skip=1,
        pad_for_asym=False):
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
            s = np.array([int(d) for d in \
                    line.strip().split("\t")[1].split(",")])
            sets.append(s)
            sys.stdout.write("\rRead {} sets".format(len(sets)))
        sys.stdout.write("\n")
    sets = list(sets)
    keys = list(range(len(sets)))
    # Generate paddings for asym.
    max_size = max(len(s) for s in sets)
    paddings = dict()
    if pad_for_asym:
        padding_sizes = sorted(list(set([max_size-len(s) for s in sets])))
        for num_perm in num_perms:
            paddings[num_perm] = dict()
            for i, padding_size in enumerate(padding_sizes):
                if i == 0:
                    prev_size = 0
                    pad = MinHash(num_perm, hashfunc=_hash_32)
                else:
                    prev_size = padding_sizes[i-1]
                    pad = paddings[num_perm][prev_size].copy()
                for w in range(prev_size, padding_size):
                    pad.update(str(w)+"_tmZZRe8DE23s")
                paddings[num_perm][padding_size] = pad
    # Generate minhash
    print("Creating MinHash...")
    minhashes = dict()
    for num_perm in num_perms:
        print("Using num_parm = {}".format(num_perm))
        ms = []
        for s in sets:
            m = MinHash(num_perm, hashfunc=_hash_32)
            for word in s:
                m.update(str(word))
            if pad_for_asym:
                # Add padding to the minhash
                m.merge(paddings[num_perm][max_size-len(s)])
            ms.append(m)
            sys.stdout.write("\rMinhashed {} sets".format(len(ms)))
        sys.stdout.write("\n")
        minhashes[num_perm] = ms

    return (minhashes, sets, keys)


def benchmark_lshensemble(threshold, num_perm, num_part, m, storage_config,
        index_data, query_data):
    print("Building LSH Ensemble index")
    (minhashes, indexed_sets, keys) = index_data
    lsh = MinHashLSHEnsemble(threshold=threshold, num_perm=num_perm,
            num_part=num_part, m=m, storage_config=storage_config)
    lsh.index((key, minhash, len(s))
            for key, minhash, s in \
                    zip(keys, minhashes[num_perm], indexed_sets))
    print("Querying")
    (minhashes, sets, keys) = query_data
    probe_times = []
    process_times = []
    results = []
    for qs, minhash in zip(sets, minhashes[num_perm]):
        # Record probing time
        start = time.perf_counter()
        result = list(lsh.query(minhash, len(qs)))
        probe_times.append(time.perf_counter() - start)
        # Record post processing time.
        start = time.perf_counter()
        [_compute_containment(qs, indexed_sets[key]) for key in result]
        process_times.append(time.perf_counter() - start)
        results.append(result)
        sys.stdout.write("\rQueried {} sets".format(len(results)))
    sys.stdout.write("\n")
    return results, probe_times, process_times


def benchmark_ground_truth(threshold, index, query_data):
    (_, query_sets, _) = query_data
    times = []
    results = []
    for q in query_sets:
        start = time.perf_counter()
        result = [key for key, _ in index.query(q)]
        duration = time.perf_counter() - start
        times.append(duration)
        results.append(result)
        sys.stdout.write("\rQueried {} sets".format(len(results)))
    sys.stdout.write("\n")
    return results, times


def _compute_containment(x, y):
    if len(x) == 0 or len(y) == 0:
        return 0.0
    intersection = len(np.intersect1d(x, y, assume_unique=True))
    return float(intersection) / float(len(x))


levels = {
    "test": {
        "thresholds": [1.0,],
        "num_parts": [4,],
        "num_perms": [32,],
        "m": 2,
    },
    "lite": {
        "thresholds": [0.5, 0.75, 1.0],
        "num_parts": [8, 16],
        "num_perms": [32, 64],
        "m": 8,
    },
    "medium": {
        "thresholds": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "num_parts": [8, 16, 32],
        "num_perms": [32, 128, 224],
        "m": 8,
    },
    "complete": {
        "thresholds": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "num_parts": [8, 16, 32],
        "num_perms": [32, 64, 96, 128, 160, 192, 224, 256],
        "m": 8,
    },
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Run LSH Ensemble benchmark using data sets obtained "
            "from https://github.com/ekzhu/set-similarity-search-benchmarks.")
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
    parser.add_argument("--indexed-sets-sample-ratio", type=float, default=0.1)
    parser.add_argument("--level", type=str, choices=levels.keys(), 
            default="complete")
    parser.add_argument("--skip-ground-truth", action="store_true")
    parser.add_argument("--use-asym-minhash", action="store_true")
    parser.add_argument("--use-redis", action="store_true")
    parser.add_argument("--redis-host", type=str, default="localhost")
    parser.add_argument("--redis-port", type=int, default=6379)
    args = parser.parse_args(sys.argv[1:])

    level = levels[args.level]
    
    index_data, query_data = None, None
    index_data_cache = "{}.pickle".format(args.indexed_sets)
    query_data_cache = "{}.pickle".format(args.query_sets)
    if os.path.exists(index_data_cache):
        print("Using cached indexed sets {}".format(index_data_cache))
        with open(index_data_cache, "rb") as d:
            index_data = pickle.load(d)
    else:
        print("Using indexed sets {}".format(args.indexed_sets))
        index_data = bootstrap_sets(args.indexed_sets, 
                args.indexed_sets_sample_ratio, num_perms=level["num_perms"],
                pad_for_asym=args.use_asym_minhash)
        with open(index_data_cache, "wb") as d:
            pickle.dump(index_data, d)
    if os.path.exists(query_data_cache):
        print("Using cached query sets {}".format(query_data_cache))
        with open(query_data_cache, "rb") as d:
            query_data = pickle.load(d)
    else:
        print("Using query sets {}".format(args.query_sets))
        query_data = bootstrap_sets(args.query_sets, 1.0, 
                num_perms=level["num_perms"], skip=0)
        with open(query_data_cache, "wb") as d:
            pickle.dump(query_data, d)

    if not args.skip_ground_truth:
        rows = []
        # Build search index separately, only works for containment.
        print("Building search index...")
        index = SearchIndex(index_data[1], similarity_func_name="containment",
                similarity_threshold=0.1)
        for threshold in level["thresholds"]:
            index.similarity_threshold = threshold
            print("Running ground truth benchmark threshold = {}".format(threshold))
            ground_truth_results, ground_truth_times = \
                    benchmark_ground_truth(threshold, index, query_data)
            for t, r, query_set, query_key in zip(ground_truth_times,
                    ground_truth_results, query_data[1], query_data[2]):
                rows.append((query_key, len(query_set), threshold, t,
                    ",".join(str(k) for k in r)))
        df_groundtruth = pd.DataFrame.from_records(rows,
            columns=["query_key", "query_size", "threshold",
                "query_time", "results"])
        df_groundtruth.to_csv(args.ground_truth_results)
    
    storage_config = {"type": "dict"}
    if args.use_redis:
        storage_config = {
            "type": "redis",
            "redis": {
                "host": args.redis_host,
                "port": args.redis_port,
            },
        }

    rows = []
    for threshold in level["thresholds"]:
        for num_part in level["num_parts"]:
            for num_perm in level["num_perms"]:
                print("Running LSH Ensemble benchmark "
                        "threshold = {}, num_part = {}, num_perm = {}".format(
                            threshold, num_part, num_perm))
                results, probe_times, process_times = benchmark_lshensemble(
                        threshold, num_perm, num_part, level["m"], storage_config, 
                        index_data, query_data)
                for probe_time, process_time, result, query_set, query_key in zip(\
                        probe_times, process_times, results, \
                        query_data[1], query_data[2]):
                    rows.append((query_key, len(query_set), threshold,
                        num_part, num_perm, probe_time, process_time,
                        ",".join(str(k) for k in result)))
    df = pd.DataFrame.from_records(rows,
        columns=["query_key", "query_size", "threshold", "num_part",
            "num_perm", "probe_time", "process_time", "results"])
    df.to_csv(args.query_results)

