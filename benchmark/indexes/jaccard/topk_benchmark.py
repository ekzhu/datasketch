import sys
import argparse

import farmhash
import numpy as np

from exact import search_jaccard_topk
from hnsw import search_hnsw_jaccard_topk
from lsh import search_lsh_jaccard_topk
from lshforest import search_lshforest_jaccard_topk
from utils import read_sets_from_file, save_results, create_minhashes_from_sets, \
        get_run, init_results_db


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-set-file", required=True, 
            help="The set file from set-similarity-search-benchmark to use "
            "as the source of index sets.")
    parser.add_argument("--query-set-file", required=True, 
            help="The set file from set-similarity-search-benchmark to use "
            "as the source of query sets.")
    parser.add_argument("--index-sample-ratio", default=1.0, type=float,
            help="The fraction of sets from the index-set-file to be used "
            "as query sets.")
    parser.add_argument("--query-sample-ratio", default=0.1, type=float,
            help="The fraction of sets from the query-set-file to be used "
            "as query sets.")
    parser.add_argument("--output", default="jaccard_topk_benchmark.sqlite",
            help="The output SQLite3 database to write results.")
    args = parser.parse_args(sys.argv[1:])

    # Global parameters.
    k = 100

    # LSH parameters, num_perm = b * r
    bs = [4, 8, 16, 32, 64, 128, 256]
    rs = [2, 4, 8]

    # HNSW Index parameters.
    Ms = [4, 8, 12, 16, 24, 36, 48, 64, 96]
    efCs = [100, 500, 1000, 1500, 2000]
    num_threads = 1

    # Benchmark settings.
    benchmark_settings = {
        "k": k,
        "index_set_file": args.index_set_file,
        "query_set_file": args.query_set_file,
        "index_sample_ratio": args.index_sample_ratio,
        "query_sample_ratio": args.query_sample_ratio,
    } 

    # Read benchmark dataset.
    print("Reading benchmark dataset.")
    (index_sets, index_keys) = read_sets_from_file(args.index_set_file, 
            sample_ratio=args.index_sample_ratio, 
            skip=1)
    (query_sets, query_keys) = read_sets_from_file(args.index_set_file, 
            sample_ratio=args.query_sample_ratio, 
            skip=1)
    
    # Initialize output SQLite database.
    init_results_db(args.output)

    # Run ground truth.
    params = {"benchmark": benchmark_settings}
    if get_run("ground_truth", k, None, params, args.output) is None:
        print("Running Ground Truth.")
        ground_truth_results, ground_truth_times = search_jaccard_topk(
                (index_sets, index_keys), (query_sets, query_keys), k)
        save_results("ground_truth", k, None, params,
                ground_truth_results, ground_truth_times, args.output)

    # Run HNSW
    for M in Ms:
        for efC in efCs:
            index_params = {
                'M': M, 
                'indexThreadQty': num_threads, 
                'efConstruction': efC, 
                'post' : 0,
            }
            params = {"index": index_params, "benchmark": benchmark_settings}
            if get_run("hnsw", k, None, params, args.output) is not None:
                continue
            print(f"Running HNSW using M = {M}, efC = {efC}")
            hnsw_results, hnsw_times = search_hnsw_jaccard_topk(
                    (index_sets, index_keys), 
                    (query_sets, query_keys), index_params, k)
            # Save results
            save_results("hnsw", k, None, params, 
                    hnsw_results, hnsw_times, args.output)

    # Initialize storage for Minhashes.
    index_minhashes = {}
    query_minhashes = {}

    # Run LSH.
    for b in bs:
        for r in rs:
            params = {
                "index": {"b": b, "r": r}, 
                "benchmark": benchmark_settings,
            }
            if get_run("lsh", k, None, params, args.output) is not None:
                continue
            print(f"Running LSH using b = {b}, r = {r}.")
            # Lazily create MinHashes.
            if b*r not in index_minhashes:
                index_minhashes.update(create_minhashes_from_sets(
                        index_sets, [b*r,], hashfunc=farmhash.hash32))
            if b*r not in query_minhashes:
                query_minhashes.update(create_minhashes_from_sets(
                        query_sets, [b*r,], hashfunc=farmhash.hash32))
            # Run benchmark.
            results, times = search_lsh_jaccard_topk(
                    (index_sets, index_keys, index_minhashes),
                    (query_sets, query_keys, query_minhashes),
                    b, r, k)
            # Save result to SQLite database.
            save_results("lsh", k, None, params, results, times, args.output)
    
    # Run LSH Forest.
    for b in bs:
        for r in rs:
            params = {
                "index": {"b": b, "r": r}, 
                "benchmark": benchmark_settings,
            }
            if get_run("lshforest", k, None, params, args.output) is not None:
                continue
            print(f"Running LSH Forest using b = {b}, r = {r}.")
            # Lazily create MinHashes.
            if b*r not in index_minhashes:
                index_minhashes.update(create_minhashes_from_sets(
                        index_sets, [b*r,], hashfunc=farmhash.hash32))
            if b*r not in query_minhashes:
                query_minhashes.update(create_minhashes_from_sets(
                        query_sets, [b*r,], hashfunc=farmhash.hash32))
            # Run benchmark.
            results, times = search_lshforest_jaccard_topk(
                    (index_sets, index_keys, index_minhashes),
                    (query_sets, query_keys, query_minhashes),
                    b, r, k)
            # Save result to SQLite database.
            save_results("lshforest", k, None, params, results, times, 
                    args.output)
