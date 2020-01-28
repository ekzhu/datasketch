import sys
import argparse

import farmhash
import numpy as np

from exact_set_similarity_search import search_jaccard_topk
from hnsw import search_hnsw_jaccard_topk
from lsh import search_lsh_jaccard_topk
from lshforest import search_lshforest_jaccard_topk
from utils import read_sets_from_file, save_results, create_minhashes_from_sets, \
        is_run_exist


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
    rs = [1, 2, 4, 8]

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

    # Run ground truth.
    print("Running Ground Truth.")
    ground_truth_results, ground_truth_times = search_jaccard_topk(
            (index_sets, index_keys), (query_sets, query_keys), k)
    save_results("ground_truth", k, None, {"benchmark": benchmark_settings},
            ground_truth_results, ground_truth_times, args.output)

    # Run HNSW
    print("Running HNSW.")
    for M in Ms:
        for efC in efCs:
            print(f"Using M = {M}, efC = {efC}")
            index_params = {
                'M': M, 
                'indexThreadQty': num_threads, 
                'efConstruction': efC, 
                'post' : 0,
            }
            hnsw_results, hnsw_times = search_hnsw_jaccard_topk(
                    (index_sets, index_keys), 
                    (query_sets, query_keys), index_params, k)
            # Save results
            params = {"index": index_params, "benchmark": benchmark_settings}
            save_results("hnsw", k, None, params, 
                    hnsw_results, hnsw_times, args.output)

    # Create Minhashes.
    print("Creating minhashes.")
    num_perms = list(np.unique([b*r for r in rs for b in bs]))
    index_minhashes = create_minhashes_from_sets(index_sets, num_perms, 
            hashfunc=farmhash.hash32)
    query_minhashes = create_minhashes_from_sets(query_sets, num_perms, 
            hashfunc=farmhash.hash32)

    # Run LSH.
    for b in bs:
        for r in rs:
            params = {
                "index": {"b": b, "r": r}, 
                "benchmark": benchmark_settings,
            }
            print(f"Running LSH using b = {b}, r = {r}.")
            results, times = search_lsh_jaccard_topk(
                    (index_sets, index_keys, index_minhashes),
                    (query_sets, query_keys, query_minhashes),
                    b, r, k)
            save_results("lsh", k, None, params, results, times, args.output)
            print(f"Running LSH Forest using b = {b}, r = {r}.")
            results, times = search_lshforest_jaccard_topk(
                    (index_sets, index_keys, index_minhashes),
                    (query_sets, query_keys, query_minhashes),
                    b, r, k)
            save_results("lshforest", k, None, params, results, times, 
                    args.output)