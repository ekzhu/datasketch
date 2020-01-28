import sys
import argparse

import farmhash

from exact_set_similarity_search import search_jaccard_threshold
from lsh import search_lsh_jaccard_threshold
from utils import read_sets_from_file, save_results, create_minhashes_from_sets


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
    parser.add_argument("--output", default="jaccard_threshold_benchmark.sqlite",
            help="The output SQLite3 database to write results.")
    args = parser.parse_args(sys.argv[1:])

    # MinHash LSH parameters.
    num_perms = [8, 16, 32, 64, 128, 256, 512]
    thresholds = [0.5,]
    
    # Benchmark settings.
    benchmark_settings = {
        "index_set_file": args.index_set_file,
        "query_set_file": args.query_set_file,
        "index_sample_ratio": args.index_sample_ratio,
        "query_sample_ratio": args.query_sample_ratio,
    } 
    
    # Read benchmark dataset.
    print("Reading benchmark dataset.")
    index_sets, index_keys = read_sets_from_file(args.index_set_file, 
            sample_ratio=args.index_sample_ratio, 
            skip=1)
    query_sets, query_keys = read_sets_from_file(args.index_set_file, 
            sample_ratio=args.query_sample_ratio, 
            skip=1)
    
    # Create minhashes.
    print("Creating minhashes.")
    index_minhashes = create_minhashes_from_sets(index_sets, num_perms, 
            hashfunc=farmhash.hash32)
    query_minhashes = create_minhashes_from_sets(query_sets, num_perms, 
            hashfunc=farmhash.hash32)

    for threshold in thresholds:
        # Run ground truth.
        print("Running Ground Truth.")
        ground_truth_results, ground_truth_times = search_jaccard_threshold(
                (index_sets, index_keys), (query_sets, query_keys), threshold)
        params = {"benchmark": {"threshold": threshold, **benchmark_settings}}
        save_results("ground_truth", None, threshold, params, 
                ground_truth_results, ground_truth_times, args.output)
        # Run LSH.
        print("Running LSH.")
        for num_perm in num_perms:
            print(f"Using num_perm = {num_perm}.")
            results, times = search_lsh_jaccard_threshold(
                    (index_sets, index_keys, index_minhashes),
                    (query_sets, query_keys, query_minhashes),
                    num_perm, threshold)
            params = {
                "index": {"num_perm": num_perm},
                "benchmark": {"threshold": threshold, **benchmark_settings},
            }
            save_results("lsh", None, threshold, params,
                    results, times, args.output)
