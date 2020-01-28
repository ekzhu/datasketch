import sys
import argparse

import farmhash

from exact_set_similarity_search import search_jaccard_topk
from hnsw import search_hnsw_jaccard_topk
from lshforest import search_lshforest_jaccard_topk
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
    parser.add_argument("--output", default="jaccard_topk_benchmark.sqlite",
            help="The output SQLite3 database to write results.")
    args = parser.parse_args(sys.argv[1:])

    # Global parameters.
    k = 100

    # LSH Forest parameters.
    num_perms = [8, 16, 32, 64, 128, 256, 512]
    ls = [4, 8, 16, 32, 64]

    # HNSW Index parameters.
    Ms = [4, 8, 12, 16, 24, 36, 48, 64, 96]
    efCs = [100, 500, 1000, 1500, 2000]
    num_threads = 1

    # Benchmark settings.
    benchmark_settings = {
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
    save_results("ground_truth", k, None, benchmark_settings,
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
            index_params.update(benchmark_settings)
            save_results("hnsw", k, None, index_params, 
                    hnsw_results, hnsw_times, args.output)

    # Create Minhashes.
    print("Creating minhashes.")
    index_minhashes = create_minhashes_from_sets(index_sets, num_perms, 
            hashfunc=farmhash.hash32)
    query_minhashes = create_minhashes_from_sets(query_sets, num_perms, 
            hashfunc=farmhash.hash32)

    # Run LSH Forest.
    print("Running LSH Forest.")
    for num_perm in num_perms:
        for l in ls:
            if l > num_perm:
                continue
            print(f"Using num_perm = {num_perm}, l = {l}.")
            params = {"num_perm": num_perm, "l": l}
            params.update(benchmark_settings)
            results, times = search_lshforest_jaccard_topk(
                    (index_sets, index_keys, index_minhashes),
                    (query_sets, query_keys, query_minhashes),
                    num_perm, l, k)
            save_results("lshforest", k, None, params,
                    results, times, args.output)
