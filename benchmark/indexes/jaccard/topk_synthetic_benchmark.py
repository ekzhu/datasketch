import sys
import argparse

import farmhash

from exact_set_similarity_search import search_jaccard_topk
from hnsw import search_hnsw_jaccard_topk
from lshforest import search_lshforest_jaccard_topk
from utils import read_sets_from_file, save_results, create_minhashes_from_sets
from synthetic_data import generate_sets, sample_sets


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", 
            default="jaccard_topk_synthetic_benchmark.sqlite",
            help="The output SQLite3 database to write results.")
    args = parser.parse_args(sys.argv[1:])

    # Global parameters.
    k = 10
    num_sets = 1000
    max_set_size = 100
    query_sample_ratio = 0.1
    vocabulary_sizes = [200, 400, 600, 800, 1000]

    # LSH Forest parameters (only one each allowed for each).
    num_perms = [128,]
    ls = [32,]

    # HNSW Index parameters (only one value allowed for each).
    Ms = [30,]
    efCs = [100,]
    num_threads = 1

    for vocabulary_size in vocabulary_sizes:
        print(f"Token Dictionary(size={vocabulary_size})")
        benchmark_settings = {
            "num_sets": num_sets,
            "max_set_size": max_set_size,
            "query_sample_ratio": query_sample_ratio,
            "vocabulary_size": vocabulary_size,
        }

        # Create synthetic benchmark dataset.
        print("Creating synthetic benchmark dataset.")
        (index_sets, index_keys) = generate_sets(num_sets, 
                vocabulary_size, max_set_size)
        (query_sets, query_keys) = sample_sets(index_sets, index_keys, 
                query_sample_ratio)

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
