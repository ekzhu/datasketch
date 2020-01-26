import sys
import argparse
import time

import farmhash
from datasketch import MinHashLSH, MinHash

from exact_set_similarity_search import search_jaccard_threshold
from utils import read_sets_from_file, compute_jaccard, save_results, \
        create_minhashes_from_sets


def search_lsh_jaccard_threshold(index_data, query_data, num_perm, threshold):
    (index_sets, index_keys, index_minhashes) = index_data
    (query_sets, query_keys, query_minhashes) = query_data
    print("Building LSH Index.")
    start = time.perf_counter()
    index = MinHashLSH(threshold, num_perm)
    # Use the indices of the indexed sets as keys in LSH.
    for i in range(len(index_keys)):
        index.insert(i, index_minhashes[num_perm][i])
    end = time.perf_counter()
    print("Indexing time: {:.3f}.".format(end-start))
    print("Querying.")
    times = []
    results = []
    for query_minhash, query_key, query_set in \
            zip(query_minhashes[num_perm], query_keys, query_sets):
        start = time.perf_counter()
        result = index.query(query_minhash)
        # Recover the retrieved indexed sets and 
        # compute the exact Jaccard similarities.
        result = [[index_keys[i], compute_jaccard(query_set, index_sets[i])]
                               for i in result]
        # Filter out incorrect candidates.
        result = [[key, similarity] for key, similarity in result
                if similarity >= threshold]
        # Sort by similarity.
        result.sort(key=lambda x : x[1], reverse=True)
        duration = time.perf_counter() - start
        times.append(duration)
        results.append((query_key, result))
    return (results, times)


def search_minhash_jaccard_threshold(index_data, query_data, num_perm, 
        threshold):
    """Run the linear scan algorithm using MinHash sketches for Jaccard
    threshold-based search."""
    (index_sets, index_keys, index_minhashes) = index_data
    (query_sets, query_keys, query_minhashes) = query_data
    times = []
    results = []
    for query_key, query_set, query_minhash in \
            zip(query_keys, query_sets, query_minhashes[num_perm]):
        start = time.perf_counter()
        result = []
        for index_key, index_set, index_minhash in \
                zip(index_keys, index_sets, index_minhashes[num_perm]):
            similarity = query_minhash.jaccard(index_minhash)
            if similarity < threshold:
                continue
            similarity = compute_jaccard(query_set, index_set)
            if similarity < threshold:
                continue
            result.append([index_key, similarity])
        # Sort by similarity.
        result.sort(key=lambda x : x[1], reverse=True)
        duration = time.perf_counter() - start
        times.append(duration)
        results.append((query_key, result))
    return (results, times)


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
    num_perms = [32, 64, 96, 128, 160, 192, 224, 256]
    #thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    thresholds = [0.5,]
    
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
        save_results("ground_truth", args.index_set_file, args.query_set_file,
                args.index_sample_ratio, args.query_sample_ratio, 
                None, threshold,
                {}, ground_truth_results, ground_truth_times, args.output)
        # Run LSH.
        print("Running LSH.")
        for num_perm in num_perms:
            print(f"Using num_perm = {num_perm}.")
            results, times = search_lsh_jaccard_threshold(
                    (index_sets, index_keys, index_minhashes),
                    (query_sets, query_keys, query_minhashes),
                    num_perm, threshold)
            save_results("lsh", args.index_set_file, args.query_set_file,
                    args.index_sample_ratio, args.query_sample_ratio,
                    None, threshold,
                    {"num_perm": num_perm}, results, times, args.output)
