import sys
import argparse
import time

import farmhash
import nmslib
from datasketch import MinHashLSHForest

from exact_set_similarity_search import search_jaccard_topk
from utils import read_sets_from_file, compute_jaccard, save_results,\
        create_minhashes_from_sets


def search_hnsw_jaccard_topk(index_data, query_data, index_params, k):
    (index_sets, index_keys) = index_data
    (query_sets, query_keys) = query_data
    print("Building HNSW Index.")
    start = time.perf_counter()
    index = nmslib.init(method="hnsw", space="jaccard_sparse", 
            data_type=nmslib.DataType.OBJECT_AS_STRING)
    index.addDataPointBatch(
            [" ".join(str(v) for v in s) for s in index_sets],
            index_keys)
    index.createIndex(index_params)
    end = time.perf_counter()
    print("Indexing time: {:.3f}.".format(end-start))
    print("Querying.")
    times = []
    results = []
    for query_set, query_key in zip(query_sets, query_keys):
        start = time.perf_counter()
        keys, distances = index.knnQuery(" ".join(str(v) for v in query_sets), k)
        result = [[int(key), float(1.0-distance)] 
                for key, distance in zip(keys, distances)]
        result.sort(key=lambda x : x[1], reverse=True)
        duration = time.perf_counter() - start
        times.append(duration)
        results.append((query_key, result))
        sys.stdout.write("\rQueried {} sets".format(len(results)))
    sys.stdout.write("\n")
    return (results, times)


def search_lshforest_jaccard_topk(index_data, query_data, num_perm, l, k):
    (index_sets, index_keys, index_minhashes) = index_data
    (query_sets, query_keys, query_minhashes) = query_data
    print("Building LSH Forest Index.")
    start = time.perf_counter()
    index = MinHashLSHForest(num_perm=num_perm, l=l)
    # Use the indices of the indexed sets as keys in LSH.
    for i in range(len(index_keys)):
        index.add(i, index_minhashes[num_perm][i])
    index.index()
    end = time.perf_counter()
    print("Indexing time: {:.3f}.".format(end-start))
    print("Querying.")
    times = []
    results = []
    for query_minhash, query_key, query_set in \
            zip(query_minhashes[num_perm], query_keys, query_sets):
        start = time.perf_counter()
        result = index.query(query_minhash, k)
        # Recover the retrieved indexed sets and 
        # compute the exact Jaccard similarities.
        result = [[index_keys[i], compute_jaccard(query_set, index_sets[i])]
                               for i in result]
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
    parser.add_argument("--output", default="jaccard_topk_benchmark.sqlite",
            help="The output SQLite3 database to write results.")
    args = parser.parse_args(sys.argv[1:])

    # Global parameters.
    k = 100

    # LSH Forest parameters.
    num_perms = [32, 64, 96, 128, 160, 192, 224, 256]
    ls = [4, 8, 16, 32]

    # HNSW Index parameters.
    Ms = [20, 40, 60, 80, 100]
    efCs = [100, 500, 100, 1500, 2000]
    num_threads = 1

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
    save_results("ground_truth", args.index_set_file, args.query_set_file,
            args.index_sample_ratio, args.query_sample_ratio, k, None,
            {}, ground_truth_results, ground_truth_times, args.output)

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
            save_results("hnsw", args.index_set_file, args.query_set_file,
                    args.index_sample_ratio, args.query_sample_ratio, 
                    k, None, index_params, 
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
            print(f"Using num_perm = {num_perm}, l = {l}.")
            results, times = search_lshforest_jaccard_topk(
                    (index_sets, index_keys, index_minhashes),
                    (query_sets, query_keys, query_minhashes),
                    num_perm, l, k)
            save_results("lshforest", args.index_set_file, args.query_set_file,
                    args.index_sample_ratio, args.query_sample_ratio,
                    k, None,
                    {"num_perm": num_perm, "l": l}, 
                    results, times, args.output)
