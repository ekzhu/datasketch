import sys
import argparse
import time

import farmhash

from exact import search_jaccard_topk
from hnsw import search_hnsw_jaccard_topk
from lsh import search_lsh_jaccard_topk
from lshforest import search_lshforest_jaccard_topk
from utils import (
    read_sets_from_file,
    save_results,
    create_minhashes_from_sets,
    get_run,
    init_results_db,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--index-set-file",
        required=True,
        help="The set file from set-similarity-search-benchmark to use "
        "as the source of index sets.",
    )
    parser.add_argument(
        "--query-set-file",
        required=True,
        help="The set file from set-similarity-search-benchmark to use "
        "as the source of query sets.",
    )
    parser.add_argument(
        "--index-sample-ratio",
        default=1.0,
        type=float,
        help="The fraction of sets from the index-set-file to be used "
        "as query sets.",
    )
    parser.add_argument(
        "--query-sample-ratio",
        default=0.1,
        type=float,
        help="The fraction of sets from the query-set-file to be used "
        "as query sets.",
    )
    parser.add_argument(
        "--run-specific",
        type=str,
        choices=["ground_truth", "hnsw", "lsh", "lshforest"],
        default=None,
        help="Run a specific algorithm.",
    )
    parser.add_argument(
        "--output",
        default="jaccard_topk_benchmark.sqlite",
        help="The output SQLite3 database to write results.",
    )
    parser.add_argument(
        "--cache-minhash",
        action="store_true",
        help="Cache minhashes of sets to speed up the benchmark.",
    )
    args = parser.parse_args(sys.argv[1:])

    # Global parameters.
    k = 100

    # LSH parameters, num_perm = b * r
    bs = [4, 8, 16, 32, 64, 128, 256, 512]
    rs = [2, 4, 8]

    # HNSW Index parameters.
    Ms = [4, 8, 12, 16, 24, 36]
    efCs = [100, 500, 1000, 1500]
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
    (index_sets, index_keys) = read_sets_from_file(
        args.index_set_file, sample_ratio=args.index_sample_ratio, skip=1
    )
    (query_sets, query_keys) = read_sets_from_file(
        args.index_set_file, sample_ratio=args.query_sample_ratio, skip=1
    )

    # Initialize output SQLite database.
    init_results_db(args.output)

    # Run ground truth.
    if args.run_specific is None or args.run_specific == "ground_truth":
        params = {"benchmark": benchmark_settings}
        if get_run("ground_truth", k, None, params, args.output) is None:
            print("Running Ground Truth.")
            (
                groupd_truth_indexing_time,
                ground_truth_results,
                ground_truth_times,
            ) = search_jaccard_topk(
                (index_sets, index_keys), (query_sets, query_keys), k
            )
            save_results(
                "ground_truth",
                k,
                None,
                params,
                groupd_truth_indexing_time,
                ground_truth_results,
                ground_truth_times,
                args.output,
            )

    # Run HNSW
    if args.run_specific is None or args.run_specific == "hnsw":
        for M in Ms:
            for efC in efCs:
                index_params = {
                    "M": M,
                    "indexThreadQty": num_threads,
                    "efConstruction": efC,
                    "post": 0,
                }
                params = {"index": index_params, "benchmark": benchmark_settings}
                if get_run("hnsw", k, None, params, args.output) is not None:
                    continue
                print(f"Running HNSW using M = {M}, efC = {efC}")
                hnsw_indexing_time, hnsw_results, hnsw_times = search_hnsw_jaccard_topk(
                    (index_sets, index_keys), (query_sets, query_keys), index_params, k
                )
                # Save results
                save_results(
                    "hnsw",
                    k,
                    None,
                    params,
                    hnsw_indexing_time,
                    hnsw_results,
                    hnsw_times,
                    args.output,
                )

    # Initialize storage for Minhashes.
    index_minhashes = {}
    query_minhashes = {}

    # Run LSH.
    if args.run_specific is None or args.run_specific == "lsh":
        for b in sorted(bs):
            # Clean up cached MinHashes as bs is in ascending order.
            for num_perm in list(index_minhashes.keys()):
                if num_perm < b * min(rs):
                    del index_minhashes[num_perm]
            for r in rs:
                params = {
                    "index": {"b": b, "r": r},
                    "benchmark": benchmark_settings,
                }
                if get_run("lsh", k, None, params, args.output) is not None:
                    continue
                print(f"Running LSH using b = {b}, r = {r}.")
                # Lazily create MinHashes.
                if b * r not in index_minhashes:
                    start = time.perf_counter()
                    index_minhashes.update(
                        create_minhashes_from_sets(
                            index_sets,
                            [
                                b * r,
                            ],
                            hashfunc=farmhash.hash32,
                        )
                    )
                    index_minhashes_time = time.perf_counter() - start
                    params["index_minhashes_time"] = index_minhashes_time
                    print(f"Index MinHashes took {index_minhashes_time} seconds.")
                else:
                    params["index_minhashes_time"] = 0
                if b * r not in query_minhashes:
                    start = time.perf_counter()
                    query_minhashes.update(
                        create_minhashes_from_sets(
                            query_sets,
                            [
                                b * r,
                            ],
                            hashfunc=farmhash.hash32,
                        )
                    )
                    query_minhashes_time = time.perf_counter() - start
                    params["query_minhashes_time"] = query_minhashes_time
                    print(f"Query MinHashes took {query_minhashes_time} seconds.")
                else:
                    params["query_minhashes_time"] = 0
                # Run benchmark.
                indexing_time, results, times = search_lsh_jaccard_topk(
                    (index_sets, index_keys, index_minhashes),
                    (query_sets, query_keys, query_minhashes),
                    b,
                    r,
                    k,
                )
                # Save result to SQLite database.
                save_results(
                    "lsh", k, None, params, indexing_time, results, times, args.output
                )
                if not args.cache_minhash:
                    index_minhashes.clear()
                    query_minhashes.clear()

    # Run LSH Forest.
    if args.run_specific is None or args.run_specific == "lshforest":
        for b in sorted(bs):
            # Clean up cached MinHashes as bs is in ascending order.
            for num_perm in list(index_minhashes.keys()):
                if num_perm < b * min(rs):
                    del index_minhashes[num_perm]
            for r in rs:
                params = {
                    "index": {"b": b, "r": r},
                    "benchmark": benchmark_settings,
                }
                if get_run("lshforest", k, None, params, args.output) is not None:
                    continue
                print(f"Running LSH Forest using b = {b}, r = {r}.")
                # Lazily create MinHashes.
                if b * r not in index_minhashes:
                    start = time.perf_counter()
                    index_minhashes.update(
                        create_minhashes_from_sets(
                            index_sets,
                            [
                                b * r,
                            ],
                            hashfunc=farmhash.hash32,
                        )
                    )
                    index_minhashes_time = time.perf_counter() - start
                    print(f"Index MinHashes took {index_minhashes_time} seconds.")
                    params["index_minhashes_time"] = index_minhashes_time
                else:
                    params["index_minhashes_time"] = 0
                if b * r not in query_minhashes:
                    start = time.perf_counter()
                    query_minhashes.update(
                        create_minhashes_from_sets(
                            query_sets,
                            [
                                b * r,
                            ],
                            hashfunc=farmhash.hash32,
                        )
                    )
                    query_minhashes_time = time.perf_counter() - start
                    print(f"Query MinHashes took {query_minhashes_time} seconds.")
                    params["query_minhashes_time"] = query_minhashes_time
                else:
                    params["query_minhashes_time"] = 0
                # Run benchmark.
                indexing_time, results, times = search_lshforest_jaccard_topk(
                    (index_sets, index_keys, index_minhashes),
                    (query_sets, query_keys, query_minhashes),
                    b,
                    r,
                    k,
                )
                # Save result to SQLite database.
                save_results(
                    "lshforest",
                    k,
                    None,
                    params,
                    indexing_time,
                    results,
                    times,
                    args.output,
                )
                if not args.cache_minhash:
                    index_minhashes.clear()
                    query_minhashes.clear()
