import itertools
import sys
import argparse


from exact import search_jaccard_topk
from hnsw import (
    search_hnsw_jaccard_topk,
    search_hnsw_minhash_jaccard_topk,
    search_nswlib_jaccard_topk,
)
from lsh import search_lsh_jaccard_topk
from lshforest import search_lshforest_jaccard_topk
from utils import (
    read_sets_from_file,
    save_results,
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
        choices=["exact", "nswlib", "lsh", "lshforest", "hnsw", "hnsw_minhash"],
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

    # LSH parameters, num_perm = b * r
    bs = [2, 4, 8, 16, 32, 64]
    rs = [2, 4, 8, 16, 32, 64]
    lsh_params = list({"b": b, "r": r} for b, r in itertools.product(bs, rs))

    # HNSW Index parameters.
    Ms = [8, 16, 24, 32, 40]
    efCs = [
        100,
        200,
        300,
        400,
        500,
    ]
    hnsw_params = list(
        {
            "m": M,
            "ef_construction": efC,
        }
        for M, efC in itertools.product(Ms, efCs)
    )

    # NSWLib parameters.
    nswlib_params = list(
        {
            "M": M,
            "indexThreadQty": 1,
            "efConstruction": efC,
            "post": 0,
        }
        for M, efC in itertools.product(Ms, efCs)
    )

    # HNSW MinHash parameters.
    hnsw_minhash_num_perms = [128]
    hnsw_minhash_params = list(
        {"num_perm": num_perm, "m": M, "ef_construction": efC}
        for num_perm, M, efC in itertools.product(hnsw_minhash_num_perms, Ms, efCs)
    )

    index_configs = {
        "exact": [{}],
        "lsh": lsh_params,
        "lshforest": lsh_params,
        "hnsw": hnsw_params,
        "hnsw_minhash": hnsw_minhash_params,
        "nswlib": nswlib_params,
    }

    index_runners = {
        "exact": search_jaccard_topk,
        "lsh": search_lsh_jaccard_topk,
        "lshforest": search_lshforest_jaccard_topk,
        "hnsw": search_hnsw_jaccard_topk,
        "hnsw_minhash": search_hnsw_minhash_jaccard_topk,
        "nswlib": search_nswlib_jaccard_topk,
    }

    # Query settings.
    query_configs = [{"k": k} for k in [1, 5, 10, 50, 100]]

    # Benchmark settings.
    benchmark_config = {
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

    # Initialize storage for Minhashes.
    index_minhashes = {}
    query_minhashes = {}

    # Run algorithms.
    for name, configs in index_configs.items():
        if args.run_specific is not None and args.run_specific != name:
            continue
        for index_config in configs:
            index_cache = {}
            for query_config in query_configs:
                config = {
                    "index": index_config,
                    "query": query_config,
                    "benchmark": benchmark_config,
                }
                if get_run(name, config, args.output) is not None:
                    continue
                print(f"Running {name} using {index_config}.")
                index_runner = index_runners[name]
                indexing, results, times = index_runner(
                    (index_sets, index_keys, index_minhashes, index_cache),
                    (query_sets, query_keys, query_minhashes),
                    index_config,
                    query_config["k"],
                )
                save_results(name, config, indexing, results, times, args.output)
            if not args.cache_minhash:
                index_minhashes.clear()
                query_minhashes.clear()
