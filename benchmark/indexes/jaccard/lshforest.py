import json
import time
import sys

import tqdm

from datasketch import MinHashLSHForest

from utils import compute_jaccard, lazy_create_minhashes_from_sets


def search_lshforest_jaccard_topk(index_data, query_data, index_params, k):
    (index_sets, index_keys, index_minhashes, index_cache) = index_data
    (query_sets, query_keys, query_minhashes) = query_data
    b, r = index_params["b"], index_params["r"]
    num_perm = b * r
    cache_key = json.dumps(index_params)
    if cache_key not in index_cache:
        # Create minhashes
        index_minhash_time, query_minhash_time = lazy_create_minhashes_from_sets(
            index_minhashes,
            index_sets,
            query_minhashes,
            query_sets,
            num_perm,
        )
        print("Building LSH Forest Index.")
        start = time.perf_counter()
        index = MinHashLSHForest(num_perm=num_perm, l=b)
        # Use the indices of the indexed sets as keys in LSH.
        for i in tqdm.tqdm(
            range(len(index_keys)),
            desc="Queuing ",
            unit=" query",
            total=len(index_keys),
        ):
            index.add(i, index_minhashes[num_perm][i])
        print(f"Indexing {len(index_keys)} minhashes...")
        index.index()
        indexing_time = time.perf_counter() - start
        print("Indexing time: {:.3f}.".format(indexing_time))
        index_cache[cache_key] = (
            index,
            {
                "index_minhash_time": index_minhash_time,
                "query_minhash_time": query_minhash_time,
                "indexing_time": indexing_time,
            },
        )
    index, indexing = index_cache[cache_key]
    print("Querying.")
    times = []
    results = []
    for query_minhash, query_key, query_set in tqdm.tqdm(
        zip(query_minhashes[num_perm], query_keys, query_sets),
        total=len(query_keys),
        desc="Querying",
        unit=" minhash",
    ):
        start = time.perf_counter()
        result = index.query(query_minhash, k * 2)
        # Recover the retrieved indexed sets and
        # compute the exact Jaccard similarities.
        result = [
            [index_keys[i], compute_jaccard(query_set, index_sets[i])] for i in result
        ]
        # Sort by similarity.
        result.sort(key=lambda x: x[1], reverse=True)
        # Take the top k.
        result = result[:k]
        duration = time.perf_counter() - start
        times.append(duration)
        results.append((query_key, result))
    return (indexing, results, times)
