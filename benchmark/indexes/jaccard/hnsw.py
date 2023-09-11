import json
import time
import sys

import tqdm
from datasketch.hnsw import HNSW

from utils import (
    compute_jaccard,
    compute_jaccard_distance,
    compute_minhash_jaccard_distance,
    lazy_create_minhashes_from_sets,
)


def search_nswlib_jaccard_topk(index_data, query_data, index_params, k):
    import nmslib

    (index_sets, index_keys, _, index_cache) = index_data
    (query_sets, query_keys, _) = query_data
    cache_key = json.dumps(index_params)
    if cache_key not in index_cache:
        print("Building HNSW Index.")
        start = time.perf_counter()
        index = nmslib.init(
            method="hnsw",
            space="jaccard_sparse",
            data_type=nmslib.DataType.OBJECT_AS_STRING,
        )
        index.addDataPointBatch(
            [" ".join(str(v) for v in s) for s in index_sets], range(len(index_keys))
        )
        index.createIndex(index_params)
        indexing_time = time.perf_counter() - start
        print("Indexing time: {:.3f}.".format(indexing_time))
        index_cache[cache_key] = (
            index,
            {
                "indexing_time": indexing_time,
            },
        )
    index, indexing = index_cache[cache_key]
    print("Querying.")
    times = []
    results = []
    index.setQueryTimeParams({"efSearch": index_params["efConstruction"]})
    for query_set, query_key in tqdm.tqdm(
        zip(query_sets, query_keys),
        total=len(query_keys),
        desc="Querying",
        unit=" query",
    ):
        start = time.perf_counter()
        result, _ = index.knnQuery(" ".join(str(v) for v in query_set), k)
        result = [
            [index_keys[i], compute_jaccard(query_set, index_sets[i])] for i in result
        ]
        result.sort(key=lambda x: x[1], reverse=True)
        duration = time.perf_counter() - start
        times.append(duration)
        results.append((query_key, result))
    return (indexing, results, times)


def search_hnsw_jaccard_topk(index_data, query_data, index_params, k):
    (index_sets, index_keys, _, index_cache) = index_data
    (query_sets, query_keys, _) = query_data
    cache_key = json.dumps(index_params)
    if cache_key not in index_cache:
        print("Building HNSW Index.")
        start = time.perf_counter()
        index = HNSW(distance_func=compute_jaccard_distance, **index_params)
        for i in tqdm.tqdm(
            range(len(index_keys)),
            desc="Indexing",
            unit=" set",
            total=len(index_keys),
        ):
            index.insert(i, index_sets[i])
        indexing_time = time.perf_counter() - start
        print("Indexing time: {:.3f}.".format(indexing_time))
        index_cache[cache_key] = (
            index,
            {
                "indexing_time": indexing_time,
            },
        )
    index, indexing = index_cache[cache_key]
    print("Querying.")
    times = []
    results = []
    for query_set, query_key in tqdm.tqdm(
        zip(query_sets, query_keys),
        total=len(query_keys),
        desc="Querying",
        unit=" query",
    ):
        start = time.perf_counter()
        result = index.query(query_set, k)
        # Convert distances to similarities.
        result = [(index_keys[i], 1.0 - dist) for i, dist in result]
        duration = time.perf_counter() - start
        times.append(duration)
        results.append((query_key, result))
    return (indexing, results, times)


def search_hnsw_minhash_jaccard_topk(index_data, query_data, index_params, k):
    (index_sets, index_keys, index_minhashes, index_cache) = index_data
    (query_sets, query_keys, query_minhashes) = query_data
    num_perm = index_params["num_perm"]
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
        print("Building HNSW Index for MinHash.")
        start = time.perf_counter()
        kwargs = index_params.copy()
        kwargs.pop("num_perm")
        index = HNSW(distance_func=compute_minhash_jaccard_distance, **kwargs)
        for i in tqdm.tqdm(
            range(len(index_keys)),
            desc="Indexing",
            unit=" minhash",
            total=len(index_keys),
        ):
            index.insert(i, index_minhashes[num_perm][i])
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
        unit=" query",
    ):
        start = time.perf_counter()
        result = index.query(query_minhash, k)
        # Recover the retrieved indexed sets and
        # compute the exact Jaccard similarities.
        result = [
            [index_keys[i], compute_jaccard(query_set, index_sets[i])]
            for i, _ in result
        ]
        # Sort by similarity.
        result.sort(key=lambda x: x[1], reverse=True)
        duration = time.perf_counter() - start
        times.append(duration)
        results.append((query_key, result))
    return (indexing, results, times)
