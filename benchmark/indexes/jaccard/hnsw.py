import time
import sys
from datasketch.hnsw import HNSW

from utils import compute_jaccard


def search_hnsw_jaccard_topk(index_data, query_data, index_params, k):
    import nmslib

    (index_sets, index_keys) = index_data
    (query_sets, query_keys) = query_data
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
    print("Querying.")
    times = []
    results = []
    index.setQueryTimeParams({"efSearch": index_params["efConstruction"]})
    for query_set, query_key in zip(query_sets, query_keys):
        start = time.perf_counter()
        result, _ = index.knnQuery(" ".join(str(v) for v in query_set), k)
        result = [
            [index_keys[i], compute_jaccard(query_set, index_sets[i])] for i in result
        ]
        result.sort(key=lambda x: x[1], reverse=True)
        duration = time.perf_counter() - start
        times.append(duration)
        results.append((query_key, result))
        sys.stdout.write(f"\rQueried {len(results)} sets")
    sys.stdout.write("\n")
    return (indexing_time, results, times)


def minhash_jaccard_distance(x, y) -> float:
    return 1.0 - x.jaccard(y)


def search_hnsw_minhash_jaccard_topk(index_data, query_data, index_params, k):
    (_, index_keys, index_minhashes) = index_data
    (_, query_keys, query_minhashes) = query_data
    num_perm = index_params.pop("num_perm")
    print("Building HNSW Index for MinHash.")
    start = time.perf_counter()
    index = HNSW(distance_func=minhash_jaccard_distance, **index_params)
    index.update({index_keys[i]: index_minhashes[i] for i in range(len(index_keys))})
    indexing_time = time.perf_counter() - start
    print("Indexing time: {:.3f}.".format(indexing_time))
    print("Querying.")
    times = []
    results = []
    for query_minhash, query_key in zip(query_minhashes[num_perm], query_keys):
        start = time.perf_counter()
        result = index.query(query_minhash, k)
        # Convert distances to similarities.
        result = [(key, 1.0 - dist) for key, dist in result]
        duration = time.perf_counter() - start
        times.append(duration)
        results.append((query_key, result))
        sys.stdout.write(f"\rQueried {len(results)} sets")
    sys.stdout.write("\n")
    return (indexing_time, results, times)
