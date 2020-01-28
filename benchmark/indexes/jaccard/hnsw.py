import time
import sys

import nmslib

from utils import compute_jaccard


def search_hnsw_jaccard_topk(index_data, query_data, index_params, k):
    (index_sets, index_keys) = index_data
    (query_sets, query_keys) = query_data
    print("Building HNSW Index.")
    start = time.perf_counter()
    index = nmslib.init(method="hnsw", space="jaccard_sparse", 
            data_type=nmslib.DataType.OBJECT_AS_STRING)
    index.addDataPointBatch(
            [" ".join(str(v) for v in s) for s in index_sets],
            range(len(index_keys)))
    index.createIndex(index_params)
    end = time.perf_counter()
    print("Indexing time: {:.3f}.".format(end-start))
    print("Querying.")
    times = []
    results = []
    index.setQueryTimeParams({"efSearch": index_params["efConstruction"]})
    for query_set, query_key in zip(query_sets, query_keys):
        start = time.perf_counter()
        result, _ = index.knnQuery(" ".join(str(v) for v in query_set), k)
        result = [[index_keys[i], compute_jaccard(query_set, index_sets[i])] 
                for i in result]
        result.sort(key=lambda x : x[1], reverse=True)
        duration = time.perf_counter() - start
        times.append(duration)
        results.append((query_key, result))
        sys.stdout.write("\rQueried {} sets".format(len(results)))
    sys.stdout.write("\n")
    return (results, times)

