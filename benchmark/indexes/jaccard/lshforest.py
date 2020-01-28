import time
import sys

from datasketch import MinHashLSHForest

from utils import compute_jaccard


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

