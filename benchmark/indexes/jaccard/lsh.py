import time

from datasketch import MinHashLSH

from utils import compute_jaccard


def search_lsh_jaccard_topk(index_data, query_data, b, r, k):
    (index_sets, index_keys, index_minhashes) = index_data
    (query_sets, query_keys, query_minhashes) = query_data
    num_perm = b * r
    print("Building LSH Index.")
    start = time.perf_counter()
    index = MinHashLSH(params=(b, r))
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
        # Sort by similarity.
        result.sort(key=lambda x : x[1], reverse=True)
        # Take the first k.
        result = result[:k]
        duration = time.perf_counter() - start
        times.append(duration)
        results.append((query_key, result))
    return (results, times)


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

