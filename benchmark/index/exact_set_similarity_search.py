import time
import sys

from SetSimilaritySearch import SearchIndex


def search_jaccard_threshold(index_data, query_data, threshold):
    (index_sets, index_keys) = index_data
    print("Building jaccard search index for threshold = {}".format(threshold))
    start = time.perf_counter()
    index = SearchIndex(index_sets, similarity_func_name="jaccard",
            similarity_threshold=threshold)
    duration = time.perf_counter() - start
    print("Finished building index in {:.3f}.".format(duration)) 
    (query_sets, query_keys) = query_data
    times = []
    results = []
    for query_set, query_key in zip(query_sets, query_keys):
        start = time.perf_counter()
        # Obtain the search results and recover the original keys.
        result = [[index_keys[i], similarity] 
                for i, similarity in index.query(query_set)]
        duration = time.perf_counter() - start
        times.append(duration)
        results.append((query_key, result))
        sys.stdout.write("\rQueried {} sets.".format(len(results)))
    sys.stdout.write("\n")
    return (results, times)


def search_jaccard_topk(index_data, query_data, k, start_threshold=1.0,
        threshold_decrement=0.05, minimum_threshold=0.01):
    (index_sets, index_keys) = index_data
    (query_sets, query_keys) = query_data
    print("Building jaccard search index.")
    start = time.perf_counter()
    # Build the search index with the 0 threshold first to index all tokens.
    index = SearchIndex(index_sets, similarity_func_name="jaccard",
            similarity_threshold=0.0)
    duration = time.perf_counter() - start
    print("Finished building index in {:.3f}.".format(duration)) 
    times = []
    results = []
    for query_set, query_key in zip(query_sets, query_keys):
        start = time.perf_counter()
        # Reset the threshold to a higher value for top-k.
        index.similarity_threshold = start_threshold
        result = []
        while len(result) < k and index.similarity_threshold >= minimum_threshold:
            index.similarity_threshold -= threshold_decrement
            # Obtain the search results and recover the original keys.
            result = [[index_keys[i], similarity] 
                    for i, similarity in index.query(query_set)]
        result.sort(key=lambda x : x[1], reverse=True)
        result = result[:k]
        duration = time.perf_counter() - start
        if len(result) < k:
            print("Found less than k results.")
        times.append(duration)
        results.append((query_key, result))
        sys.stdout.write("\rQueried {} sets.".format(len(results)))
    sys.stdout.write("\n")
    return (results, times)

