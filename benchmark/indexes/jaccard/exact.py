import json
import time
import sys
import collections

from SetSimilaritySearch import SearchIndex
import tqdm


def _query_jaccard_topk(index, query, k):
    """Query the search index for the best k candidates."""
    assert index.similarity_threshold == 0.0
    s1 = [index.order[token] for token in query if token in index.order]
    # Get the number of occurrances of candidates in the posting lists.
    counter = collections.Counter(i for token in s1 for i, _ in index.index[token])
    # Compute the Jaccard similarities based on the counts.
    candidates = [
        (i, float(c) / float(len(s1) + len(index.sets[i]) - c))
        for (i, c) in counter.items()
    ]
    # Sort candidates based on similarities.
    candidates.sort(key=lambda x: x[1], reverse=True)
    # Return the top-k candidates.
    return candidates[:k]


def search_jaccard_topk(index_data, query_data, index_params, k):
    (index_sets, index_keys, _, index_cache) = index_data
    (query_sets, query_keys, _) = query_data
    cache_key = json.dumps(index_params)
    if cache_key not in index_cache:
        print("Building jaccard search index.")
        start = time.perf_counter()
        # Build the search index with the 0 threshold to index all tokens.
        index = SearchIndex(
            index_sets, similarity_func_name="jaccard", similarity_threshold=0.0
        )
        indexing_time = time.perf_counter() - start
        print("Finished building index in {:.3f}.".format(indexing_time))
        index_cache[cache_key] = (
            index,
            {
                "indexing_time": indexing_time,
            },
        )
    index, indexing = index_cache[cache_key]
    times = []
    results = []
    for query_set, query_key in tqdm.tqdm(
        zip(query_sets, query_keys), total=len(query_keys), desc="Querying", unit=" set"
    ):
        start = time.perf_counter()
        result = [
            [index_keys[i], similarity]
            for i, similarity in _query_jaccard_topk(index, query_set, k)
        ]
        duration = time.perf_counter() - start
        times.append(duration)
        results.append((query_key, result))
    return (indexing, results, times)
