import time, argparse, sys, json
from hashlib import sha1
import numpy as np
import nltk
import scipy.stats
import random
import collections
from datasketch import MinHashLSHForest, MinHash

def bootstrap_data(num_perms, n, population_size, set_size_dist):
    random.seed(42)
    print("Creating sets...")
    population = [str(i) for i in range(population_size)]
    sets = [set(random.sample(population, set_size_dist.rvs()))
            for _ in range(n)]
    keys = list(range(len(sets)))
    print("Creating minhashes...")
    minhashes = dict()
    for num_perm in num_perms:
        ms = []
        for s in sets:
            m = MinHash(num_perm)
            for word in s:
                m.update(word.encode("utf8"))
            ms.append(m)
        minhashes[num_perm] = ms
    Data = collections.namedtuple('Data', ['minhashes', 'sets', 'keys']) 
    index_data = Data(minhashes, sets, keys)
    query_indices = random.sample(list(range(len(sets))), int(len(sets)*0.1))
    if len(query_indices) == 0:
        raise RuntimeError("Empty query sets")
    query_data = Data(dict((num_perm, [minhashes[num_perm][i] for i in query_indices])
                           for num_perm in num_perms),
                      [sets[i] for i in query_indices],
                      [keys[i] for i in query_indices])
    return index_data, query_data


def get_twitter_data(num_perms, subset):
    nltk.download("stopwords")
    nltk.download("twitter_samples")
    stopwords = set(nltk.corpus.stopwords.words("english"))
    tweets = nltk.corpus.twitter_samples.tokenized('tweets.20150430-223406.json')[:subset]
    minhashes = []
    sets = []
    print("Creating sets...")
    for i, text in enumerate(tweets):
        s = set(w for w in text 
                if w not in stopwords)
        if len(s) == 0:
            continue
        sets.append(s)
    keys = list(range(len(sets)))
    print("Creating MinHash...")
    minhashes = dict()
    for num_perm in num_perms:
        ms = []
        for s in sets:
            m = MinHash(num_perm)
            for word in s:
                m.update(word.encode("utf8"))
            ms.append(m)
        minhashes[num_perm] = ms
    Data = collections.namedtuple('Data', ['minhashes', 'sets', 'keys']) 
    index_data = Data(minhashes, sets, keys)
    query_indices = random.sample(list(range(len(sets))), int(len(sets)*0.1))
    if len(query_indices) == 0:
        raise RuntimeError("Empty query sets")
    query_data = Data(dict((num_perm, [minhashes[num_perm][i] for i in query_indices])
                           for num_perm in num_perms),
                      [sets[i] for i in query_indices],
                      [keys[i] for i in query_indices])
    return index_data, query_data


def benchmark_lshforest(num_perm, l, k, index_data, query_data):
    print("Building LSH Forest index")
    forest = MinHashLSHForest(num_perm=num_perm, l=l)
    for key, minhash in zip(index_data.keys, index_data.minhashes[num_perm]):
        forest.add(key, minhash)
    forest.index()
    print("Querying")
    times = []
    results = []
    for qs, minhash in zip(query_data.sets, query_data.minhashes[num_perm]):
        start = time.clock()
        result = forest.query(minhash, k)
        duration = time.clock() - start
        times.append(duration)
        results.append(sorted([[key, _compute_jaccard(qs, index_data.sets[key])]
                               for key in result], 
                              key=lambda x : x[1], reverse=True))
    return times, results


def benchmark_linearscan(num_perm, k, index_data, query_data):
    times = []
    results = []
    for qs, q in zip(query_data.sets, query_data.minhashes[num_perm]):
        start = time.clock()
        result = []
        result = [(key, q.jaccard(m))
                  for key, m in zip(index_data.keys, index_data.minhashes[num_perm])]
        result.sort(key=lambda x : x[1], reverse=True)
        result = [x[0] for x in result[:k]]
        duration = time.clock() - start
        times.append(duration)
        results.append(sorted([[key, _compute_jaccard(qs, index_data.sets[key])]
                               for key in result], 
                              key=lambda x : x[1], reverse=True))
    return times, results


def benchmark_ground_truth(k, index_data, query_data):
    times = []
    results = []
    # use less decimal precision for collecting top-k ground truth
    truncate_decimal = lambda x : float(int(x*100)) / 100.0
    for q in query_data.sets:
        start = time.clock()
        result = [(key, _compute_jaccard(q, a)) 
                  for key, a in zip(index_data.keys, index_data.sets)]
        result.sort(key=lambda x : x[1], reverse=True)
        duration = time.clock() - start
        topk_result = [] 
        curr_rank = 0
        curr_j2 = -1.0
        for key, j in result:
            j2 = truncate_decimal(j)
            if j2 != curr_j2:
                curr_j2 = j2
                curr_rank += 1
            if curr_rank > k:
                break
            topk_result.append([key, j])
        assert(len(set(truncate_decimal(x[1]) for x in topk_result)) == k)
        results.append(topk_result)
        times.append(duration)
    return times, results


def _compute_jaccard(x, y):
    if len(x) == 0 or len(y) == 0:
        return 0.0
    intersection = 0
    for w in x:
        if w in y:
            intersection += 1
    return float(intersection) / float(len(x) + len(y) - intersection)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="lshforest_benchmark.json")
    args = parser.parse_args(sys.argv[1:])

    num_perms = [32, 64, 96, 128, 160, 192, 224, 256]
    l = 8
    k = 10
    output = {"num_perms" : num_perms,
              "l" : l,
              "k" : k,
              "lsh_times" : [], "lsh_results" : [],
              "linearscan_times" : [], "linearscan_results" : [],
              "ground_truth_times" : None, "ground_truth_results" : None}
    index_data, query_data = bootstrap_data(num_perms, 10000, 5000, 
            scipy.stats.randint(10, 5000))

    for num_perm in num_perms:
        print("Use num_perm = %d" % num_perm)
        result = {}
        print("Running LSH benchmark l = %d" % l)
        lsh_times, lsh_results = benchmark_lshforest(num_perm, l, k, index_data, query_data)
        print("Running linear scan benchmark")
        linearscan_times, linearscan_results = benchmark_linearscan(num_perm, k, index_data, query_data)
        output["lsh_times"].append(lsh_times)
        output["lsh_results"].append(lsh_results)
        output["linearscan_times"].append(linearscan_times)
        output["linearscan_results"].append(linearscan_results)
    
    print("Running ground truth benchmark")
    output["ground_truth_times"], output["ground_truth_results"] =\
            benchmark_ground_truth(k, index_data, query_data)

    average_cardinality = np.mean([len(s) for s in
        index_data.sets + query_data.sets])
    print("Average cardinality is", average_cardinality)

    with open(args.output, 'w') as f:
        json.dump(output, f)
