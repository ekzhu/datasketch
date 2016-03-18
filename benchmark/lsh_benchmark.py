import time, argparse, sys, json
from hashlib import sha1
from sklearn.datasets import fetch_20newsgroups
import numpy as np
from datasketch import MinHashLSH, MinHash


def get_ngrams(text, n=3):
    onegrams = text.split()
    if n == 1:
        return onegrams
    if len(onegrams) < n:
        return text
    return [" ".join(onegrams[i:i+n]) for i in range(0, len(onegrams)-n+1)]


def get_newsgroup_data(num_perm, subset):
    print("Downloading 20 News Group Text Data...")
    newsgroup = fetch_20newsgroups(subset=subset,
            #categories=["sci.space"],
            remove=("header", "footers", "quotes"))
    print("Finished download, creating MinHash...")
    minhashes = [None for _ in range(len(newsgroup.data))]
    shingles = [None for _ in range(len(newsgroup.data))]
    for i, text in enumerate(newsgroup.data):
        minhashes[i] = MinHash(num_perm)
        shingles[i] = set(get_ngrams(text))
        for ngram in shingles[i]:
            minhashes[i].update(ngram.encode("utf8"))
    newsgroup.minhashes = minhashes
    newsgroup.shingles = shingles
    return newsgroup


def benchmark_lsh(threshold, index_data, query_data):
    print("Building LSH index")
    num_perm = len(index_data.minhashes[0].hashvalues)
    lsh = MinHashLSH(threshold, num_perm)
    for key, minhash in zip(index_data.filenames, index_data.minhashes):
        lsh.insert(key, minhash)
    print("Querying")
    times = []
    results = []
    for minhash in query_data.minhashes:
        start = time.clock()
        result = lsh.query(minhash)
        duration = time.clock() - start
        times.append(duration)
        results.append(result)
    return times, results


def benchmark_linearscan(threshold, index_data, query_data):
    times = []
    results = []
    for q in query_data.minhashes:
        start = time.clock()
        result = []
        for key, m in zip(index_data.filenames, index_data.minhashes):
            j = q.jaccard(m)
            if j >= threshold:
                result.append(key)
        duration = time.clock() - start
        times.append(duration)
        results.append(result)
    return times, results


def benchmark_groud_truth(threshold, index_data, query_data):
    times = []
    results = []
    for q in query_data.shingles:
        start = time.clock()
        result = []
        for key, a in zip(index_data.filenames, index_data.shingles):
            intersection = 0
            for w in a:
                if w in q:
                    intersection += 1
            j = float(intersection) / float(len(a) + len(q) - intersection)
            if j >= threshold:
                result.append(key)
        duration = time.clock() - start
        results.append(result)
    return times, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="lsh_benchmark.json")
    args = parser.parse_args(sys.argv[1:])

    num_perms = [32, 64, 96, 128, 160, 192, 224, 256]
    output = {"num_perms" : num_perms,
              "lsh_times" : [], "lsh_results" : [],
              "linearscan_times" : [], "linearscan_results" : [],
              "ground_truth_times" : None, "ground_truth_results" : None}

    for num_perm in num_perms:
        print("Use num_perm = %d" % num_perm)
        index_data = get_newsgroup_data(num_perm, "train")
        query_data = get_newsgroup_data(num_perm, "test")
        result = {}
        print("Running linear scan benchmark")
        linearscan_times, linearscan_results = benchmark_linearscan(0.5, index_data, query_data)
        print("Running LSH benchmark")
        lsh_times, lsh_results = benchmark_lsh(0.5, index_data, query_data)
        output["lsh_times"].append(lsh_times)
        output["lsh_results"].append(lsh_results)
        output["linearscan_times"].append(linearscan_times)
        output["linearscan_results"].append(linearscan_results)

    print("Running ground truth benchmark")
    output["ground_truth_times"], output["ground_truth_results"] =\
            benchmark_groud_truth(0.5, index_data, query_data)

    average_cardinality = np.mean([len(s) for s in
        index_data.shingles + query_data.shingles])
    print("Average cardinality is", average_cardinality)

    with open(args.output, 'w') as f:
        json.dump(output, f)
