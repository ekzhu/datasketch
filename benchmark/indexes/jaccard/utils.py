import sys
import gzip
import random
import collections
import sqlite3
import json

import numpy as np

from datasketch import MinHash


def read_sets_from_file(sets_file, sample_ratio, skip=1):
    """Read sets from set-similarity-search benchmark:
    https://github.com/ekzhu/set-similarity-search-benchmark.

    Args:
        sets_file: the input file for reading sets from, 
            must be Gzip-compressed.
        sample_ration: the fraction of sets actually read from
            the input file.
        skip: the number of lines from the beginning of the file 
            to skip.

    Returns: (sets, keys)
    """
    sets = collections.deque([])
    keys = collections.deque([])
    random.seed(41)
    with gzip.open(sets_file, "rt") as f:
        for i, line in enumerate(f):
            if i < skip:
                # Skip lines
                continue
            if random.random() > sample_ratio:
                continue
            # Take tokens after the \t character splitting by ,.
            s = np.array([int(d) for d in \
                    line.strip().split("\t")[1].split(",")])
            sets.append(s)
            # Use the line number as the key.
            keys.append(i)
            sys.stdout.write("\r{} sets.".format(len(sets)))
        sys.stdout.write("\n")
    sets = list(sets)
    keys = list(keys)
    return (sets, keys)


def create_minhashes_from_sets(sets, num_perms, hashfunc, pad_for_asym=False):
    # Generate paddings for asym.
    max_size = max(len(s) for s in sets)
    paddings = dict()
    if pad_for_asym:
        padding_sizes = sorted(list(set([max_size-len(s) for s in sets])))
        for num_perm in num_perms:
            paddings[num_perm] = dict()
            for i, padding_size in enumerate(padding_sizes):
                if i == 0:
                    prev_size = 0
                    pad = MinHash(num_perm, hashfunc=hashfunc)
                else:
                    prev_size = padding_sizes[i-1]
                    pad = paddings[num_perm][prev_size].copy()
                for w in range(prev_size, padding_size):
                    pad.update(str(w)+"_tmZZRe8DE23s")
                paddings[num_perm][padding_size] = pad
    # Generate minhash
    minhashes = dict()
    for num_perm in num_perms:
        print("Using num_perm = {}".format(num_perm))
        ms = []
        for s in sets:
            m = MinHash(num_perm, hashfunc=hashfunc)
            for word in s:
                m.update(str(word))
            if pad_for_asym:
                # Add padding to the minhash
                m.merge(paddings[num_perm][max_size-len(s)])
            ms.append(m)
            sys.stdout.write("\rMinhashed {} sets".format(len(ms)))
        sys.stdout.write("\n")
        minhashes[num_perm] = ms
    return minhashes


def compute_containment(x, y):
    if len(x) == 0 or len(y) == 0:
        return 0.0
    intersection = len(np.intersect1d(x, y, assume_unique=True))
    return float(intersection) / float(len(x))


def compute_jaccard(x, y):
    if len(x) == 0 or len(y) == 0:
        return 0.0
    intersection = len(np.intersect1d(x, y, assume_unique=True))
    return float(intersection) / float(len(x) + len(y) - intersection)


def init_results_db(output_sqlite):
    conn = sqlite3.connect(output_sqlite)
    cursor = conn.cursor()
    cursor.execute("""CREATE TABLE IF NOT EXISTS runs (
        key integer primary key,
        name text not null,
        time datetime not null,
        k integer,
        threshold real,
        params text not null
    )""")
    cursor.execute("""CREATE TABLE IF NOT EXISTS results (
        run_key integer not null,
        query_key integer not null,
        result text not null,
        time real not null,
        FOREIGN KEY(run_key) REFERENCES runs(key)
    )""")
    cursor.execute("""CREATE INDEX IF NOT EXISTS run_key_idx 
            on results(run_key)""")
    conn.commit()
    conn.close()


def save_results(run_name, k, threshold, params, results, times, 
        output_sqlite):
    conn = sqlite3.connect(output_sqlite)
    cursor = conn.cursor()
    cursor.execute("""INSERT INTO runs 
            (name, time, k, threshold, params)
            VALUES (?, datetime('now'), ?, ?, ?)""", 
            (run_name, k, threshold, json.dumps(params)))
    cursor.execute("SELECT last_insert_rowid()")
    run_key = cursor.fetchone()[0]
    rows = ([run_key, result[0], json.dumps(result[1]), time] 
            for result, time in zip(results, times))
    cursor.executemany("""INSERT INTO results 
        (run_key, query_key, result, time)
        VALUES (?, ?, ?, ?)""", rows)
    conn.commit()
    conn.close()


def load_results(run_key, conn):
    cursor = conn.cursor()
    cursor.execute("""SELECT query_key, result, time FROM results
            WHERE run_key = ?""", (run_key,))
    results = []
    times = []
    for query_key, result, time in cursor:
        results.append((query_key, json.loads(result)))
        times.append(time)
    cursor.close()
    return (results, times)


def _compute_recall(result, ground):
    result_keys = [x[0] for x in result]
    ground_keys = [x[0] for x in ground]
    intersection = len(np.intersect1d(result_keys, ground_keys))
    return float(intersection) / float(len(ground_keys)) 


def compute_recalls(results, grounds):
    results.sort(key=lambda x: x[0])
    grounds.sort(key=lambda x: x[0])
    recalls = []
    for (query_key_1, result), (query_key_2, ground) in zip(results, grounds):
        assert(query_key_1 == query_key_2)
        recall = _compute_recall(result, ground)
        recalls.append((query_key_1, recall))
    return recalls


def compute_similarities(results):
    similarities = []
    for (query_key_1, result) in results:
        similarity = np.mean([x[1] for x in result])
        similarities.append((query_key_1, similarity))
    return similarities


def get_run(name, k, threshold, params, result_sqlite):
    conn = sqlite3.connect(result_sqlite)
    cursor = conn.cursor()
    cursor.execute("""SELECT key, k, threshold, params 
            FROM runs WHERE name = ?""", (name,))
    runs = [row[0] for row in cursor 
            if row[1] == k and row[2] == threshold 
            and json.loads(row[3]) == params]
    conn.close()
    if len(runs) > 0:
        return runs[0]
    return None


def evaluate_runs(result_sqlite, names=None):
    conn = sqlite3.connect(result_sqlite)
    cursor = conn.cursor()
    if not names:
        cursor.execute("""SELECT key, name, k, threshold, params FROM runs""")
    else:
        cursor.execute("""SELECT key, name, k, threshold, params 
                FROM runs 
                WHERE name IN ? OR name == 'ground_truth""", (names,))
    runs = [{
                "key": key,
                "name": name, 
                "k": k, 
                "threshold": threshold,
                **json.loads(params),
            }
            for (key, name, k, threshold, params) in cursor]
    cursor.close()

    # TODO: add filter to avoid hard-code.
    runs = [run for run in runs
            if run["name"] != 'lsh' or 
            (run["name"] == 'lsh' and run["index"]["r"] > 1)]

    # Get ground truth results first.
    for i, run in enumerate([run for run in runs 
            if run["name"] == "ground_truth"]):
        results, times = load_results(run["key"], conn)
        run.update({"results": results, "times": times})
    
    # Compute mean recall and query time of every run.
    for i, run in enumerate([run for run in runs 
            if run["name"] != "ground_truth"]):
        # Load results of this run.
        results, times = load_results(run["key"], conn)
        # Find the corresponding ground truth run with the same 
        # benchmark settings.
        ground_truth = [x for x in runs if x["name"] == "ground_truth" and
                x["benchmark"] == run["benchmark"]][0]
        # Compute metrics.
        recalls = compute_recalls(results, ground_truth["results"])
        similarities = compute_similarities(results)
        mean_recall = np.mean([x[1] for x in recalls])
        mean_similarity = np.mean([x[1] for x in similarities])
        mean_time = np.mean(times * 1000)
        # Update run with computed metrics.
        run.update({
            "mean_recall": mean_recall, 
            "mean_similarity": mean_similarity,
            "mean_time": mean_time,
        })
    conn.close()
    return runs