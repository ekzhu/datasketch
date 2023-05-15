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
            must be uncompressed.
        sample_ration: the fraction of sets actually read from
            the input file.
        skip: the number of lines from the beginning of the file
            to skip.

    Returns: (sets, keys)
    """
    sets = collections.deque([])
    keys = collections.deque([])
    random.seed(41)
    with open(sets_file, "rt") as f:
        for i, line in enumerate(f):
            if i < skip:
                # Skip lines
                continue
            if random.random() > sample_ratio:
                continue
            # Take tokens after the \t character splitting by ,.
            s = np.array([int(d) for d in line.strip().split("\t")[1].split(",")])
            sets.append(s)
            # Use the line number as the key.
            keys.append(i)
            sys.stdout.write("\r{} sets.".format(len(sets)))
        sys.stdout.write("\n")
    sets = list(sets)
    keys = list(keys)
    return (sets, keys)


def create_minhashes_from_sets(sets, num_perms, hashfunc):
    # Generate minhash
    minhashes = dict()
    for num_perm in num_perms:
        print(f"Generating {len(sets)} MinHashes with {num_perm} permutations...")
        ms = MinHash.bulk(
            ([str(w) for w in s] for s in sets), num_perm=num_perm, hashfunc=hashfunc
        )
        print("Done.")
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
    cursor.execute(
        """CREATE TABLE IF NOT EXISTS runs (
        key integer primary key,
        name text not null,
        time datetime not null,
        k integer,
        threshold real,
        params text not null,
        indexing_time real
    )"""
    )
    cursor.execute(
        """CREATE TABLE IF NOT EXISTS results (
        run_key integer not null,
        query_key integer not null,
        result text not null,
        time real not null,
        FOREIGN KEY(run_key) REFERENCES runs(key)
    )"""
    )
    cursor.execute(
        """CREATE INDEX IF NOT EXISTS run_key_idx 
            on results(run_key)"""
    )
    conn.commit()
    conn.close()


def save_results(
    run_name, k, threshold, params, indexing_time, results, times, output_sqlite
):
    conn = sqlite3.connect(output_sqlite)
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO runs 
            (name, time, k, threshold, params, indexing_time)
            VALUES (?, datetime('now'), ?, ?, ?, ?)""",
        (run_name, k, threshold, json.dumps(params), indexing_time),
    )
    cursor.execute("SELECT last_insert_rowid()")
    run_key = cursor.fetchone()[0]
    rows = (
        [run_key, result[0], json.dumps(result[1]), time]
        for result, time in zip(results, times)
    )
    cursor.executemany(
        """INSERT INTO results 
        (run_key, query_key, result, time)
        VALUES (?, ?, ?, ?)""",
        rows,
    )
    conn.commit()
    conn.close()


def load_results(run_key, conn):
    cursor = conn.cursor()
    cursor.execute(
        """SELECT query_key, result, time FROM results
            WHERE run_key = ?""",
        (run_key,),
    )
    results = []
    times = []
    for query_key, result, time in cursor:
        results.append((query_key, json.loads(result)))
        times.append((query_key, time))
    cursor.close()
    return (results, times)


def _compute_recall(result, ground, ignore_key=None):
    result_keys = [x[0] for x in result if x[0] != ignore_key]
    ground_keys = [x[0] for x in ground if x[0] != ignore_key]
    if len(ground_keys) == 0:
        return None
    intersection = len(np.intersect1d(result_keys, ground_keys))
    return float(intersection) / float(len(ground_keys))


def get_similarities_at_k(grounds):
    grounds.sort(key=lambda x: x[0])
    thresholds = []
    for query_key, ground in grounds:
        threshold = np.min([x[1] for x in ground])
        thresholds.append((query_key, threshold))
    return thresholds


def compute_recalls(results, grounds, ignore_query=False):
    results.sort(key=lambda x: x[0])
    grounds.sort(key=lambda x: x[0])
    recalls = []
    for (query_key_1, result), (query_key_2, ground) in zip(results, grounds):
        assert query_key_1 == query_key_2
        if ignore_query:
            ignore_key = query_key_1
        else:
            ignore_key = None
        recall = _compute_recall(result, ground, ignore_key=ignore_key)
        recalls.append((query_key_1, recall))
    return recalls


def compute_mean_similarities(results, ignore_query=False):
    similarities = []
    for query_key_1, result in results:
        sims = [x[1] for x in result if not ignore_query or x[0] != query_key_1]
        if len(sims) == 0:
            similarity = None
        else:
            similarity = np.mean(sims)
        similarities.append((query_key_1, similarity))
    return similarities


def get_run(name, k, threshold, params, result_sqlite):
    conn = sqlite3.connect(result_sqlite)
    cursor = conn.cursor()
    cursor.execute(
        """SELECT key, k, threshold, params, indexing_time 
            FROM runs WHERE name = ?""",
        (name,),
    )
    runs = [
        row[0]
        for row in cursor
        if row[1] == k and row[2] == threshold and json.loads(row[3]) == params
    ]
    conn.close()
    if len(runs) > 0:
        return runs[0]
    return None


def evaluate_runs(result_sqlite, names=None, ignore_query=False):
    conn = sqlite3.connect(result_sqlite)
    cursor = conn.cursor()
    if not names:
        cursor.execute(
            """SELECT key, name, k, threshold, params, indexing_time FROM runs"""
        )
    else:
        cursor.execute(
            """SELECT key, name, k, threshold, params, indexing_time
                FROM runs 
                WHERE name IN ? OR name == 'ground_truth""",
            (names,),
        )
    runs = [
        {
            "key": key,
            "name": name,
            "k": k,
            "threshold": threshold,
            **json.loads(params),
            "indexing_time": indexing_time,
        }
        for (key, name, k, threshold, params, indexing_time) in cursor
    ]
    cursor.close()

    # TODO: add filter to avoid hard-code.
    runs = [
        run
        for run in runs
        if run["name"] != "lsh" or (run["name"] == "lsh" and run["index"]["r"] > 1)
    ]

    # Get ground truth results first.
    for run in [run for run in runs if run["name"] == "ground_truth"]:
        results, times = load_results(run["key"], conn)
        run.update({"results": results, "times": times})

    # Compute mean recall and query time of every run.
    for run in [run for run in runs if run["name"] != "ground_truth"]:
        # Load results of this run.
        results, times = load_results(run["key"], conn)
        # Find the corresponding ground truth run with the same
        # benchmark settings.
        ground_truth = [
            x
            for x in runs
            if x["name"] == "ground_truth" and x["benchmark"] == run["benchmark"]
        ][0]
        # Compute metrics.
        similarities_at_k = get_similarities_at_k(ground_truth["results"])
        recalls = compute_recalls(
            results, ground_truth["results"], ignore_query=ignore_query
        )
        mean_similarities = compute_mean_similarities(
            results, ignore_query=ignore_query
        )
        # Update run with computed metrics.
        run.update(
            {
                "similarities_at_k": similarities_at_k,
                "recalls": recalls,
                "mean_similarities": mean_similarities,
                "times": times,
            }
        )

    conn.close()
    return runs
