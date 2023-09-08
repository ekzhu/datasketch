import pprint
import sys
import gzip
import random
import collections
import sqlite3
import json
import time

import numpy as np
import farmhash

from datasketch import MinHash


def read_set_sizes_from_file(sets_file, skip=1):
    set_sizes = collections.deque([])
    with open(sets_file, "rt") as f:
        for i, line in enumerate(f):
            if i < skip:
                # Skip lines
                continue
            size = int(line.strip().split("\t")[0])
            set_sizes.append(size)
    return list(set_sizes)


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


def lazy_create_minhashes_from_sets(
    index_minhashes,
    index_sets,
    query_minhashes,
    query_sets,
    num_perm,
):
    index_minhashes_time = 0
    query_minhashes_time = 0
    if num_perm not in index_minhashes:
        start = time.perf_counter()
        index_minhashes.update(
            create_minhashes_from_sets(
                index_sets,
                [
                    num_perm,
                ],
                hashfunc=farmhash.hash32,
            )
        )
        index_minhashes_time = time.perf_counter() - start
        print(f"Index MinHashes took {index_minhashes_time} seconds.")
    if num_perm not in query_minhashes:
        start = time.perf_counter()
        query_minhashes.update(
            create_minhashes_from_sets(
                query_sets,
                [
                    num_perm,
                ],
                hashfunc=farmhash.hash32,
            )
        )
        query_minhashes_time = time.perf_counter() - start
        print(f"Query MinHashes took {query_minhashes_time} seconds.")
    return (index_minhashes_time, query_minhashes_time)


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


def compute_jaccard_distance(x, y):
    return 1.0 - compute_jaccard(x, y)


def compute_minhash_jaccard_distance(x, y):
    return 1.0 - x.jaccard(y)


def init_results_db(output_sqlite):
    conn = sqlite3.connect(output_sqlite)
    cursor = conn.cursor()
    cursor.execute(
        """CREATE TABLE IF NOT EXISTS runs (
        key integer primary key,
        name text not null,
        time datetime not null,
        config text not null,
        indexing text not null
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


def save_results(run_name, config, indexing, results, times, output_sqlite):
    conn = sqlite3.connect(output_sqlite)
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO runs 
            (name, time, config, indexing)
            VALUES (?, datetime('now'), ?, ?)""",
        (run_name, json.dumps(config), json.dumps(indexing)),
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


def compute_mean_distances(results, ignore_query=False):
    distances = []
    for query_key_1, result in results:
        ds = [1.0 - x[1] for x in result if not ignore_query or x[0] != query_key_1]
        if len(ds) == 0:
            distance = None
        else:
            distance = np.mean(ds)
        distances.append((query_key_1, distance))
    return distances


def get_run(name, config, result_sqlite):
    conn = sqlite3.connect(result_sqlite)
    cursor = conn.cursor()
    cursor.execute(
        """SELECT key, config
            FROM runs WHERE name = ?""",
        (name,),
    )
    runs = [row[0] for row in cursor if config == json.loads(row[1])]
    conn.close()
    if len(runs) > 0:
        return runs[0]
    return None


def evaluate_runs(result_sqlite, ignore_query=False):
    conn = sqlite3.connect(result_sqlite)
    cursor = conn.cursor()
    cursor.execute("""SELECT key, name, config, indexing FROM runs""")
    runs = [
        {
            "key": key,
            "name": name,
            **json.loads(config),
            **json.loads(indexing),
        }
        for (key, name, config, indexing) in cursor
    ]
    cursor.close()

    # TODO: add filter to avoid hard-code.
    runs = [
        run
        for run in runs
        if run["name"] != "lsh" or (run["name"] == "lsh" and run["index"]["r"] > 1)
    ]

    # Get ground truth results first.
    for run in [run for run in runs if run["name"] == "exact"]:
        results, times = load_results(run["key"], conn)
        run.update({"results": results, "times": times})

    # Compute mean recall and query time of every run.
    for run in [run for run in runs]:
        # Load results of this run.
        results, times = load_results(run["key"], conn)
        # Find the corresponding ground truth run with the same
        # query and benchmark settings.
        if run["name"] == "exact":
            exact = run
        else:
            exact = [
                x
                for x in runs
                if x["name"] == "exact"
                and x["benchmark"] == run["benchmark"]
                and x["query"] == run["query"]
            ][0]
        # Compute metrics.
        counts = [(query_key, len(result)) for query_key, result in results]
        similarities_at_k = get_similarities_at_k(exact["results"])
        distances_at_k = [
            (query_key, 1.0 - sim) for query_key, sim in similarities_at_k
        ]
        recalls = compute_recalls(results, exact["results"], ignore_query=ignore_query)
        mean_similarities = compute_mean_similarities(
            results, ignore_query=ignore_query
        )
        mean_distances = compute_mean_distances(results, ignore_query=ignore_query)
        # Update run with computed metrics.
        run.update(
            {
                "similarities_at_k": similarities_at_k,
                "distances_at_k": distances_at_k,
                "recalls": recalls,
                "mean_similarities": mean_similarities,
                "mean_distances": mean_distances,
                "times": times,
                "counts": counts,
            }
        )

    conn.close()
    return runs
