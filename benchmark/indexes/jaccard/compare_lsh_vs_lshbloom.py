from datasketch.lsh import MinHashLSH
from datasketch.lsh_bloom import MinHashLSHBloom
from datasketch.minhash import MinHash
from datasets import load_dataset
import csv
import pandas as pd
from matplotlib import pyplot as plt
import pickle
import numpy as np
from tqdm import trange

DATASET = load_dataset("wikipedia", "20220301.simple", trust_remote_code=True)
INSERT_SPLIT = 0.6
QUERY_SPLIT = 1.0 - INSERT_SPLIT

def compute_minhash(text: str, num_perm: int) -> MinHash:
	s = set(text.split())
	if not s:
		return None
	m = MinHash(num_perm=num_perm)
	for d in s:
		m.update(d.encode("utf8"))
	return m

if __name__ == '__main__':
	data_len = len(DATASET['train'])
	split_point = int(data_len*INSERT_SPLIT)
	query_len = int(data_len*QUERY_SPLIT)
	insert_set = DATASET['train'][:split_point]
	query_set = DATASET['train'][split_point:]

	lsh_csv = 'lsh.csv'
	sim_threshold = 0.8
	n_hash_funcs = 128
	fps = [0.7, 0.5, 0.25, 0.1, 0.01, 0.001, 0.0001, 0.00001]

	# Compute Index for MinHashLSH
	lsh = MinHashLSH(
		threshold=sim_threshold,
		num_perm=n_hash_funcs
	)

	# insert into LSH Index
	minhashes = []
	for item in insert_set['text']:
		minhashes.append(compute_minhash(item, n_hash_funcs))

	query_minhashes = []
	for item in query_set['text']:
		query_minhashes.append(compute_minhash(item, n_hash_funcs))

	for i in trange(len(minhashes), desc="Inserting into LSH Index"):
		key = insert_set['id'][i]
		m = minhashes[i]
		if m is not None:
			lsh.insert(key, m)

	# query against index, log whether id is duplicated
	with open(lsh_csv, 'w') as csvfile:
		writer = csv.writer(csvfile)
		for i in trange(len(query_set['text']), desc="Querying LSH Index"):
			key = query_set['id'][i]
			m = query_minhashes[i]
			if m is None:
				continue
			
			result = lsh.query(m)
			is_duplicate = bool(len(result))
			writer.writerow([key, is_duplicate])

	del lsh

	insert_times = []
	query_times = []

	for fp in fps:
		# Compute Index for LSHBloom
		lsh = MinHashLSHBloom(
			threshold=sim_threshold, 
			num_perm=n_hash_funcs, 
			n=(len(insert_set['text'])+len(query_set['text'])),
			fp=fp
		)

		bloom_csv = f'lsh_bloom_{fp}.csv'

		# insert
		for i in trange(len(minhashes), desc=f"Inserting into LSHBloom Index (fp={fp})"):
			key = insert_set['id'][i]
			m = minhashes[i]
			if m is not None:
				lsh.insert(m)
		
		# query against index, log whether id is duplicated
		with open(bloom_csv, 'w') as csvfile:
			writer = csv.writer(csvfile)
			for i in trange(len(query_set['text']), desc=f"Querying LSHBloom Index"):
				key = query_set['id'][i]
				m = query_minhashes[i]
				if m is None:
					continue

				is_duplicate = lsh.query(m)
				writer.writerow([key, is_duplicate])
				
	del lsh
	

	# compare results
	col_names = ['key', 'is_duplicated']
	lsh_df = pd.read_csv(lsh_csv, header=None, index_col=False, names=col_names)
	agreement_pcts = []

	for fp in fps:
		bloom_csv = f'lsh_bloom_{fp}.csv'
		bloom_df = pd.read_csv(bloom_csv, header=None, index_col=False, names=col_names)

		merged_df = lsh_df.merge(bloom_df, on='key', suffixes=['_lsh', '_bloom'])
		merged_df['agree'] = merged_df['is_duplicated_lsh'] == merged_df['is_duplicated_bloom']

		pct_agreement = merged_df['agree'].value_counts().get(True,0) / len(merged_df)
		agreement_pcts.append(pct_agreement)

	fig, ax = plt.subplots()
	plt.plot(fps, agreement_pcts, marker='o')
	plt.ylim(0,1.05)
	plt.xticks(fps)
	plt.xscale('log')
	plt.title("Agreement between LSH and LSHBloom at various values of p")
	plt.xlabel("Bloom filter false positive rate (p)")
	plt.ylabel("% Agreement with MinHashLSH")

	plt.tight_layout()
	plt.savefig("plots/lsh_bloom_benchmark.png")

