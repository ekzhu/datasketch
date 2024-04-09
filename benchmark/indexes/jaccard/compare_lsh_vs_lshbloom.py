from datasketch.lsh import MinHashLSH
from datasketch.lsh_bloom import MinHashLSHBloom
from datasketch.minhash import MinHash
from datasketch.b_bit_minhash import bBitMinHash
from datasets import load_dataset
import csv
import pandas as pd
from matplotlib import pyplot as plt
import pickle

from tqdm import trange

DATASET = load_dataset("wikipedia", "20220301.simple")
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
	# perform for MinHashLSH
	data_len = len(DATASET['train'])
	split_point = int(data_len*INSERT_SPLIT)
	query_len = int(data_len*QUERY_SPLIT)
	insert_set = DATASET['train'][:split_point]
	query_set = DATASET['train'][split_point:]

	lsh_csv = 'lsh.csv'
	test_bits = [1, 2, 4, 8, 16, 32]

	basename = b"test_bench"
	sim_threshold = 0.8
	n_hash_funcs = 128

	COMPUTE_LSH = True
	COMPUTE_LSH_BLOOM = True

	if COMPUTE_LSH:
		lsh = MinHashLSH(
			threshold=sim_threshold,
			num_perm=n_hash_funcs
		)

		# insert into LSH Index
		for i in trange(len(insert_set['text']), desc="Inserting into LSH Index"):
			key = insert_set['id'][i]
			m = compute_minhash(insert_set['text'][i], n_hash_funcs)
			if m is not None:
				lsh.insert(key, m)

		# query against index, log whether id is duplicated
		with open(lsh_csv, 'w') as csvfile:
			writer = csv.writer(csvfile)
			for i in trange(len(query_set['text']), desc="Querying LSH Index"):
				key = query_set['id'][i]
				m = compute_minhash(query_set['text'][i], n_hash_funcs)
				if m is None:
					continue
				
				result = lsh.query(m)
				is_duplicate = bool(len(result))
				writer.writerow([key, is_duplicate])


		del lsh


	# now do the same with lsh bloom
	if COMPUTE_LSH_BLOOM:
		for num_bits in test_bits:
			lsh = MinHashLSHBloom(
				threshold=sim_threshold, 
				num_perm=n_hash_funcs, 
				num_bits=num_bits,
				n=(len(insert_set['text'])+len(query_set['text'])),
				fp=0.01,
			)
	
			trunc_csv = f'lsh_bloom_{num_bits}_bit.csv'

			# insert
			for i in trange(len(insert_set['text']), desc=f"Inserting into {num_bits}-bit LSH Bloom Index"):
				key = insert_set['id'][i]
				m = compute_minhash(insert_set['text'][i], n_hash_funcs)
				if m is not None:
					lsh.insert(m)
				
			
			# query against index, log whether id is duplicated
			with open(trunc_csv, 'w') as csvfile:
				writer = csv.writer(csvfile)
				for i in trange(len(query_set['text']), desc=f"Querying {num_bits}-bit LSH Bloom Index"):
					key = query_set['id'][i]
					m = compute_minhash(query_set['text'][i], n_hash_funcs)
					if m is None:
						continue

					result = lsh.query(m)
					is_duplicate = result
					writer.writerow([key, is_duplicate])

		f = open('bloom64.pkl', 'wb')
		pickle.dump(lsh, f)
		f.close()

		for i, h in enumerate(lsh.hashtables):
			print(f"Band {i}")
			for j, arr in enumerate(h.arrays):
				b = arr.bit_array
				print(f"\t{j} - {(b.count(1)/len(b))*100} % full")
			print()
			

	

	# compare results
	x = test_bits
	agreement_pcts = [] # y
	
	col_names = ['key', 'is_duplicated']
	lsh_df = pd.read_csv(lsh_csv, header=None, index_col=False, names=col_names)

	for num_bits in test_bits:
		trunc_csv = f'lsh_bloom_{num_bits}_bit.csv'
		trunc_df = pd.read_csv(trunc_csv, header=None, index_col=False, names=col_names)

		merged_df = lsh_df.merge(trunc_df, on='key', suffixes=['_lsh', '_trunc'])
		merged_df['agree'] = merged_df['is_duplicated_lsh'] == merged_df['is_duplicated_trunc']
		merged_df['fp'] = (merged_df['is_duplicated_lsh'] == False) & (merged_df['is_duplicated_trunc'] == True)

		merged_df.to_csv('results_lshbloom.csv')

		pct_duplicated_lsh = merged_df['is_duplicated_lsh'].value_counts().get(True,0) / len(merged_df)
		pct_duplicated_trunc = merged_df['is_duplicated_trunc'].value_counts().get(True,0)  / len(merged_df)
		pct_agreement = merged_df['agree'].value_counts().get(True,0) / len(merged_df)
		fp_rate = merged_df['fp'].value_counts().get(True,0) / len(merged_df)

		print(f"{num_bits}-bit LSH Bloom:")
		print(pct_duplicated_lsh)
		print(pct_duplicated_trunc)
		print(pct_agreement)
		print(fp_rate)
		print()

		agreement_pcts.append(pct_agreement)

	plt.figure()
	plt.scatter(x, agreement_pcts)
	plt.ylim(0,1.05)
	plt.xticks(x)
	plt.title("Agreement between LSH and LSHBloom at various levels of precision")
	plt.xlabel("Bit Precision of Hashvalues (32 is full-precision)")
	plt.ylabel("% Agreement with full-precision LSH")
	plt.savefig("lsh_bloom_benchmark.png")

