from datasketch import MinHashLSHForest, MinHash

data1 = ['minhash', 'is', 'a', 'probabilistic', 'data', 'structure', 'for',
        'estimating', 'the', 'similarity', 'between', 'datasets']
data2 = ['minhash', 'is', 'a', 'probability', 'data', 'structure', 'for',
        'estimating', 'the', 'similarity', 'between', 'documents']
data3 = ['minhash', 'is', 'probability', 'data', 'structure', 'for',
        'estimating', 'the', 'similarity', 'between', 'documents']

# Create MinHash objects
m1 = MinHash(num_perm=128)
m2 = MinHash(num_perm=128)
m3 = MinHash(num_perm=128)
for d in data1:
    m1.update(d.encode('utf8'))
for d in data2:
    m2.update(d.encode('utf8'))
for d in data3:
    m3.update(d.encode('utf8'))

forest = MinHashLSHForest(num_perm=128)

# Add m2 and m3 into the index
forest.add("m2", m2)
forest.add("m3", m3)

# IMPORTANT: must call index() otherwise the keys won't be searchable
forest.index()

# Check for membership using the key
print("m2" in forest)
print("m3" in forest)

# Using m1 as the query, retrieve top 2 keys that have the higest Jaccard
result = forest.query(m1, 2)
print("Top 2 candidates", result)
