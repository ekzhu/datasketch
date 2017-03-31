from datasketch import MinHashLSHEnsemble, MinHash

set1 = set(["cat", "dog", "fish", "cow"])
set2 = set(["cat", "dog", "fish", "cow", "pig", "elephant", "lion", "tiger",
             "wolf", "bird", "human"])
set3 = set(["cat", "dog", "car", "van", "train", "plane", "ship", "submarine",
             "rocket", "bike", "scooter", "motorcyle", "SUV", "jet", "horse"])

# Create MinHash objects
m1 = MinHash()
m2 = MinHash()
m3 = MinHash()
for d in set1:
    m1.update(d.encode('utf8'))
for d in set2:
    m2.update(d.encode('utf8'))
for d in set3:
    m3.update(d.encode('utf8'))

# Create an LSH Ensemble index with a threshold 
lshensemble = MinHashLSHEnsemble(threshold=0.8)

# Index takes an iterable of (key, minhash, size)
lshensemble.index([("m2", m2, len(set2)), ("m3", m3, len(set3))])

# Check for membership using the key
print("m2" in lshensemble)
print("m3" in lshensemble)

# Using m1 as the query, get an result iterator 
print("Sets with containment > 0.8:")
for key in lshensemble.query(m1, len(set1)):
    print(key)
