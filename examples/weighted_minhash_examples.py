from datasketch import WeightedMinHashGenerator
import numpy as np

v1 = [1, 3, 4, 5, 6, 7, 8, 9, 10, 4]
v2 = [2, 4, 3, 8, 4, 7, 10, 9, 0, 0]

min_sum = np.sum(np.minimum(v1, v2))
max_sum = np.sum(np.maximum(v1, v2))
true_jaccard = float(min_sum) / float(max_sum)

wmg = WeightedMinHashGenerator(len(v1))
wm1 = wmg.minhash(v1)
wm2 = wmg.minhash(v2)
print("Estimated Jaccard is", wm1.jaccard(wm2))
print("True Jaccard is", true_jaccard)
