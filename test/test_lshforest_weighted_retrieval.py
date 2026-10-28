import pickle

import numpy as np
import pytest

from datasketch import (
    LeanMinHash,
    MinHash,
    MinHashLSHForest,
    WeightedMinHash,
    WeightedMinHashGenerator,
    WeightedMinHashLSHForest,
)


@pytest.mark.parametrize("l", [1, 4, 16])
@pytest.mark.parametrize("roundtrip", [False, True])
@pytest.mark.parametrize("method", ["minhash", "minhash_many"])
def test_retrieve_weighted_hashvalues(l, roundtrip, method):
    generator = WeightedMinHashGenerator(3, sample_size=16, seed=7)
    vectors = np.array([[1, 2, 3], [0.00001, 0.00002, 0.00003]], dtype=np.float32)
    if method == "minhash_many":
        sketches = generator.minhash_many(vectors)
    else:
        sketches = [generator.minhash(vector) for vector in vectors]
    forest = WeightedMinHashLSHForest(num_perm=16, l=l)
    for key, sketch in enumerate(sketches):
        forest.add(key, sketch)
    forest.index()
    if roundtrip:
        forest = pickle.loads(pickle.dumps(forest))

    for key, sketch in enumerate(sketches):
        assert key in forest.query(sketch, len(sketches))
        values = forest.get_minhash_hashvalues(key)
        assert values.shape == sketch.hashvalues.shape
        assert values.dtype == sketch.hashvalues.dtype
        np.testing.assert_array_equal(values, sketch.hashvalues)
        restored = WeightedMinHash(seed=sketch.seed, hashvalues=values)
        assert sketch.jaccard(restored) == 1.0
        assert sketches[0].jaccard(restored) == sketches[0].jaccard(sketch)
        values[:] = 0
        np.testing.assert_array_equal(forest.get_minhash_hashvalues(key), sketch.hashvalues)


@pytest.mark.parametrize("scheme", ["legacy", "affine32", "affine64"])
@pytest.mark.parametrize("lean", [False, True])
def test_retrieve_unweighted_hashvalues(scheme, lean):
    sketch = MinHash(num_perm=16, scheme=scheme)
    sketch.update_batch([b"alpha", b"beta", b"gamma"])
    if lean:
        sketch = LeanMinHash(sketch)
    forest = MinHashLSHForest(num_perm=16, l=4)
    forest.add("document", sketch)
    values = forest.get_minhash_hashvalues("document")
    assert values.shape == sketch.hashvalues.shape
    assert values.dtype == sketch.hashvalues.dtype
    np.testing.assert_array_equal(values, sketch.hashvalues)


def test_retrieve_weighted_hashvalues_missing_key():
    forest = WeightedMinHashLSHForest(num_perm=16, l=4)
    with pytest.raises(KeyError):
        forest.get_minhash_hashvalues("missing")
