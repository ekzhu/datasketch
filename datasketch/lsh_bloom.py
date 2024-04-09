from __future__ import annotations
import pickle
import struct
from typing import Callable, Dict, Hashable, List, Optional, Tuple, Union
from datasketch.minhash import MinHash
from datasketch.b_bit_minhash import bBitMinHash
from datasketch.storage import ordered_storage, unordered_storage, _random_name
from scipy.integrate import quad as integrate
from bitarray import bitarray
import mmh3
import numpy as np
import math

_mersenne_prime = np.uint64((1 << 61) - 1)

def _false_positive_probability(threshold, b, r):
	_probability = lambda s: 1 - (1 - s ** float(r)) ** float(b)
	a, err = integrate(_probability, 0.0, threshold)
	return a


def _false_negative_probability(threshold, b, r):
	_probability = lambda s: 1 - (1 - (1 - s ** float(r)) ** float(b))
	a, err = integrate(_probability, threshold, 1.0)
	return a


def _optimal_param(threshold, num_perm, false_positive_weight, false_negative_weight):
	"""
	Compute the optimal `MinHashLSH` parameter that minimizes the weighted sum
	of probabilities of false positive and false negative.
	"""
	min_error = float("inf")
	opt = (0, 0)
	for b in range(1, num_perm + 1):
		max_r = int(num_perm / b)
		for r in range(1, max_r + 1):
			fp = _false_positive_probability(threshold, b, r)
			fn = _false_negative_probability(threshold, b, r)
			error = fp * false_positive_weight + fn * false_negative_weight
			if error < min_error:
				min_error = error
				opt = (b, r)
	return opt


# https://www.geeksforgeeks.org/bloom-filters-introduction-and-python-implementation/
class BloomFilter(object): 
	''' 
	Class for Bloom filter, using murmur3 hash function 
	'''
  
	def __init__(self, items_count, fp_prob, max_size=None): 
		''' 
		items_count : int 
			Number of items expected to be stored in bloom filter 
		fp_prob : float 
			False Positive probability in decimal 
		'''
		# False possible probability in decimal 
		self.fp_prob = fp_prob 
  
		# Size of bit array to use 
		self.size = self.get_size(items_count, fp_prob) 

		if max_size is not None and max_size < self.size:
			assert type(max_size) is int, "Max Size for Bloom Filter must be an integer"
			self.size = max_size
  
		# number of hash functions to use 
		self.hash_count = self.get_hash_count(self.size, items_count) 
  
		# Bit array of given size 
		self.bit_array = bitarray(self.size) 
  
		# initialize all bits as 0 
		self.bit_array.setall(0) 

	def add(self, item): 
		''' 
		Add an item in the filter 
		'''
		digests = [] 
		for i in range(self.hash_count): 
  
			# create digest for given item. 
			# i work as seed to mmh3.hash() function 
			# With different seed, digest created is different 
			digest = mmh3.hash(item, i) % self.size 
			digests.append(digest)
  
			# set the bit True in bit_array 
			self.bit_array[digest] = True

		# if LOG:
		# 	print(f"Hashing {item} to {digests}")
  
	def check(self, item): 
		''' 
		Check for existence of an item in filter 
		'''
		for i in range(self.hash_count): 
			digest = mmh3.hash(item, i) % self.size 
			if self.bit_array[digest] == False: 
  
				# if any of bit is False then,its not present 
				# in filter 
				# else there is probability that it exist 
				return False
		return True
  
	@classmethod
	def get_size(self, n, p): 
		''' 
		Return the size of bit array(m) to used using 
		following formula 
		m = -(n * lg(p)) / (lg(2)^2) 
		n : int 
			number of items expected to be stored in filter 
		p : float 
			False Positive probability in decimal 
		'''
		m = -(n * math.log(p))/(math.log(2)**2) 
		return int(m) 
  
	@classmethod
	def get_hash_count(self, m, n): 
		''' 
		Return the hash function(k) to be used using 
		following formula 
		k = (m/n) * lg(2) 
  
		m : int 
			size of bit array 
		n : int 
			number of items expected to be stored in filter 
		'''
		k = (m/n) * math.log(2) 
		return max(int(k), 1)

class BloomTable:
	"""
	Interface to a Bloom Filter meant to model a single band of the signature matrix
	"""
	def __init__(self, item_count: int, fp: float, num_arrays: int, max_size: int = None):
		self.r = num_arrays
		self.bloom_filter = BloomFilter(item_count, fp, max_size=max_size)


	def assert_size(self, hashvalues: List[int]):
		if not len(hashvalues) == self.r:
			raise RuntimeError(f"Invalid length for indices, {len(hashvalues)}, expected {self.r} items")
		
	def insert(self, hashvalues: List[int]) -> None:
		"""
		Takes as input the indices for a single band and inserts them into the corresponding bit arrays
		"""
		self.assert_size(hashvalues)
		# https://en.wikipedia.org/wiki/Universal_hashing#Hashing_vectors
		# as the hashvalues are the result of a universal hashing function, their sum is also a univeral hash function
		x = sum(hashvalues) % _mersenne_prime
		self.bloom_filter.add(x.tobytes())

	def query(self, hashvalues: List[int]) -> bool:
		"""
		Takes as input the indices for a single band and queries them against the corresponding arrays
		returns True if the each query returns True, otherwise returns False
		"""
		self.assert_size(hashvalues)
		x = sum(hashvalues) % _mersenne_prime
		return self.bloom_filter.check(x.tobytes())


class MinHashLSHBloom(object):
	"""
	The :ref:`minhash_lsh` index.
	It supports query with `Jaccard similarity`_ threshold.
	Reference: `Chapter 3, Mining of Massive Datasets
	<http://www.mmds.org/>`_.

	Args:
		threshold (float): The Jaccard similarity threshold between 0.0 and
			1.0. The initialized MinHash LSH will be optimized for the threshold by
			minizing the false positive and false negative.
		num_perm (int): The number of permutation functions used
			by the MinHash to be indexed. For weighted MinHash, this
			is the sample size (`sample_size`).
		weights (Tuple[float, float]): Used to adjust the relative importance of
			minimizing false positive and false negative when optimizing
			for the Jaccard similarity threshold.
			`weights` is a tuple in the format of
			:code:`(false_positive_weight, false_negative_weight)`.
		params (Optiona[Tuple[int, int]]): The LSH parameters (i.e., number of bands and size
			of each bands). This is used to bypass the parameter optimization
			step in the constructor. `threshold` and `weights` will be ignored
			if this is given.
		storage_config (Optional[Dict]): Type of storage service to use for storing
			hashtables and keys.
			`basename` is an optional property whose value will be used as the prefix to
			stored keys. If this is not set, a random string will be generated instead. If you
			set this, you will be responsible for ensuring there are no key collisions.
		prepickle (Optional[bool]): If True, all keys are pickled to bytes before
			insertion. If not specified, a default value is chosen based on the
			`storage_config`.
		hashfunc (Optional[Callable[[bytes], bytes]]): If a hash function is provided it will be used to
			compress the index keys to reduce the memory footprint. This could cause a higher
			false positive rate.

	Note:
		`weights` must sum to 1.0, and the format is
		(false positive weight, false negative weight).
		For example, if minimizing false negative (or maintaining high recall) is more
		important, assign more weight toward false negative: weights=(0.4, 0.6).
		Try to live with a small difference between weights (i.e. < 0.5).

	Examples:

		Create an index with 128 permutation functions optimized for Jaccard
		threshold 0.9:

		.. code-block:: python

			lsh = MinHashLSH(threshold=0.9, num_perm=128)
			print(lsh.b, lsh.r)
			# 5 25

		The built-in optimizer will try to minimize the weighted sum of
		probabilities of false positive and false negative. The algorithm is
		a simple grid search over the space of possible parameters.

		Note that it is possible to get :attr:`b` (number of bands) and
		:attr:`r` (band size) that do not sum to :attr:`num_perm`, leading to
		unused permutation values in the indexed MinHash.
		This is because the optimizer only considers bands of
		the same size, and the number of bands is not necessarily a divisor of
		:attr:`num_perm`.

		Instead of using the built-in optimizer, you can customize the LSH
		parameters your self. The snippet below creates an index with 128
		permutation functions and 16 bands each with size 8, skipping the
		optimization step:

		.. code-block:: python

			lsh = MinHashLSH(num_perm=128, params=(16, 8))
			print(lsh.b, lsh.r)
			# 16 8

		Create an index backed by Redis storage:

		.. code-block:: python

			lsh = MinHashLSH(threshold=0.9, num_perm=128, storage_config={
				'type': 'redis',
				'basename': b'mylsh', # optional, defaults to a random string.
				'redis': {'host': 'localhost', 'port': 6379},
			})

		The `basename` property is optional. It is used to generate key prefixes
		in the storage layer to uniquely identify data associated with this LSH.
		Thus, if you create a new LSH object with the same `basename`, you will
		be using the same underlying data in the storage layer associated with
		a previous LSH object. If you do not set this property, a random string
		will be generated instead.

	"""

	def __init__(
		self,
		threshold: float = 0.9,
		num_perm: int = 128,
		weights: Tuple[float, float] = (0.5, 0.5),
		num_bits: int = 32, # size in bits of each hashvalue, minhashes will be truncated to this size via bBitMinHashing
		n: int = None,
		fp: float = None,
		params: Optional[Tuple[int, int]] = None,
		hashfunc: Optional[Callable[[bytes], bytes]] = None,
	) -> None:
		self._buffer_size = 50000
		self.num_bits = num_bits
		if threshold > 1.0 or threshold < 0.0:
			raise ValueError("threshold must be in [0.0, 1.0]")
		if num_perm < 2:
			raise ValueError("Too few permutation functions")
		if any(w < 0.0 or w > 1.0 for w in weights):
			raise ValueError("Weight must be in [0.0, 1.0]")
		if sum(weights) != 1.0:
			raise ValueError("Weights must sum to 1.0")
		self.h = num_perm
		if params is not None:
			self.b, self.r = params
			if self.b * self.r > num_perm:
				raise ValueError(
					"The product of b and r in params is "
					"{} * {} = {} -- it must be less than num_perm {}. "
					"Did you forget to specify num_perm?".format(
						self.b, self.r, self.b * self.r, num_perm
					)
				)
		else:
			false_positive_weight, false_negative_weight = weights
			self.b, self.r = _optimal_param(
				threshold, num_perm, false_positive_weight, false_negative_weight
			)
		if self.b < 2:
			raise ValueError("The number of bands are too small (b < 2)")

		self.hashfunc = hashfunc
		if hashfunc:
			self._H = self._hashed_byteswap
		else:
			self._H = self._byteswap

		# create a bitarray for each signature row
		hashrange = 2**self.num_bits
		max_size = self.r * hashrange
		self.hashtables = [
			BloomTable(item_count=n, fp=fp, num_arrays=self.r, max_size=max_size)
			for i in range(self.b)
		]
		self.hashranges = [(i * self.r, (i + 1) * self.r) for i in range(self.b)]

	def insert(
		self,
		minhash: MinHash,
		check_duplication: bool = True,
	):
		"""
		Insert a key to the index, together with a MinHash or Weighted MinHash
		of the set referenced by the key.

		Args:
			key (Hashable): The unique identifier of the set.
			minhash (Union[MinHash, WeightedMinHash]): The MinHash of the set.
			check_duplication (bool): To avoid duplicate keys in the storage
				(`default=True`). It's recommended to not change the default, but
				if you want to avoid the overhead during insert you can set
				`check_duplication = False`.

		"""
		self._insert(minhash)

	def _insert(
		self,
		minhash: MinHash
	):
		if len(minhash) != self.h:
			raise ValueError(
				"Expecting minhash with length %d, got %d" % (self.h, len(minhash))
			)
		
		# truncate minhash to desired bit width
		trunc_minhash = minhash
		if self.num_bits <= 32:
			trunc_minhash = bBitMinHash(minhash=minhash, b=self.num_bits)
		Hs = [trunc_minhash.hashvalues[start:end] for start, end in self.hashranges]

		for H, hashtable in zip(Hs, self.hashtables):
			hashtable.insert(H)

	def query(self, minhash) -> bool:
		"""
		Giving the MinHash of the query set, retrieve
		the keys that reference sets with Jaccard
		similarities likely greater than the threshold.

		Results are based on minhash segment collision
		and are thus approximate. For more accurate results,
		filter again with :meth:`MinHash.jaccard`. For exact results,
		filter by computing Jaccard similarity using original sets.

		Args:
			minhash (MinHash): The MinHash of the query set.

		Returns:
			list: a list of unique keys.

		Example:

			Query and rank results using :meth:`MinHash.jaccard`.

			.. code-block:: python

				from datasketch import MinHash, MinHashLSH
				import numpy as np

				# Generate 100 random MinHashes.
				minhashes = MinHash.bulk(
					np.random.randint(low=0, high=30, size=(100, 10)),
					num_perm=128
				)

				# Create LSH index.
				lsh = MinHashLSH(threshold=0.5, num_perm=128)
				for i, m in enumerate(minhashes):
					lsh.insert(i, m)

				# Get the initial results from LSH.
				query = minhashes[0]
				results = lsh.query(query)

				# Rank results using Jaccard similarity estimated by MinHash.
				results = [(query.jaccard(minhashes[key]), key) for key in results]
				results.sort(reverse=True)
				print(results)

			Output:

			.. code-block::

				[(1.0, 0), (0.421875, 4), (0.4140625, 19), (0.359375, 58), (0.3359375, 78), (0.265625, 62), (0.2578125, 11), (0.25, 98), (0.171875, 21)]

			Note that although the threshold is set to 0.5, the results are not
			guaranteed to be above 0.5 because the LSH index is approximate and
			the Jaccard similarity is estimated by MinHash.

		"""
		if len(minhash) != self.h:
			raise ValueError(
				"Expecting minhash with length %d, got %d" % (self.h, len(minhash))
			)
		# if we match in any band, this is a candidate pair
		trunc_minhash = minhash
		if self.num_bits <= 32:
			trunc_minhash = bBitMinHash(minhash=minhash, b=self.num_bits)
		for (start, end), hashtable in zip(self.hashranges, self.hashtables):
			H = trunc_minhash.hashvalues[start:end]
			collision = hashtable.query(H)
			if collision:
				return True
		return False


	def _hashed_byteswap(self, hs):
		return self.hashfunc(bytes(hs.byteswap().data))

	def _byteswap(self, hs):
		return bytes(hs.byteswap().data)
