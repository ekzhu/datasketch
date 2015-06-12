'''
This module implements the HyperLogLog data sketch for estimating
cardinality of very large dataset in a single pass.

The original HyperLogLog is described here:
http://algo.inria.fr/flajolet/Publications/FlFuGaMe07.pdf

This HyperLogLog implementation is based on:
https://github.com/svpcom/hyperloglog
with enhanced functionalities for serialization and similarities.
'''

import struct, math
try:
    from .hyperloglog_const import _thresholds, _raw_estimate, _bias
except ImportError:
    # For Python 2
    from hyperloglog_const import _thresholds, _raw_estimate, _bias


# Get the number of bits starting from the first non-zero bit to the right
_bit_length = lambda bits : bits.bit_length()
# For < Python 2.7
if not hasattr(int, 'bit_length'):
    _bit_length = lambda bits : len(bin(bits)) - 2 if bits > 0 else 0


class HyperLogLog(object):
    '''
    The HyperLogLog class.
    '''

    __slots__ = ('p', 'm', 'reg', 'alpha', 'max_rank')

    # The range of the hash values used for HyperLogLog
    _hash_range_bit = 32
    _hash_range_byte = 4
    _struct_fmt_str = '<I'

    def _get_alpha(self, p):
        if not (4 <= p <= 16):
            raise ValueError("p=%d should be in range [4 : 16]" % p)
        if p == 4:
            return 0.673
        if p == 5:
            return 0.697
        if p == 6:
            return 0.709
        return 0.7213 / (1.0 + 1.079 / (1 << p))

    def __init__(self, p=8, reg=None):
        '''
        Create a HyperLogLog with precision parameter `p` and (optionally) a
        register vector `reg`. If `reg` is specified, the constructor will
        use it as the underlying regiser, instead of creating a new one, and
        the `p` parameter value is ignored.
        '''
        if reg is None:
            self.p = p
            self.m = 1 << p
            self.reg = [0 for _ in range(self.m)]
        else:
            # We have to check if the imported register has the correct length.
            self.m = len(reg)
            self.p = _bit_length(self.m) - 1
            if 1 << self.p != self.m:
                raise ValueError("The imported register has \
                    incorrect size. Expect a power of 2.")
            # Generally we trust the user to import register that contains
            # reasonable counter values, so we don't check for every values.
            self.reg = reg
        # Common settings
        self.alpha = self._get_alpha(self.p)
        self.max_rank = self._hash_range_bit - self.p

    def is_empty(self):
        '''
        Check if the current HyperLogLog is empty - at the state of just
        initialized.
        '''
        if any(v != 0 for v in self.reg):
            return False
        return True

    def _get_rank(self, bits):
        rank = self.max_rank - _bit_length(bits) + 1
        if rank <= 0:
            raise ValueError("Hash value overflow, maximum size is %d\
                    bits" % self.max_rank)
        return rank

    def digest(self, hashobj):
        '''
        Digest a hash object that implemented `digest` as in hashlib.
        The `digest` function of the hashobj must return at least same
        number of bytes as the hash range (in bytes).
        '''
        # Digest the hash object to get the hash value
        hv = struct.unpack(self._struct_fmt_str, 
                hashobj.digest()[:self._hash_range_byte])[0]
        # Get the index of the register using the first p bits of the hash
        reg_index = hv & (self.m - 1)
        # Get the rest of the hash
        bits = hv >> self.p
        # Update the register
        self.reg[reg_index] = max(self.reg[reg_index], self._get_rank(bits))

    def merge(self, other):
        '''
        Merge the other HyperLogLog with this one, making this the union of the
        two.
        '''
        if self.m != other.m or self.p != other.p:
            raise ValueError("Cannot merge HyperLogLog with different\
                    precisions.")
        self.reg = [max(*vs) for vs in zip(self.reg, other.reg)]

    def _linearcounting(self, num_zero):
        return self.m * math.log(self.m / float(num_zero))

    def _largerange_correction(self, e):
        return - (1 << 32) * math.log((1.0 - e / (1 << 32)), 2)

    def count(self):
        '''
        Estimate the cardinality of the data seen so far.
        '''
        # Use HyperLogLog estimation function
        e = self.alpha * float(self.m ** 2) / sum(1.0/(1 << int(v)) for v in self.reg)
        # Small range correction
        if e <= (5.0 / 2.0) * self.m:
            num_zero = sum(1 for v in self.reg if v == 0)
            return self._linearcounting(num_zero)
        # Normal range, no correction
        if e <= (1.0 / 30.0) * (1 << 32):
            return e
        # Large range correction
        return self._largerange_correction(e)

    def union_count(self, other):
        '''
        Estimate the cardinality of the union of this and the other HyperLogLogs.
        '''
        if self.p != other.p:
            raise ValueError("Cannot union HyperLogLogs with different\
                    precisions.")

        # Use HyperLogLog estimation function
        e = self.alpha * float(self.m ** 2) / sum(1.0/(1<<int(max(v1, v2)))
                for v1, v2 in zip(self.reg, other.reg))
        # Small range correction
        if e <= (5.0 / 2.0) * self.m:
            num_zero = sum(1 for v1, v2 in zip(self.reg, other.reg)
                    if v1 == 0 and v2 == 0)
            return self._linearcounting(num_zero)
        # Normal range, no correction
        if e <= (1.0 / 30.0) * (1 << 32):
            return e
        # Large range correction
        return self._largerange_correction(e)

    def intersection_count(self, other):
        '''
        Estimate the cardinality of the intersection of this and the other
        HyperLogLogs. The value may be negative due to estimation error.
        '''
        uc = self.union_count(other)
        return self.count() + other.count() - uc

    def jaccard(self, other):
        '''
        Estimate the Jaccard similarity between the multiset counted by this
        HyperLogLog and the multiset counted by the other HyperLogLog.
        The value may be negative due to estimation error.
        '''
        uc = self.union_count(other)
        if uc == 0.0:
            return 1.0
        ic = self.count() + other.count() - uc
        return ic / uc

    def inclusion(self, other):
        '''
        Estimate the inclusion of this HyperLogLog against the other.
        It measures the fraction of the multiset counted by this HyperLogLog
        overlapping with the multiset counted by the other HyperLogLog.
        The value may be negative due to estimation error.
        '''
        # Use inclusion-exclusion principle to compute the intersection size
        c = self.count()
        if c == 0.0:
            return 1.0
        uc = self.union_count(other)
        ic = c + other.count() - uc
        return ic / c

    def bytesize(self):
        '''
        Return the size of the HyperLogLog in bytes.
        '''
        # Since p is no larger than 64, use 8 bits
        p_size = struct.calcsize('B')
        # Each register value is no larger than 64, use 8 bits
        # TODO: is there a way to use 5 bits instead of 8 bits
        # to store integer in Python?
        reg_val_size = struct.calcsize('B')
        return p_size + reg_val_size * self.m

    def serialize(self, buffer):
        '''
        Serialize this HyperLogLog into bytes, store in the `buffer`.
        This is more efficient than using pickle.dumps on the object.
        '''
        if len(buffer) < self.bytesize():
            raise ValueError("The buffer does not have enough space\
                    for holding this HyperLogLog.")
        fmt = 'B%dB' % self.m
        struct.pack_into(fmt, buffer, 0, self.p, *self.reg)

    @classmethod
    def deserialize(cls, buffer):
        '''
        Reconstruct a HyperLogLog from bytes in `buffer`.
        This is more efficient than using the pickle.loads on the pickled
        bytes.
        '''
        size = struct.calcsize('B')
        p = struct.unpack_from('B', buffer, 0)[0]
        h = cls(p)
        offset = size
        for i in range(h.m):
            h.reg[i] = struct.unpack_from('B', buffer, offset)[0]
            offset += size
        return h

    def __getstate__(self):
        '''
        This function is called when pickling the HyperLogLog object.
        Returns a bytearray which will then be pickled.
        Note that the bytes returned by the Python pickle.dumps is not
        the same as the buffer returned by this function.
        '''
        buffer = bytearray(self.bytesize())
        self.serialize(buffer)
        return buffer

    def __setstate__(self, buffer):
        '''
        This function is called when unpickling the HyperLogLog object.
        Initialize the object with data in the buffer.
        Note that the input buffer is not the same as the input to the
        Python pickle.loads function.
        '''
        size = struct.calcsize('B')
        p = struct.unpack_from('B', buffer, 0)[0]
        self.__init__(p=p)
        offset = size
        for i in range(self.m):
            self.reg[i] = struct.unpack_from('B', buffer, offset)[0]
            offset += size
    
    @classmethod
    def union(cls, *hyperloglogs):
        '''
        Return the union of all given HyperLogLogs
        '''
        if len(hyperloglogs) < 2:
            raise ValueError("Cannot union less than 2 HyperLogLog\
                    sketches")
        m = hyperloglogs[0].m
        if not all(h.m == m for h in hyperloglogs):
            raise ValueError("Cannot union HyperLogLog sketches with\
                    different precisions")
        reg = [max(*vs) for vs in zip(*[h.reg for h in hyperloglogs])]
        h = cls(reg=reg)
        return h

    def __eq__(self, other):
        '''
        Check equivalence between two HyperLogLogs
        '''
        if self.p != other.p:
            return False
        if self.m != other.m:
            return False
        if any(v1 != v2 for v1, v2 in zip(self.reg, other.reg)):
            return False
        return True


class HyperLogLogPlusPlus(HyperLogLog):
    '''
    The HyperLogLog++, an enhanced HyperLogLog from Google.
    http://research.google.com/pubs/pub40671.html
    Main changes:
    1) Use 64 bits instead of 32 bits for hash function
    2) A new small-cardinality estimation scheme
    3) Sparse representation (not implemented here)
    '''
    
    _hash_range_bit = 64
    _hash_range_byte = 8
    _struct_fmt_str = '<Q'

    def _get_threshold(self, p):
        return _thresholds[p - 4]

    def _get_nearest_neighbors(self, e, estimate_vector):
        distance_map = [((e - float(v)) ** 2, i) for i, v in enumerate(estimate_vector)]
        distance_map.sort()
        return [idx for dist, idx in distance_map[:6]]

    def _estimate_bias(self, e, p):
        bias_vector = _bias[p - 4]
        nearest_neighbors = self._get_nearest_neighbors(e, _raw_estimate[p - 4])
        return sum([float(bias_vector[i]) for i in nearest_neighbors]) /\
                len(nearest_neighbors)

    def count(self):
        num_zero = sum(1 for v in self.reg if v == 0)
        if num_zero > 0:
            # linear counting
            lc = self.m * math.log(self.m / float(num_zero))
            if lc <= self._get_threshold(self.p):
                return lc
        # Use HyperLogLog estimation function
        e = self.alpha * float(self.m ** 2) / sum(1.0/(1<<int(v)) for v in self.reg)
        if e <= 5 * self.m:
            return e - self._estimate_bias(e, self.p)
        else:
            return e

    def union_count(self, other):
        if self.p != other.p:
            raise ValueError("Cannot union HyperLogLogs with different\
                    precisions.")
        num_zero = sum(1 for v1, v2 in zip(self.reg, other.reg)
                if v1 == 0 and v2 == 0)
        if num_zero > 0:
            # linear counting
            lc = self.m * math.log(self.m / float(num_zero))
            if lc <= self._get_threshold(self.p):
                return lc
        # Use HyperLogLog estimation function
        e = self.alpha * float(self.m ** 2) / sum(1.0/(1<<int(max(v1, v2)))
                for v1, v2 in zip(self.reg, other.reg))
        if e <= 5 * self.m:
            return e - self._estimate_bias(e, self.p)
        else:
            return e
