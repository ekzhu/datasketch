import struct, copy
import numpy as np
try:
    from .hyperloglog_const import _thresholds, _raw_estimate, _bias
except ImportError:
    # For Python 2
    from hyperloglog_const import _thresholds, _raw_estimate, _bias

from datasketch.hashfunc import sha1_hash32, sha1_hash64

# Get the number of bits starting from the first non-zero bit to the right
_bit_length = lambda bits : bits.bit_length()
# For < Python 2.7
if not hasattr(int, 'bit_length'):
    _bit_length = lambda bits : len(bin(bits)) - 2 if bits > 0 else 0


class HyperLogLog(object):
    '''
    The HyperLogLog data sketch for estimating
    cardinality of very large dataset in a single pass.
    The original HyperLogLog is described `here
    <http://algo.inria.fr/flajolet/Publications/FlFuGaMe07.pdf>`_.

    This HyperLogLog implementation is based on:
    https://github.com/svpcom/hyperloglog

    Args:
        p (int, optional): The precision parameter. It is ignored if
            the `reg` is given.
        reg (numpy.array, optional): The internal state.
            This argument is for initializing the HyperLogLog from
            an existing one.
        hashfunc (optional): The hash function used by this MinHash.
            It takes the input passed to the `update` method and
            returns an integer that can be encoded with 32 bits.
            The default hash function is based on SHA1 from hashlib_.
        hashobj (**deprecated**): This argument is deprecated since version
            1.4.0. It is a no-op and has been replaced by `hashfunc`.
    '''

    __slots__ = ('p', 'm', 'reg', 'alpha', 'max_rank', 'hashfunc')

    # The range of the hash values used for HyperLogLog
    _hash_range_bit = 32
    _hash_range_byte = 4

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

    def __init__(self, p=8, reg=None, hashfunc=sha1_hash32, hashobj=None):
        if reg is None:
            self.p = p
            self.m = 1 << p
            self.reg = np.zeros((self.m,), dtype=np.int8)
        else:
            # Check if the register has the correct type
            if not isinstance(reg, np.ndarray):
                raise ValueError("The imported register must be a numpy.ndarray.")
            # We have to check if the imported register has the correct length.
            self.m = reg.size
            self.p = _bit_length(self.m) - 1
            if 1 << self.p != self.m:
                raise ValueError("The imported register has \
                    incorrect size. Expect a power of 2.")
            # Generally we trust the user to import register that contains
            # reasonable counter values, so we don't check for every values.
            self.reg = reg
        # Check the hash function.
        if not callable(hashfunc):
            raise ValueError("The hashfunc must be a callable.")
        # Check for use of hashobj and issue warning.
        if hashobj is not None:
            warnings.warn("hashobj is deprecated, use hashfunc instead.",
                    DeprecationWarning)
        self.hashfunc = hashfunc
        # Common settings
        self.alpha = self._get_alpha(self.p)
        self.max_rank = self._hash_range_bit - self.p

    def update(self, b):
        '''
        Update the HyperLogLog with a new data value in bytes.
        The value will be hashed using the hash function specified by
        the `hashfunc` argument in the constructor.

        Args:
            b: The value to be hashed using the hash function specified.

        Example:
            To update with a new string value (using the default SHA1 hash
            function, which requires bytes as input):

            .. code-block:: python

                hll = HyperLogLog()
                hll.update("new value".encode('utf-8'))

            We can also use a different hash function, for example, `pyfarmhash`:

            .. code-block:: python

                import farmhash
                def _hash_32(b):
                    return farmhash.hash32(b)
                hll = HyperLogLog(hashfunc=_hash_32)
                hll.update("new value")
        '''
        # Digest the hash object to get the hash value
        hv = self.hashfunc(b)
        # Get the index of the register using the first p bits of the hash
        reg_index = hv & (self.m - 1)
        # Get the rest of the hash
        bits = hv >> self.p
        # Update the register
        self.reg[reg_index] = max(self.reg[reg_index], self._get_rank(bits))

    def count(self):
        '''
        Estimate the cardinality of the data values seen so far.

        Returns:
            int: The estimated cardinality.
        '''
        # Use HyperLogLog estimation function
        e = self.alpha * float(self.m ** 2) / np.sum(2.0**(-self.reg))
        # Small range correction
        if e <= (5.0 / 2.0) * self.m:
            num_zero = self.m - np.count_nonzero(self.reg)
            return self._linearcounting(num_zero)
        # Normal range, no correction
        if e <= (1.0 / 30.0) * (1 << 32):
            return e
        # Large range correction
        return self._largerange_correction(e)

    def merge(self, other):
        '''
        Merge the other HyperLogLog with this one, making this the union of the
        two.

        Args:
            other (datasketch.HyperLogLog):
        '''
        if self.m != other.m or self.p != other.p:
            raise ValueError("Cannot merge HyperLogLog with different\
                    precisions.")
        self.reg = np.maximum(self.reg, other.reg)

    def digest(self, hashobj):
        '''
        Returns:
            numpy.array: The current internal state.
        '''
        return copy.copy(self.reg)

    def copy(self):
        '''
        Create a copy of the current HyperLogLog by exporting its state.

        Returns:
            datasketch.HyperLogLog:
        '''
        return HyperLogLog(reg=self.digest())

    def is_empty(self):
        '''
        Returns:
            bool: True if the current HyperLogLog is empty - at the state of just
            initialized.
        '''
        if np.any(self.reg):
            return False
        return True

    def clear(self):
        '''
        Reset the current HyperLogLog to empty.
        '''
        self.reg = np.zeros((self.m,), dtype=np.int8)

    def __len__(self):
        '''
        Returns:
            int: Get the size of the HyperLogLog as the size of
                `reg`.
        '''
        return len(self.reg)

    def __eq__(self, other):
        '''
        Check equivalence between two HyperLogLogs

        Args:
            other (datasketch.HyperLogLog):

        Returns:
            bool: True if both have the same internal state.
        '''
        return type(self) is type(other) and \
            self.p == other.p and \
            self.m == other.m and \
            np.array_equal(self.reg, other.reg)

    def _get_rank(self, bits):
        rank = self.max_rank - _bit_length(bits) + 1
        if rank <= 0:
            raise ValueError("Hash value overflow, maximum size is %d\
                    bits" % self.max_rank)
        return rank

    def _linearcounting(self, num_zero):
        return self.m * np.log(self.m / float(num_zero))

    def _largerange_correction(self, e):
        return - (1 << 32) * np.log(1.0 - e / (1 << 32))

    @classmethod
    def union(cls, *hyperloglogs):
        if len(hyperloglogs) < 2:
            raise ValueError("Cannot union less than 2 HyperLogLog\
                    sketches")
        m = hyperloglogs[0].m
        if not all(h.m == m for h in hyperloglogs):
            raise ValueError("Cannot union HyperLogLog sketches with\
                    different precisions")
        reg = np.maximum.reduce([h.reg for h in hyperloglogs])
        h = cls(reg=reg)
        return h

    def bytesize(self):
        # Since p is no larger than 64, use 8 bits
        p_size = struct.calcsize('B')
        # Each register value is no larger than 64, use 8 bits
        # TODO: is there a way to use 5 bits instead of 8 bits
        # to store integer in Python?
        reg_val_size = struct.calcsize('B')
        return p_size + reg_val_size * self.m

    def serialize(self, buf):
        if len(buf) < self.bytesize():
            raise ValueError("The buffer does not have enough space\
                    for holding this HyperLogLog.")
        fmt = 'B%dB' % self.m
        struct.pack_into(fmt, buf, 0, self.p, *self.reg)

    @classmethod
    def deserialize(cls, buf):
        size = struct.calcsize('B')
        try:
            p = struct.unpack_from('B', buf, 0)[0]
        except TypeError:
            p = struct.unpack_from('B', buffer(buf), 0)[0]
        h = cls(p)
        offset = size
        try:
            h.reg = np.array(struct.unpack_from('%dB' % h.m,
                buf, offset), dtype=np.int8)
        except TypeError:
            h.reg = np.array(struct.unpack_from('%dB' % h.m,
                buffer(buf), offset), dtype=np.int8)
        return h

    def __getstate__(self):
        buf = bytearray(self.bytesize())
        self.serialize(buf)
        return buf

    def __setstate__(self, buf):
        size = struct.calcsize('B')
        try:
            p = struct.unpack_from('B', buf, 0)[0]
        except TypeError:
            p = struct.unpack_from('B', buffer(buf), 0)[0]
        self.__init__(p=p)
        offset = size
        try:
            self.reg = np.array(struct.unpack_from('%dB' % self.m,
                buf, offset), dtype=np.int8)
        except TypeError:
            self.reg = np.array(struct.unpack_from('%dB' % self.m,
                buffer(buf), offset), dtype=np.int8)


class HyperLogLogPlusPlus(HyperLogLog):
    '''
    HyperLogLog++ is an enhanced HyperLogLog `from Google
    <http://research.google.com/pubs/pub40671.html>`_.
    Main changes from the original HyperLogLog:

    1. Use 64 bits instead of 32 bits for hash function
    2. A new small-cardinality estimation scheme
    3. Sparse representation (not implemented here)

    Args:
        p (int, optional): The precision parameter. It is ignored if
            the `reg` is given.
        reg (numpy.array, optional): The internal state.
            This argument is for initializing the HyperLogLog from
            an existing one.
        hashfunc (optional): The hash function used by this MinHash.
            It takes the input passed to the `update` method and
            returns an integer that can be encoded with 64 bits.
            The default hash function is based on SHA1 from hashlib_.
        hashobj (**deprecated**): This argument is deprecated since version
            1.4.0. It is a no-op and has been replaced by `hashfunc`.
    '''

    _hash_range_bit = 64
    _hash_range_byte = 8

    def __init__(self, p=8, reg=None, hashfunc=sha1_hash64,
            hashobj=None):
        super(HyperLogLogPlusPlus, self).__init__(p=p, reg=reg,
                hashfunc=hashfunc, hashobj=hashobj)

    def _get_threshold(self, p):
        return _thresholds[p - 4]

    def _estimate_bias(self, e, p):
        bias_vector = _bias[p - 4]
        estimate_vector = _raw_estimate[p - 4]
        nearest_neighbors = np.argsort((e - estimate_vector)**2)[:6]
        return np.mean(bias_vector[nearest_neighbors])

    def count(self):
        num_zero = self.m - np.count_nonzero(self.reg)
        if num_zero > 0:
            # linear counting
            lc = self._linearcounting(num_zero)
            if lc <= self._get_threshold(self.p):
                return lc
        # Use HyperLogLog estimation function
        e = self.alpha * float(self.m ** 2) / np.sum(2.0**(-self.reg))
        if e <= 5 * self.m:
            return e - self._estimate_bias(e, self.p)
        else:
            return e
