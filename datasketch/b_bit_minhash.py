'''
This module implements the b-bit MinHash.
http://research.microsoft.com/pubs/120078/wfc0398-liPS.pdf

b-bit MinHash reduces storage space by storing only the b lowest bits
of each minimum hashed value, without significant loss of accuracy.
'''

import struct
import numpy as np

class bBitMinHash(object):
    '''
    The b-bit MinHash object
    '''

    __slots__ = ('seed', 'b', 'r', 'hashvalues')

    # seed as int64
    # b as uint8
    # r as float64
    # num_perm as int32
    _serial_fmt_params = '<qBdi'
    # each block as uint64
    _serial_fmt_block = 'Q'

    def __init__(self, minhash, b=1, r=0.0):
        '''
        Initialize a b-bit MinHash given an existing full MinHash
        object and parameter b - the number of bits to store for
        each minimum hashed values in the MinHash object.
        '''
        b = int(b)
        r = float(r)
        if b > 32 or b < 0:
            raise ValueError("b must be an integer in [0, 32]")
        if r > 1.0:
            raise ValueError("r must be a float in [0.0, 1.0]")
        bmask = (1 << b) - 1
        self.hashvalues = np.bitwise_and(minhash.hashvalues, bmask)\
                .astype(np.uint32)
        self.seed = minhash.seed
        self.b = b
        self.r = r

    def __eq__(self, other):
        '''
        Check for full equality of two b-bit MinHash objects.
        '''
        return type(self) is type(other) and \
            self.seed == other.seed and \
            self.b == other.b and \
            self.r == other.r and \
            np.array_equal(self.hashvalues, other.hashvalues)


    def jaccard(self, other):
        '''
        Estimate the Jaccard similarity (resemblance) between this b-bit
        MinHash and the other.
        '''
        if self.b != other.b:
            raise ValueError("Cannot compare two b-bit MinHashes with different\
                    b values")
        if self.seed != other.seed:
            raise ValueError("Cannot compare two b-bit MinHashes with different\
                    set of permutations")
        intersection = np.count_nonzero(self.hashvalues==other.hashvalues)
        raw_est = float(intersection) / float(self.hashvalues.size)
        a1 = self._calc_a(self.r, self.b)
        a2 = self._calc_a(other.r, other.b)
        c1, c2 = self._calc_c(a1, a2, self.r, other.r)
        return (raw_est - c1) / (1 - c2)

    def bytesize(self):
        '''
        Get the serialized size of this b-bit MinHash in number of bytes.
        '''
        return self._bytesize()[-1]

    def __getstate__(self):
        '''
        This function is called when pickling the b-bit MinHash object.
        Returns a bytearray which will then be pickled.
        '''
        slot_size, n, num_blocks, total = self._bytesize()
        buffer = bytearray(total)
        blocks = np.zeros((num_blocks,), dtype=np.uint64)
        for i in range(num_blocks):
            # Obtain the current segment of n hashed values
            start = i * n
            hvs = self.hashvalues[start:start+n]
            # Store the n b-bit hashed values in the current block
            for j, hv in enumerate(hvs):
                blocks[i] |= np.uint64(hv << (n - 1 - j) * slot_size)
        fmt = self._serial_fmt_params + \
                "%d%s" % (num_blocks, self._serial_fmt_block)
        struct.pack_into(fmt, buffer, 0, self.seed, self.b, self.r, \
                self.hashvalues.size, *blocks)
        return buffer

    def __setstate__(self, buf):
        '''
        This function is called when unpickling the b-bit MinHash object.
        Initialize the object with data in the buffer.
        '''
        try:
            self.seed, self.b, self.r, num_perm = \
                    struct.unpack_from(self._serial_fmt_params, buf, 0)
        except TypeError:
            self.seed, self.b, self.r, num_perm = \
                    struct.unpack_from(self._serial_fmt_params, buffer(buf), 0)
        offset = struct.calcsize(self._serial_fmt_params)
        self.hashvalues = np.zeros((num_perm,), dtype=np.uint32)
        # Reconstruct the hash values
        slot_size, n, num_blocks, total = self._bytesize()
        fmt = "%d%s" % (num_blocks, self._serial_fmt_block)
        try:
            blocks = struct.unpack_from(fmt, buf, offset)
        except TypeError:
            blocks = struct.unpack_from(fmt, buffer(buf), offset)
        mask = (1 << slot_size) - 1
        for i in range(num_blocks):
            start = i * n
            for j, _ in enumerate(self.hashvalues[start:start+n]):
                hv = (blocks[i] >> (n - 1 - j) * slot_size) & mask
                self.hashvalues[start+j] = np.uint32(hv)

    def _calc_a(self, r, b):
        '''
        Compute the function A(r, b)
        '''
        if r == 0.0:
            # Find the limit of A(r, b) as r -> 0.
            return 1.0 / (1 << b)
        return r * (1 - r) ** (2 ** b - 1) / (1 - (1 - r) ** (2 * b))

    def _calc_c(self, a1, a2, r1, r2):
        '''
        Compute the functions C1 and C2
        '''
        if r1 == 0.0 and r2 == 0.0:
            # Find the limits of C1 and C2 as r1 -> 0 and r2 -> 0
            # Since the b-value must be the same and r1 = r2,
            # we have A1(r1, b1) = A2(r2, b2) = A,
            # then the limits for both C1 and C2 are A.
            return a1, a2
        div = 1 / (r1 + r2)
        c1 = (a1 * r2 + a2 * r1) * div
        c2 = (a1 * r1 + a2 * r2) * div
        return c1, c2

    def _find_slot_size(self, b):
        if b == 1:
            return 1
        if b == 2:
            return 2
        if b <= 4:
            return 4
        if b <= 8:
            return 8
        if b <= 16:
            return 16
        if b <= 32:
            return 32
        raise ValueError("Incorrect value of b")

    def _bytesize(self):
        block_size = struct.calcsize(self._serial_fmt_block)
        # Get the size of the slot for storing one b-bit hashed value
        slot_size = self._find_slot_size(self.b)
        # Get the number of slots to be stored in each block
        num_slots_per_block = int(block_size * 8 / slot_size)
        # Get the number of blocks required
        num_blocks = int(np.ceil(float(self.hashvalues.size) /\
                num_slots_per_block))
        # Get the total serialized size
        total = struct.calcsize(self._serial_fmt_params + \
                "%d%s" % (num_blocks, self._serial_fmt_block))
        return slot_size, num_slots_per_block, num_blocks, total
