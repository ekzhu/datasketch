'''
This module implements the b-bit MinHash.
http://research.microsoft.com/pubs/120078/wfc0398-liPS.pdf

b-bit MinHash reduces storage space by storing only the b lowest bits 
of each minimum hashed value, without significant loss of accuracy.
'''

import struct 
import numpy as np
from datasketch.minhash import MinHash
from datasketch.hyperloglog import _bit_length


class bBitMinHash(object):
    '''
    The b-bit MinHash object
    '''

    __slots__ = ('seed', 'b', 'hashvalues')  

    def __init__(self, minhash, b=1):
        '''
        Initialize a b-bit MinHash given an existing full MinHash
        object and parameter b - the number of bits to store for
        each minimum hashed values in the MinHash object.
        '''
        b = int(b)
        if b > 32 or b < 0:
            raise ValueError("b must be an integer in [0, 32]")
        bmask = (1 << b) - 1
        self.hashvalues = np.bitwise_and(minhash.hashvalues, bmask)\
                .astype(np.uint32)
        self.b = b
        self.seed = minhash.seed

    def __eq__(self, other):
        '''
        Check for full equality of all b-bit hashed values.
        '''
        return self.seed == other.seed and self.b == other.b and \
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
        return 2.0 * (float(intersection) / float(self.hashvalues.size) - 0.5)

    def bytesize(self):
        '''
        Get the serialized size of this b-bit MinHash in number of bytes.
        '''
        return self._bytesize()[-1]

    def __getstate__(self):
        '''
        This function is called when pickling the b-bit MinHash object.
        Returns a bytearray which will then be pickled.
        Note that the bytes returned by the Python pickle.dumps is not
        the same as the buffer returned by this function.
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
        fmt = '<qiB%dQ' % num_blocks
        struct.pack_into(fmt, buffer, 0, self.seed, self.hashvalues.size,\
                self.b, *blocks)
        return buffer

    def __setstate__(self, buffer):
        '''
        This function is called when unpickling the b-bit MinHash object.
        Initialize the object with data in the buffer.
        Note that the input buffer is not the same as the input to the
        Python pickle.loads function.
        '''
        self.seed, num_perm, self.b = struct.unpack_from('<qiB', buffer, 0)
        offset = struct.calcsize('<qiB')
        self.hashvalues = np.zeros((num_perm,), dtype=np.uint32)
        # Reconstruct the hash values
        slot_size, n, num_blocks, total = self._bytesize()
        blocks = struct.unpack_from('%dQ' % num_blocks, buffer, offset)
        mask = (1 << slot_size) - 1
        for i in range(num_blocks):
            start = i * n
            for j in range(n):
                hv = (blocks[i] >> (n - 1 - j) * slot_size) & mask
                self.hashvalues[start+j] = np.uint32(hv)

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
        # Use 8 bytes to store the seed integer
        seed_size = struct.calcsize('q')
        # Use 4 bytes to store the number of hashed values
        length_size = struct.calcsize('i')
        # Use 1 byte to store the parameter b
        b_size = struct.calcsize('B')
        # Get the size of the slot for storing one b-bit hashed value
        slot_size = self._find_slot_size(self.b)
        # Use 64-bit as a block unit
        block_size = struct.calcsize('Q')
        # Get the number of slots to be stored in each block
        n = int(block_size * 8 / slot_size)
        # Get the number of blocks required
        num_blocks = int(np.ceil(float(self.hashvalues.size) / n))
        # Get the total serialized size
        total = struct.calcsize('<qiB%dQ' % num_blocks) 
        return slot_size, n, num_blocks, total
