import hashlib
import struct

def sha1_hash8(data):
    """A 8-bit hash function based on SHA1.

    Args:
        data (bytes): the data to generate 8-bit integer hash from.

    Returns:
        int: an integer hash value that can be encoded using 8 bits.
    """
    return struct.unpack('<B', hashlib.sha1(data).digest()[:1])[0]

def sha1_hash16(data):
    """A 16-bit hash function based on SHA1.

    Args:
        data (bytes): the data to generate 16-bit integer hash from.

    Returns:
        int: an integer hash value that can be encoded using 16 bits.
    """
    return struct.unpack('<H', hashlib.sha1(data).digest()[:2])[0]

def sha1_hash32(data):
    """A 32-bit hash function based on SHA1.

    Args:
        data (bytes): the data to generate 32-bit integer hash from.

    Returns:
        int: an integer hash value that can be encoded using 32 bits.
    """
    return struct.unpack('<I', hashlib.sha1(data).digest()[:4])[0]

def sha1_hash64(data):
    """A 64-bit hash function based on SHA1.

    Args:
        data (bytes): the data to generate 64-bit integer hash from.

    Returns:
        int: an integer hash value that can be encoded using 64 bits.
    """
    return struct.unpack('<Q', hashlib.sha1(data).digest()[:8])[0]

def sha1_hash128(data):
    """A 128-bit hash function based on SHA1.

    Args:
        data (bytes): the data to generate 128-bit integer hash from.

    Returns:
        int: an integer hash value that can be encoded using 128 bits.
    """
    return struct.unpack('<Q', hashlib.sha1(data).digest()[:16])[0]