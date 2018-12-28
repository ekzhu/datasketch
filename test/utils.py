"""
Test utilities.
"""

def fake_hash_func(data):
    if not isinstance(data, int):
        raise ValueError("Fake hash function only takes integer as input")
    return data

