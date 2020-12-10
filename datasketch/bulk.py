from datasketch.minhash import MinHash
from collections.abc import Iterable

def compute_minhashes(b, **minhash_kwargs):
    '''Helper method to compute minhashes in bulk. This helper avoids unnecessary
    overhead when initializing many minhashes by reusing initial state.

    Args:
        b (iterable): Iterable containing lists of bytes
        minhash_kwargs: Keyword arguments used to initialize MinHash object
            The configuration of this MinHash will be used for all minhashes
    '''
    return list(compute_minhashes_generator(b, **minhash_kwargs))

def compute_minhashes_generator(b, **minhash_kwargs):
    '''Helper method to compute minhashes in bulk. This helper avoids unnecessary
    overhead when initializing many minhashes by reusing initial state. This method
    returns a generator for streaming computation.

    Args:
        b (iterable): Iterable containing lists of bytes
        minhash_kwargs: Keyword arguments used to initialize MinHash object
            The configuration of this MinHash will be used for all minhashes
    '''
    m = MinHash(**minhash_kwargs)
    assert isinstance(b, Iterable), TypeError(f'Expecting iterable, given: {type(b)}')
    for _b in b:
        _m = m.copy()
        _m.update_batch(_b)
        yield _m