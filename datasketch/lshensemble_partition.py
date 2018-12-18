"""
Assume we want to create partition intervals:

    [l1, u1], [l2, u2], ...

with all boundaries inclusive.
"""
import numpy as np

def compute_nfp_uniform(l, u, cum_counts, sizes):
    """Computes the expected number of false positives caused by using
    u to approximate set sizes in the interval [l, u], assuming uniform
    distribution of set sizes within the interval.

    Args:
        l: the lower bound on set sizes.
        u: the upper bound on set sizes.
        cum_counts: the complete cummulative distribution of set sizes.
        sizes: the complete domain of set sizes.

    Return (float): the expected number of false positives.
    """
    if l > u:
        raise ValueError("l must be less or equal to u")
    if l == 0:
        n = cum_counts[u]
    else:
        n = cum_counts[u]-cum_counts[l-1]
    return n * float(sizes[u] - sizes[l]) / float(2*sizes[u])

def compute_nfps_uniform(cum_counts, sizes):
    """Computes the matrix of expected false positives for all possible
    sub-intervals of the complete domain of set sizes, assuming uniform
    distribution of set_sizes within each sub-intervals.

    Args:
        cum_counts: the complete cummulative distribution of set sizes.
        sizes: the complete domain of set sizes.

    Return (np.array): the 2-D array of expected number of false positives
        for every pair of [l, u] interval, where l is axis-0 and u is
        axis-1.
    """
    nfps = np.zeros((len(sizes), len(sizes)))
    # All u an l are inclusive bounds for intervals.
    # Compute p = 1, the NFPs
    for l in range(len(sizes)):
        for u in range(l, len(sizes)):
            nfps[l, u] = compute_nfp_uniform(l, u, cum_counts, sizes)
    return nfps

def compute_nfp_real(l, u, counts, sizes):
    """Computes the expected number of false positives caused by using
    u to approximate set sizes in the interval [l, u], using the real
    set size distribution.

    Args:
        l: the lower bound on set sizes.
        u: the upper bound on set sizes.
        counts: the complete distribution of set sizes.
        sizes: the complete domain of set sizes.

    Return (float): the expected number of false positives.
    """
    if l > u:
        raise ValueError("l must be less or equal to u")
    return np.sum((float(sizes[u])-sizes[l:u+1])/float(sizes[u])*counts[l:u+1])

def compute_nfps_real(counts, sizes):
    """Computes the matrix of expected false positives for all possible
    sub-intervals of the complete domain of set sizes.

    Args:
        counts: the complete distribution of set sizes.
        sizes: the complete domain of set sizes.

    Return (np.array): the 2-D array of expected number of false positives
        for every pair of [l, u] interval, where l is axis-0 and u is
        axis-1.
    """
    nfps = np.zeros((len(sizes), len(sizes)))
    # All u an l are inclusive bounds for intervals.
    # Compute p = 1, the NFPs
    for l in range(len(sizes)):
        for u in range(l, len(sizes)):
            nfps[l, u] = compute_nfp_real(l, u, counts, sizes)
    return nfps

def compute_best_partitions(num_part, sizes, nfps):
    """Computes the optimal partitions given the size distributions
    and computed number of expected false positives for all sub-intervals.

    Args:
        num_part (int): number of partitions.
        sizes (numpy.array): set sizes in sorted order.
        nfps (numpy.array): the computed number of expected false positives
            for all sub-intervals; axis-0 is for lower bounds and axis-1 is
            for upper bounds.

    Returns:
        partitions (list): list of lower and upper bounds of set sizes for
            all partitions.
        total_nfps (float): total number of expected false positives from all
            partitions.
        cost (numpy.array): a N x p-1 matrix of the computed optimal NFPs for
            all sub-problems given upper bound set size and number of partitions.
    """

    if num_part < 2:
        raise ValueError("num_part cannot be less than 2")

    # If number of partitions is 2, then simply find the upper bound
    # of the first partition.
    if num_part == 2:
        total_nfps, u = min((nfps[0, u1]+nfps[u1+1, len(sizes)-1], u1)
            for u1 in range(0, len(sizes)-1))
        return [(sizes[0], sizes[u]), (sizes[u+1], sizes[-1]),], \
                total_nfps, None

    # Initialize subproblem total NFPs.
    cost = np.zeros((len(sizes), num_part-2))

    # Note: p is the number of partitions in the subproblem.
    # p2i translate the number of partition into the index in the matrix.
    p2i = lambda p : p - 2

    # Back track to find the best partitions.
    def _back_track(cost, p):
        # Find the best right-most upper bound (before the end) given
        # the number of partitions and upper bound.
        total_nfps, u = min((cost[u1, p2i(p-1)]+nfps[u1+1, len(sizes)-1], u1)
                for u1 in range((p-1)-1, len(sizes)-1))
        partitions = [(sizes[u+1], sizes[-1]),]
        p -= 1
        while p > 1:
            # Find the best right-most upper bound (before the end) given
            # the number of partitions and upper bound in the sub-problem.
            _, u1_best = min((cost[u1, p2i(p-1)]+nfps[u1+1, len(sizes)-1], u1)
                    for u1 in range((p-1)-1, u))
            partitions.insert(0, (sizes[u1_best+1], sizes[u]))
            u = u1_best
            p -= 1
        partitions.insert(0, (sizes[0], sizes[u]))
        return [partitions, total_nfps]

    # Compute p >= 2, and returns if p = num_part.
    for p in range(2, num_part+1):
        # Early return if p = num_part
        if p == num_part:
            return _back_track(cost, p) + [cost, ]
        # Compute best partition for subproblems with increasing
        # max index u, starting from the smallest possible u given the p.
        # The smallest possible u can be considered as the max index that
        # generates p partitions each with only one size.
        for u in range(p-1, len(sizes)):
            if p == 2:
                cost[u, p2i(p)] = min(nfps[0, u1]+nfps[u1+1,u]
                        for u1 in range(u))
            else:
                cost[u, p2i(p)] = min(cost[u1, p2i(p-1)] + nfps[u1+1, u]
                        for u1 in range((p-1)-1, u))
    raise RuntimeError("Function should have returned before here.")

