"""Exact balanced treatment-order randomization, issue134; no engine dependency."""
import bisect
import itertools
import math
from fractions import Fraction


def unrank(n, k, rank):
    if not 0 <= rank < math.comb(n, k):
        raise ValueError('assignment rank outside uniform sample space')
    result = []
    start = 0
    for remaining in range(k, 0, -1):
        for value in range(start, n - remaining + 1):
            count = math.comb(n - value - 1, remaining - 1)
            if rank < count:
                result.append(value)
                start = value + 1
                break
            rank -= count
    return result


def subset_sums(values):
    groups = [[] for _ in range(len(values) + 1)]
    for mask in range(1 << len(values)):
        groups[mask.bit_count()].append(sum(v for i, v in enumerate(values) if mask >> i & 1))
    return [sorted(g) for g in groups]


def exact_tail(slot_contrasts, candidate_first):
    """Inclusive upper tail over all balanced assignments, using exact integers."""
    x = list(map(Fraction, slot_contrasts))
    n = len(x)
    if not n or n % 2 or len(candidate_first) != n // 2 or len(set(candidate_first)) != n // 2 or any(i not in range(n) for i in candidate_first):
        raise ValueError('balanced unique candidate-first subset required')
    scale = math.lcm(*(v.denominator for v in x))
    weights = [int(v * scale) for v in x]
    target = sum(weights[i] for i in candidate_first)
    left = subset_sums(weights[:n // 2])
    right = subset_sums(weights[n // 2:])
    tail = 0
    for k, sums in enumerate(left):
        other = right[n // 2 - k]
        for value in sums:
            tail += len(other) - bisect.bisect_left(other, target - value)
    total = math.comb(n, n // 2)
    p = Fraction(tail, total)
    return dict(tail=tail, assignments=total, pExact=str(p), p=float(p),
                observedTExact=str(Fraction(2 * target - sum(weights), scale)))
