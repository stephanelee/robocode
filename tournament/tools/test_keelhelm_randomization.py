import itertools
import math
import unittest
from fractions import Fraction
from keelhelm_randomization import exact_tail, unrank


class ExactRandomization(unittest.TestCase):
    def test_unranking_is_bijection_in_lexicographic_order(self):
        for n in (2, 4, 6, 8):
            expected = list(itertools.combinations(range(n), n // 2))
            self.assertEqual(expected, [tuple(unrank(n, n // 2, i)) for i in range(len(expected))])
        with self.assertRaises(ValueError):
            unrank(32, 16, math.comb(32, 16))

    def test_exact_fraction_tail_matches_exhaustive_assignment_distribution(self):
        x = list(map(Fraction, ['1/3', '-2/7', '1/3', '0', '4/9', '-8/11']))
        assignments = list(itertools.combinations(range(6), 3))
        stats = [2 * sum((x[i] for i in s), Fraction()) - sum(x) for s in assignments]
        for s, observed in zip(assignments, stats):
            result = exact_tail(x, s)
            expected = sum(t >= observed for t in stats)
            self.assertEqual(expected, result['tail'])
            self.assertEqual(str(observed), result['observedTExact'])
            self.assertEqual(Fraction(expected, 20), Fraction(result['pExact']))
        # Under the sharp null, assignments are uniform: rejection frequency <= alpha.
        self.assertLessEqual(sum(Fraction(exact_tail(x,s)['pExact']) <= Fraction(1,20) for s in assignments), 1)

    def test_inclusive_ties_and_unlabelled_slot_orientation(self):
        self.assertEqual('1', exact_tail([0]*32, list(range(16)))['pExact'])
        self.assertEqual('1/6', exact_tail([4,3,-2,-1], [0,1])['pExact'])
        self.assertEqual('1', exact_tail([4,3,-2,-1], [2,3])['pExact'])
        self.assertEqual(exact_tail([4,3,-2,-1],[0,1]), exact_tail([-4,-3,2,1],[2,3]))


if __name__ == '__main__':
    unittest.main()
