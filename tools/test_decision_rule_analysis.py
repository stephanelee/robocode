import unittest

from decision_rule_analysis import T95, acceptance, lower_bound, null_guard_rates, score_rule


def block(a_wins, b_wins, a_walls, b_walls):
    return dict(a=dict(firstPlaces=a_wins, metrics=dict(wallCollisions=a_walls, turns=1000)),
                b=dict(firstPlaces=b_wins, metrics=dict(wallCollisions=b_walls, turns=1000)))


class DecisionRuleAnalysisTests(unittest.TestCase):
    def test_t_quantiles_are_monotone_and_above_z(self):
        values = [T95[df] for df in sorted(T95)]
        self.assertTrue(all(a > b for a, b in zip(values, values[1:])))
        self.assertTrue(all(value > 1.6448536 for value in values))
        self.assertAlmostEqual(T95[7], 1.894579, places=6)

    def test_lower_bounds_use_sample_se(self):
        deltas = [.1, .0, .2, .1]  # mean .1, sd .08165, se .040825
        self.assertAlmostEqual(lower_bound(deltas, 'z'), .1 - 1.6448536269514722 * .0408248, places=6)
        self.assertAlmostEqual(lower_bound(deltas, 't'), .1 - 2.353363 * .0408248, places=6)
        self.assertTrue(score_rule([.06, .07, .08, .06], 't', .05))
        self.assertFalse(score_rule([.04, .045, .046, .044], 't', .05))  # significant but below the floor

    def test_t_bound_controls_false_positives_and_floor_caps_acceptance(self):
        # Seeded checks with modest draws: z over-rejects at n = 8, t is close to nominal.
        z0 = acceptance(8, 0.0, .0559, 'z', None, 20000, 1)
        t0 = acceptance(8, 0.0, .0559, 't', None, 20000, 1)
        self.assertGreater(z0, .06)
        self.assertLess(abs(t0 - .05), .01)
        # With an observed-effect floor equal to the true effect, acceptance is near 50% regardless of blocks.
        self.assertLess(abs(acceptance(16, .05, .0559, 't', .05, 8000, 2) - .5), .03)

    def test_null_guard_bootstrap_counts_each_guard(self):
        aa = [block(1, 0, 0, 30), block(0, 1, 30, 0)]
        rates = null_guard_rates(aa, 1, 2000, 3)
        self.assertGreater(rates['winsGuardFail'], .3)
        self.assertGreater(rates['wallGuardFail'], .3)
        self.assertGreaterEqual(rates['eitherFail'], max(rates['winsGuardFail'], rates['wallGuardFail']))


if __name__ == '__main__':
    unittest.main()
