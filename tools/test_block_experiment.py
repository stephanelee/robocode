import unittest

from block_experiment import final_decision, online_decision, primary_decision, samples_pass


def arm(share, wins=1, walls=1, turns=1000, skipped=0, rounds=20):
    return dict(scoreShare=share, firstPlaces=wins, rounds=rounds,
                metrics=dict(turns=turns, wallCollisions=walls, skippedTurns=skipped, damageTaken=50.0,
                             damageDealt=20.0, shotsFired=100, bulletHits=10))


def blocks(deltas, **candidate):
    return [dict(previous=arm(.2), candidate=arm(.2 + d, **candidate)) for d in deltas]


class BlockExperimentTests(unittest.TestCase):
    def test_primary_requires_gain_and_positive_lower_bound(self):
        self.assertEqual(primary_decision(blocks([.06, .07, .05, .08, .06, .07, .05, .06]), 8)['status'], 'pass')
        noisy = primary_decision(blocks([.30, -.20, .25, -.15, .20, -.10, .10, .00]), 8)
        self.assertTrue(noisy['checks']['meanGain'])
        self.assertFalse(noisy['checks']['lowerBoundPositive'])
        self.assertFalse(primary_decision(blocks([.04] * 7 + [.05]), 8)['checks']['meanGain'])

    def test_primary_guards(self):
        good = [.06, .07, .05, .08, .06, .07, .05, .06]
        self.assertFalse(primary_decision(blocks(good, wins=0), 8)['checks']['winsNotLower'])
        self.assertFalse(primary_decision(blocks(good, skipped=1), 8)['checks']['noSkippedTurns'])
        # Wall guard is absolute: baseline 1/1000 turns -> candidate up to 11/1000 turns passes.
        self.assertTrue(primary_decision(blocks(good, walls=11), 8)['checks']['wallRate'])
        self.assertFalse(primary_decision(blocks(good, walls=12), 8)['checks']['wallRate'])
        self.assertFalse(primary_decision(blocks(good), 9)['checks']['complete'])

    def test_online_non_inferiority(self):
        self.assertEqual(online_decision(blocks([-.01, .0, .01, -.02, .02, .0, -.01, .01]), 8)['status'], 'pass')
        self.assertEqual(online_decision(blocks([-.08] * 8), 8)['status'], 'fail')

    def test_samples_and_final_decision(self):
        rows = [dict(previous=arm(.95, rounds=10), candidate=arm(.90, rounds=10)) for _ in range(3)]
        self.assertTrue(samples_pass(rows))
        rows[0]['candidate'] = arm(.8999, rounds=10)
        self.assertFalse(samples_pass(rows))
        ok = [dict(previous=arm(.95, rounds=10), candidate=arm(.95, rounds=10)) for _ in range(3)]
        passed, failed = dict(status='pass'), dict(status='fail')
        self.assertEqual(final_decision(passed, ok, passed)['status'], 'retain')
        self.assertEqual(final_decision(passed, ok, None)['status'], 'reject')
        self.assertEqual(final_decision(failed, ok, None)['status'], 'reject')
        self.assertEqual(final_decision(passed, rows, passed)['status'], 'reject')


    def test_student_t_bound_is_stricter_than_z(self):
        deltas = [.30, -.05, .20, .02, .15, -.02, .10, .00]
        z = primary_decision(blocks(deltas), 8, 'z')
        t = primary_decision(blocks(deltas), 8, 't')
        self.assertAlmostEqual(t['quantile'], 1.894579)
        self.assertLess(t['lowerBound95'], z['lowerBound95'])
        self.assertAlmostEqual(z['meanDelta'], t['meanDelta'])
        with self.assertRaises(ValueError):
            primary_decision(blocks(deltas), 8, 'normal')

if __name__ == '__main__':
    unittest.main()
