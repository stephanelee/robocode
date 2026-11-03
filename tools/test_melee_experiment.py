import unittest

from melee_experiment import final_decision, melee_decision


def arm(share, wins=1, walls=1, turns=1000, skipped=0, rounds=10):
    return dict(scoreShare=share, firstPlaces=wins, rounds=rounds,
                metrics=dict(turns=turns, wallCollisions=walls, skippedTurns=skipped, damageTaken=50.0,
                             damageDealt=20.0, shotsFired=100, bulletHits=10))


def blocks(deltas, **candidate):
    return [dict(previous=arm(.25), candidate=arm(.25 + d, **candidate)) for d in deltas]


class MeleeExperimentTests(unittest.TestCase):
    def test_superiority_needs_floor_and_positive_t_bound(self):
        good = [.05, .06, .04, .07, .05, .06, .04, .05, .06, .05]
        self.assertEqual(melee_decision(blocks(good), 10)['status'], 'pass')
        noisy = melee_decision(blocks([.20, -.15, .18, -.12, .15, -.10, .10, -.05, .02, .00]), 10)
        self.assertFalse(noisy['checks']['lowerBoundPositive'])
        self.assertFalse(melee_decision(blocks([.015] * 10), 10)['checks']['meanGain'])

    def test_guards(self):
        good = [.05, .06, .04, .07, .05, .06, .04, .05, .06, .05]
        self.assertFalse(melee_decision(blocks(good, wins=0), 10)['checks']['firstPlacesNotLower'])
        self.assertFalse(melee_decision(blocks(good, skipped=2), 10)['checks']['skippedNotMore'])
        self.assertFalse(melee_decision(blocks(good, walls=12), 10)['checks']['wallRate'])
        self.assertFalse(melee_decision(blocks(good), 11)['checks']['complete'])

    def test_final_decision_requires_all_three_stages(self):
        ok = [dict(previous=arm(.95), candidate=arm(.95)) for _ in range(3)]
        passed, failed = dict(status='pass'), dict(status='fail')
        self.assertEqual(final_decision(passed, passed, ok)['status'], 'retain')
        self.assertEqual(final_decision(failed, passed, ok)['status'], 'reject')
        self.assertEqual(final_decision(passed, failed, ok)['status'], 'reject')
        ok[0]['candidate'] = arm(.80)
        self.assertEqual(final_decision(passed, passed, ok)['status'], 'reject')


if __name__ == '__main__':
    unittest.main()
