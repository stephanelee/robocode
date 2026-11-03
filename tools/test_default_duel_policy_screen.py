import unittest
from default_duel_policy_screen import decision, mechanism


def battle(share, rounds=5, wins=1):
    return dict(rounds=rounds, firstPlaces=wins, totalScore=share,
        results=[{'totalScore': share}, {'totalScore': 1-share}],
        metrics=dict(turns=1000, shotsFired=100, bulletHits=10, damageDealt=20,
                     damageTaken=40, wallCollisions=1, skippedTurns=0))


class DefaultPolicyScreenTests(unittest.TestCase):
    def test_exact_screen_guards_and_no_retention_label(self):
        blocks = [dict(previous=battle(.2), candidate=battle(.3, wins=2)) for _ in range(3)]
        samples = [dict(opponent=name, previous=battle(.9, 3), candidate=battle(.9, 3)) for name in ('Target', 'SpinBot', 'Walls')]
        self.assertEqual(decision(blocks, samples)['status'], 'advance-to-confirmation')
        blocks[2]['candidate'] = battle(.0999999)
        self.assertFalse(decision(blocks, samples)['checks']['worstBlock'])
        samples[0]['candidate'] = battle(.8, 3)
        self.assertFalse(decision(blocks, samples)['checks']['samples'])

    def test_mechanism_requires_eligible_forced_and_executed_credit(self):
        row = dict(duelPolicy='controller', duelEligibleTurns=3, forcedDuelTurns=3, actionCounts=[3, 1, 0, 0, 0, 0])
        self.assertEqual(mechanism([row])['forcedDuelTurns'], 3)
        for changed in (dict(row, forcedDuelTurns=2), dict(row, actionCounts=[2, 0, 0, 0, 0, 0]),
                        dict(row, duelPolicy='learned'), dict(row, duelEligibleTurns=0, forcedDuelTurns=0)):
            with self.assertRaises(ValueError):
                mechanism([changed])


if __name__ == '__main__':
    unittest.main()
