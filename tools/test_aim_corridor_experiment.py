import unittest

from aim_corridor_experiment import final_decision, retcon_mechanism, samples_pass


def battle(share, rounds=10, skipped=0):
    return dict(rounds=rounds, totalScore=share, results=[{'totalScore': share}, {'totalScore': 1 - share}],
                metrics=dict(skippedTurns=skipped, damageTaken=10.0, turns=1000),
                retconStats=dict(actualGun=dict(shots=40, hits=6),
                                 virtualGuns=[dict(model='LinearAimModel', hits=5, shots=40),
                                              dict(model='HeadOnAimModel', hits=4, shots=40)]))


class AimCorridorExperimentTests(unittest.TestCase):
    def test_sample_guard_is_inclusive_at_five_points(self):
        rows = [dict(previous=battle(.8), candidate=battle(.75)) for _ in range(3)]
        self.assertTrue(samples_pass(rows))
        rows[1]['candidate'] = battle(.7499)
        self.assertFalse(samples_pass(rows))

    def test_sample_guard_requires_complete_rounds_and_no_skips(self):
        self.assertFalse(samples_pass([dict(previous=battle(.8), candidate=battle(.8, rounds=9))] * 3))
        self.assertFalse(samples_pass([dict(previous=battle(.8), candidate=battle(.8, skipped=1))] * 3))
        self.assertFalse(samples_pass([dict(previous=battle(.8), candidate=battle(.8))] * 2))

    def test_retention_requires_primary_samples_and_online_veto(self):
        rows = [dict(previous=battle(.8), candidate=battle(.8)) for _ in range(3)]
        retain, reject = dict(decision=dict(status='retain')), dict(decision=dict(status='reject'))
        self.assertEqual(final_decision(retain, rows, retain)['status'], 'retain')
        self.assertEqual(final_decision(retain, rows, None)['status'], 'reject')
        self.assertEqual(final_decision(retain, rows, reject)['status'], 'reject')
        self.assertEqual(final_decision(reject, rows, None)['status'], 'reject')
        rows[0]['candidate'] = battle(.5)
        self.assertEqual(final_decision(retain, rows, retain)['status'], 'reject')

    def test_correctness_primary_reports_tactical_gate_separately(self):
        rows = [dict(previous=battle(.8), candidate=battle(.8)) for _ in range(3)]
        primary = dict(decision=dict(status='retain', tacticalStrengthGatePassed=False))
        veto = dict(decision=dict(status='retain'))
        result = final_decision(primary, rows, veto)
        self.assertEqual(result['status'], 'retain')
        self.assertFalse(result['tacticalStrengthGatePassed'])
        self.assertTrue(final_decision(dict(decision=dict(status='retain')), rows, veto)['tacticalStrengthGatePassed'])

    def test_mechanism_aggregates_retcon_counters_per_arm(self):
        comparison = dict(blocks=[dict(previous=battle(.2), candidate=battle(.3)) for _ in range(3)])
        result = retcon_mechanism(comparison)
        self.assertEqual(result['candidate']['actualShots'], 120)
        self.assertAlmostEqual(result['previous']['actualHitRate'], .15)
        self.assertEqual(result['previous']['virtual']['HeadOnAimModel']['hits'], 12)


if __name__ == '__main__':
    unittest.main()
