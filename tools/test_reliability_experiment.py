import unittest

from reliability_experiment import decision, summarize


def arm(skipped, micros=None, slow=None, share=.5):
    return dict(skipped=skipped, maxTurnMicros=micros or [None] * len(skipped), slowTurns=slow, scoreShare=share)


class ReliabilityExperimentTests(unittest.TestCase):
    def test_summary_counts_games_with_skips_and_latency(self):
        games = [dict(previous=arm([1, 0]), candidate=arm([0, 0], [900, 30000], 1)),
                 dict(previous=arm([0, 2]), candidate=arm([0, 0], [800, 700], 0))]
        result = summarize(games)
        self.assertEqual(result['previous']['gamesWithSkips'], 2)
        self.assertEqual(result['previous']['skippedTurns'], 3)
        self.assertIsNone(result['previous']['maxTurnMicros'])
        self.assertEqual(result['candidate']['maxTurnMicros'], 30000)
        self.assertEqual(result['candidate']['slowTurns'], 1)
        self.assertEqual(result['candidate']['gamesWithSkips'], 0)

    def test_decision_distinguishes_effect_neutral_and_regression(self):
        def summary(prev_games, prev_turns, cand_games, cand_turns):
            return dict(previous=dict(gamesWithSkips=prev_games, skippedTurns=prev_turns),
                        candidate=dict(gamesWithSkips=cand_games, skippedTurns=cand_turns))
        self.assertEqual(decision(summary(3, 4, 0, 0))['status'], 'retain-effect')
        self.assertEqual(decision(summary(0, 0, 0, 0))['status'], 'retain-neutral')
        self.assertEqual(decision(summary(1, 1, 1, 1))['status'], 'retain-neutral')
        self.assertEqual(decision(summary(0, 0, 1, 1))['status'], 'reject')
        self.assertEqual(decision(summary(2, 2, 1, 3))['status'], 'reject')


if __name__ == '__main__':
    unittest.main()
