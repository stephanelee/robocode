import copy
import unittest
from frozen_duel_confirmation import decision


def games(scores, wins=11):
    return [dict(gameId=str(i), battle=dict(rounds=20, firstPlaces=wins, totalScore=s,
        results=[dict(totalScore=s), dict(totalScore=100-s)])) for i, s in enumerate(scores)]


class ConfirmationTests(unittest.TestCase):
    def test_strict_three_gates_and_equal_game_aggregation(self):
        self.assertEqual(decision(games([60]*10))['status'], 'confirmed')
        self.assertFalse(decision(games([50]*10))['checks']['meanShare'])
        self.assertFalse(decision(games([60]*10, wins=10))['checks']['majorityRoundWins'])
        self.assertFalse(decision(games([90]*5+[49]*5))['checks']['majorityGames'])
        self.assertEqual(decision(games([51]*6+[49]*4))['status'], 'confirmed')

    def test_incomplete_or_duplicate_games_rejected(self):
        rows = games([60]*10)
        for changed in (rows[:-1], rows+[rows[0]]):
            with self.assertRaises(ValueError):
                decision(changed)
        changed = copy.deepcopy(rows); changed[1]['gameId'] = changed[0]['gameId']
        with self.assertRaises(ValueError):
            decision(changed)
        changed = copy.deepcopy(rows); changed[0]['battle']['rounds'] = 19
        with self.assertRaises(ValueError):
            decision(changed)


if __name__ == '__main__':
    unittest.main()
