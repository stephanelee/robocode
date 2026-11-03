import copy
import argparse
import contextlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import frozen_duel_confirmation as confirmation
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

    def test_relative_paths_become_absolute_before_child_process_and_loaded_model_fails(self):
        layout = [dict(gameId='one', batch=1, index=1, seed=1)]
        with tempfile.TemporaryDirectory() as directory, contextlib.chdir(directory):
            args = argparse.Namespace(output=Path('batch'), bot_dir=Path('bot'), cache=Path('cache'),
                                      plan_commit='a'*40, batch=1, java='java')
            def battle(config, output, runner, samples, scenario, opponents, policy, rounds, model):
                self.assertTrue(config.bot_dir.is_absolute())
                self.assertTrue(output.is_absolute())
                self.assertTrue(model.is_absolute())
                return dict(metrics=dict(skippedTurns=0, checkpointLoaded=True), modelSHA256=None)
            with patch.object(confirmation, 'registration', return_value=(layout, b'{}', b'plan')), \
                 patch.object(confirmation.subprocess, 'check_output', return_value='source'), \
                 patch.object(confirmation.subprocess, 'run', return_value=argparse.Namespace(stdout='', stderr='java25')), \
                 patch.object(confirmation.benchmark, 'digest', return_value=confirmation.JAR), \
                 patch.object(confirmation.retcon, 'prepare', return_value=(Path('source'), Path('api'), {})), \
                 patch.object(confirmation.benchmark, 'download', return_value=Path('runner')), \
                 patch.object(confirmation.benchmark, 'battle', side_effect=battle):
                with self.assertRaisesRegex(ValueError, 'no model loads'):
                    confirmation.run(args)
            self.assertTrue(args.cache.is_absolute())
        changed = copy.deepcopy(rows); changed[0]['battle']['rounds'] = 19
        with self.assertRaises(ValueError):
            decision(changed)


if __name__ == '__main__':
    unittest.main()
