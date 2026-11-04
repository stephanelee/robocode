import copy
import json
from pathlib import Path
import tempfile
import unittest

import ais_breakwater_transfer as experiment


def games(stages):
    result = []
    for stage in stages:
        for design in experiment.layout(stage):
            game = copy.deepcopy(design)
            kind = game['kind']
            previous, candidate = {'Breakwater': (.3, .7), 'Retcon': (.8, .82),
                'diamond-port': (.1, .13), 'adaptive-prime': (.2, .23), 'self': (.3, .7)}.get(kind, (.9, .9))
            game['battles'] = {}
            for arm in game['arms']:
                share = previous if arm == 'previous' else candidate
                scores = [dict(name='own', totalScore=round(1000 * share)), dict(name='other', totalScore=1000 - round(1000 * share))]
                game['battles'][arm] = dict(name='own', totalScore=scores[0]['totalScore'], results=scores,
                    firstPlaces=int(design['rounds'] * share), rounds=design['rounds'],
                    metrics=dict(turns=1000, shotsFired=100, bulletHits=10, damageDealt=40, damageTaken=20,
                                 wallCollisions=0, skippedTurns=0))
            result.append(game)
    return result


class TransferProtocolTests(unittest.TestCase):
    def test_fixed_budget_fresh_ids_and_unique_own_seeds(self):
        population = [g for s in ('screen', 'confirmation', 'samples') for g in experiment.layout(s)]
        self.assertEqual(sum(g['rounds'] for g in population), 630)
        self.assertEqual(sum(g['rounds'] for g in experiment.layout('screen')), 90)
        self.assertEqual(sum(len(g['arms']) for g in population), 46)
        self.assertEqual(len({g['gameId'] for g in population}), len(population))
        seeds = [v for g in population for v in g['seeds'].values()]
        self.assertEqual(len(set(seeds)), len(seeds))
        opponent_seeds = [g['opponentSeed'] for g in population if g['kind'] == 'Breakwater']
        self.assertEqual(len(set(seeds + opponent_seeds)), len(seeds + opponent_seeds))

    def test_screen_and_fresh_confirmation_pass_only_their_whole_populations(self):
        self.assertEqual(experiment.decision(games(['screen']))['status'], 'advance')
        self.assertEqual(experiment.decision(games(['confirmation', 'samples']), True)['status'], 'retain')
        with self.assertRaisesRegex(ValueError, 'population'):
            experiment.decision(games(['screen'])[:-1])
        with self.assertRaisesRegex(ValueError, 'population'):
            experiment.decision(games(['screen']) + games(['confirmation', 'samples']), True)

    def test_baseline_skip_and_self_wall_regression_reject(self):
        rows = games(['screen'])
        rows[0]['battles']['previous']['metrics']['skippedTurns'] = 1
        self.assertFalse(experiment.decision(rows)['checks']['zeroAISSkips'])
        rows = games(['screen'])
        rows[-1]['battles']['candidate']['metrics']['wallCollisions'] = 11
        self.assertFalse(experiment.decision(rows)['checks']['wallRate'])

    def test_movement_activation_matches_executed_default_duel_and_rejects_missing(self):
        self.assertEqual(experiment.movement_activation([dict(forcedDuelTurns=7, breakwaterMovementTurns=7)], 'candidate'), 7)
        self.assertEqual(experiment.movement_activation([dict(forcedDuelTurns=7)], 'previous'), 0)
        for row in (dict(forcedDuelTurns=7), dict(forcedDuelTurns=7, breakwaterMovementTurns=6),
                    dict(forcedDuelTurns=1, breakwaterMovementTurns=True)):
            with self.assertRaisesRegex(ValueError, 'activation'):
                experiment.movement_activation([row], 'candidate')

    def test_retcon_absolute_win_guard_rejects_even_with_positive_delta(self):
        rows = games(['screen'])
        row = next(g for g in rows if g['kind'] == 'Retcon' and 'candidate' in g['arms'])
        row['battles']['candidate']['firstPlaces'] = 5
        self.assertFalse(experiment.decision(rows)['checks']['retconAbsolute'])

    def test_t_lower_bound_uses_game_differences_not_round_counts(self):
        rows = games(['confirmation', 'samples'])
        for row in rows:
            if row['kind'] == 'Retcon' and 'candidate' in row['arms']:
                score = {1: 600, 2: 800, 3: 1000}[row['block']]
                battle = row['battles']['candidate']
                battle['totalScore'] = score
                battle['results'][0]['totalScore'] = score
                battle['results'][1]['totalScore'] = 1000 - score
        result = experiment.decision(rows, True)
        self.assertAlmostEqual(result['opponents']['Retcon']['meanGain'], 0)
        self.assertLess(result['retconTLower95'], -.3)
        self.assertFalse(result['checks']['retconTLowerNI'])

    def test_breakwater_real_endpoint_reconciles_and_preserves_missing_win_flags(self):
        # Read the exact retained official opponent endpoint; no synthetic success evidence.
        import subprocess
        source = 'benchmarks/breakwater-envelope-v3-exact/breakwater-envelope-v3-ais-g01/'
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for destination, original in [('breakwater.jsonl', 'metrics.jsonl'), ('breakwater-model.json', 'model.json')]:
                data = subprocess.check_output(['git', 'show', experiment.BREAKWATER + ':' + source + original])
                (root / destination).write_bytes(data)
            result = experiment.check_breakwater(root, 20, 533021, dict(firstPlaces=20))
            self.assertEqual(result['officialFirstPlaces'], 20)
            self.assertEqual(result['rawWon'], 17)
            endpoint = json.loads((root / 'breakwater-model.json').read_bytes())
            endpoint['totals']['shotsFired'] += 1
            (root / 'breakwater-model.json').write_text(json.dumps(endpoint))
            with self.assertRaisesRegex(ValueError, 'cumulative'):
                experiment.check_breakwater(root, 20, 533021, dict(firstPlaces=20))


if __name__ == '__main__':
    unittest.main()
