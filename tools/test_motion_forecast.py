"""Synthetic leakage/numerical checks; no real capture or model tuning."""
import copy
import math
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from legal_scan_dataset import build_samples
from motion_forecast import (FEATURES, HORIZONS, Moments, baseline, decision,
                             evaluate, fit, inputs, predict, select_baseline, solve)
from test_legal_scan_dataset import event, manifest


def rows():
    return [event(t, game=g) for g in ('A', 'B', 'C', 'D') for t in range(1, 65)]


class ForecastTests(unittest.TestCase):
    def test_cholesky_known_system(self):
        result = solve([[4, 2], [2, 3]], [6, 5])
        self.assertAlmostEqual(result[0], 1)
        self.assertAlmostEqual(result[1], 1)

    def test_unpenalized_intercept_and_constant_columns(self):
        moments = Moments()
        for _ in range(4):
            moments.add([7] * len(FEATURES), [13, -4])
        model = moments.model()
        self.assertEqual(model['intercept'], [13, -4])
        self.assertTrue(all(v == 0 for output in model['coefficients'] for v in output))

    def test_heldout_values_cannot_change_fit_or_normalization(self):
        before = fit(rows(), manifest())
        changed = rows()
        for row in changed:
            if row['gameId'] != 'A':
                row['scan']['x'] += 500
                row['own']['speed'] = -8
        self.assertEqual(before, fit(changed, manifest()))

    def test_labels_ids_do_not_enter_feature_vector(self):
        sample = list(build_samples(rows()[:64], manifest(), (8,), 8))[3]
        before = inputs(sample, manifest()['games'][0])
        sample['label']['x'] = 999
        sample['gameId'] = 'unrelated'
        sample['targetId'] = 99
        self.assertEqual(before, inputs(sample, manifest()['games'][0]))
        self.assertEqual(len(before[0]), 24)

    def test_scan_age_in_projection_and_final_wall_clamp(self):
        sample = list(build_samples([event(1), event(3, observed=1, x=101)], manifest(), (8,), 8))[-1]
        game = manifest()['games'][0]
        self.assertEqual(baseline(sample, game, 'linear'), [111, 100])
        sample['features']['history'][-1]['scan']['x'] = 1590
        self.assertEqual(baseline(sample, game, 'current')[0], 1582)

    def test_fallback_is_frozen_baseline_and_missing_labels_count(self):
        data, spec = rows(), manifest()
        model = fit(data, spec)
        model['selectedBaseline'] = 'headon'
        sample = list(build_samples(data[:64], spec, (8,), 8))[0]
        point, eligible = predict(model, sample, spec['games'][0])
        self.assertFalse(eligible)
        self.assertEqual(point, baseline(sample, spec['games'][0], 'headon'))
        cells = evaluate(data, spec, model, ('dev',))
        self.assertEqual(cells['B']['40']['anchors'], 64)
        self.assertEqual(cells['B']['40']['observed'], 24)
        self.assertLess(cells['B']['40']['eligible'], 24)
        self.assertIn('eligibleMeanError', cells['B']['40']['methods']['ridge'])

    def test_zero_error_baseline_deterministic_dev_selection(self):
        data, spec = rows(), manifest()
        dev = evaluate(data, spec, None, ('dev',))
        self.assertEqual(select_baseline(dev), 'linear')
        self.assertEqual(dev['B']['24']['methods']['linear']['meanError'], 0)

    def test_missing_registered_games_are_inconclusive_not_dropped(self):
        data, spec = rows(), manifest()
        model = fit(data, spec)
        model['selectedBaseline'] = 'linear'
        cells = evaluate(data, spec, model, ('withheld', 'transfer'))
        self.assertEqual(decision(cells, spec, 'linear')['status'], 'inconclusive')

    def test_confirmed_shot_dedup_and_intent_missingness(self):
        data = [event(1), event(2)]
        shot = dict(turn=1, bulletId=1, power=2, direction=0, x=400, y=300)
        for row in data:
            row['confirmedShots'] = [copy.deepcopy(shot)]
        sample = list(build_samples(data, manifest(), (8,), 8))[-1]
        values, _, _ = inputs(sample, manifest()['games'][0])
        self.assertEqual(values[18:22], [1, 2, 1, 1])
        self.assertEqual(values[16:18], [0, 0])

    def test_unrounded_gate_boundary_and_coverage_veto(self):
        spec = manifest()
        spec['games'] = [dict(spec['games'][2], gameId='w' + str(i), acquisitionId='w' + str(i)) for i in range(3)] + [
            dict(spec['games'][3], gameId='x' + str(i), acquisitionId='x' + str(i)) for i in range(3)]
        cells = {g['gameId']: {str(h): {'observed': 50, 'methods': {
            'linear': {'meanError': 100, 'containment': .5},
            'ridge': {'meanError': 89, 'containment': .53}}} for h in HORIZONS} for g in spec['games']}
        self.assertEqual(decision(cells, spec, 'linear')['status'], 'offline-pass')
        for g in ('w0', 'w1', 'w2'):
            for h in (24, 32, 40):
                cells[g][str(h)]['methods']['ridge']['meanError'] = 90.00000001
        self.assertFalse(decision(cells, spec, 'linear')['checks']['primaryError'])
        cells['w0']['24']['observed'] = 49
        self.assertEqual(decision(cells, spec, 'linear')['status'], 'inconclusive')

    def test_cli_requires_frozen_hash_and_preserves_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'manifest.json').write_text(json.dumps(manifest()))
            (root / 'events.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows()))
            command = [sys.executable, str(Path(__file__).with_name('motion_forecast.py')),
                       '--manifest', str(root / 'manifest.json'), '--events', str(root / 'events.jsonl')]
            result = subprocess.run(command + ['--output', str(root / 'fit')], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((root / 'fit' / 'report.json').read_text())
            self.assertNotIn('cells', report)
            bad = subprocess.run(command + ['--stage', 'evaluate', '--model', str(root / 'fit' / 'model.json'),
                '--model-sha256', '0' * 64, '--output', str(root / 'bad')], capture_output=True)
            self.assertNotEqual(bad.returncode, 0)
            self.assertFalse((root / 'bad').exists())
            good = subprocess.run(command + ['--stage', 'evaluate', '--model', str(root / 'fit' / 'model.json'),
                '--model-sha256', report['modelSha256'], '--output', str(root / 'heldout')], capture_output=True, text=True)
            self.assertEqual(good.returncode, 0, good.stderr)
            again = subprocess.run(command + ['--output', str(root / 'fit')], capture_output=True)
            self.assertNotEqual(again.returncode, 0)


if __name__ == '__main__':
    unittest.main()
