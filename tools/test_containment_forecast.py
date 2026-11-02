"""Synthetic proofs for the prospectively fixed candidate scorer."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from containment_forecast import BinaryMoments, CANDIDATES, confirmation_decision, decision, fit, predict, screen, validate_fresh
from legal_scan_dataset import build_samples
from motion_forecast import FEATURES, HORIZONS
from test_motion_forecast import rows
from test_legal_scan_dataset import manifest


class ContainmentTests(unittest.TestCase):
    def test_binary_intercepts_and_constant_columns(self):
        moments = BinaryMoments()
        for _ in range(3):
            moments.add([7] * len(FEATURES), [1, 0, 1, 0])
        model = moments.model()
        self.assertEqual(model['intercept'], [1, 0, 1, 0])
        self.assertTrue(all(c == 0 for coefficients in model['coefficients'] for c in coefficients))

    def test_dev_and_spent_games_cannot_change_fit(self):
        records = rows()
        original = fit(records, manifest())
        for record in records:
            if record['gameId'] != 'A':
                record['scan']['x'] += 300
                record['own']['speed'] = -8
        self.assertEqual(original, fit(records, manifest()))

    def test_rank_ties_current_and_stale_fallback(self):
        model = fit(rows(), manifest())
        sample = list(build_samples(rows()[:64], manifest(), (8,), 8))[3]
        for m in model['models'].values():
            m['intercept'] = [0] * 4
            m['coefficients'] = [[0] * len(FEATURES) for _ in range(4)]
        self.assertEqual(predict(model, sample, manifest()['games'][0])[1], 'current')
        model['models']['8']['intercept'][3] = 2  # raw scores not clipped probabilities
        self.assertEqual(predict(model, sample, manifest()['games'][0])[1], 'headon')
        sample['anchorTurn'] += 3
        result = predict(model, sample, manifest()['games'][0])
        self.assertEqual(result[1], 'current')
        self.assertFalse(result[2])

    def test_full_mask_screen_coverage_and_selection_counts(self):
        model = fit(rows(), manifest())
        cells = screen(rows(), manifest(), model)
        self.assertEqual(set(cells), {'B'})
        self.assertEqual(cells['B']['40']['anchors'], 64)
        self.assertEqual(cells['B']['40']['observed'], 24)
        self.assertEqual(sum(cells['B']['40']['selections'].values()), 24)
        self.assertIn('eligibleContainment', cells['B']['40']['methods']['scorer'])
        self.assertEqual(decision(cells)['status'], 'inconclusive')

    def test_unrounded_gain_majority_short_and_missing_gates(self):
        cells = {g: {str(h): {'observed': 50, 'methods': {
            'current': {'containment': .1}, 'scorer': {'containment': .13}}} for h in HORIZONS} for g in ('a', 'b', 'c')}
        self.assertEqual(decision(cells)['status'], 'development-pass')
        for game in cells.values():
            for h in (24, 32, 40):
                game[str(h)]['methods']['scorer']['containment'] = .1199999999
        self.assertFalse(decision(cells)['checks']['primaryGain'])
        cells['a']['24']['observed'] = 49
        self.assertEqual(decision(cells)['status'], 'inconclusive')

    def test_fresh_games_reject_spent_ids_and_preserve_training_provenance(self):
        spec = manifest()
        model = dict(priorGameIds=['C'], priorAcquisitionIds=['capture-C'], captureProvenance=spec['provenance'])
        fresh = dict(spec, games=[dict(spec['games'][2], gameId='w' + str(i), acquisitionId='new-w' + str(i)) for i in range(3)] +
            [dict(spec['games'][3], gameId='x' + str(i), acquisitionId='new-x' + str(i)) for i in range(3)])
        validate_fresh(fresh, model)
        bad = copy.deepcopy(fresh)
        bad['games'][0]['acquisitionId'] = 'capture-C'
        with self.assertRaises(ValueError):
            validate_fresh(bad, model)
        bad = copy.deepcopy(fresh)
        bad['provenance']['botJarSha256'] = '1' * 64
        with self.assertRaises(ValueError):
            validate_fresh(bad, model)

    def test_confirmation_transfer_veto(self):
        spec = manifest()
        spec['games'] = [dict(spec['games'][2], gameId='w' + str(i), acquisitionId='w' + str(i)) for i in range(3)] + [
            dict(spec['games'][3], gameId='x' + str(i), acquisitionId='x' + str(i)) for i in range(3)]
        cells = {g['gameId']: {str(h): {'observed': 50, 'methods': {
            'current': {'containment': .1, 'meanError': 100},
            'scorer': {'containment': .13, 'meanError': 101}}} for h in HORIZONS} for g in spec['games']}
        self.assertEqual(confirmation_decision(cells, spec)['status'], 'offline-confirmation-pass')
        cells['x0']['24']['methods']['scorer']['meanError'] = 116.000001
        self.assertFalse(confirmation_decision(cells, spec)['checks']['transfer'])

    def test_cli_frozen_hash_screen_and_output_preservation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'manifest.json').write_text(json.dumps(manifest()))
            (root / 'events.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows()))
            cmd = [sys.executable, str(Path(__file__).with_name('containment_forecast.py')), '--manifest',
                   str(root / 'manifest.json'), '--events', str(root / 'events.jsonl')]
            fit_result = subprocess.run(cmd + ['--output', str(root / 'fit')], capture_output=True, text=True)
            self.assertEqual(fit_result.returncode, 0, fit_result.stderr)
            report = json.loads((root / 'fit' / 'report.json').read_text())
            self.assertNotIn('cells', report)
            model_args = ['--model', str(root / 'fit' / 'model.json'), '--model-sha256', report['modelSha256']]
            result = subprocess.run(cmd + model_args + ['--stage', 'screen', '--output', str(root / 'screen')], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            bad = subprocess.run(cmd + model_args + ['--stage', 'confirm', '--output', str(root / 'confirm')], capture_output=True)
            self.assertNotEqual(bad.returncode, 0)
            self.assertFalse((root / 'confirm').exists())
            again = subprocess.run(cmd + ['--output', str(root / 'fit')], capture_output=True)
            self.assertNotEqual(again.returncode, 0)


if __name__ == '__main__':
    unittest.main()
