"""Synthetic V3 feature/timing/isolation checks; never a real fit or retune."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import containment_forecast as v2
import containment_forecast_v3 as v3
import motion_forecast as v1
from legal_scan_dataset import build_samples
from own_wave_context_diagnostic import contexts
from test_legal_scan_dataset import event, manifest
from test_motion_forecast import rows
from test_own_wave_context_diagnostic import shot


class ContainmentV3Tests(unittest.TestCase):
    def sample_context(self):
        records = [event(t, x=500) for t in (1, 9, 10)]
        records[0]['confirmedShots'] = [shot()]
        sample = list(build_samples(records, manifest(), (8,), 8))[-1]
        return sample, dict(contexts(records, manifest()))[('A', 1, 10, 2)]

    def test_exact_prefix_and_original_eligibility(self):
        sample, context = self.sample_context()
        game = manifest()['games'][0]
        base = v1.inputs(sample, game)
        extended = v3.inputs(sample, game, context)
        self.assertEqual(extended[0][:24], base[0])
        self.assertEqual(extended[1:], base[1:])
        self.assertEqual(len(extended[0]), 26)
        self.assertEqual(extended[0][24], 1)
        self.assertAlmostEqual(extended[0][25], (500 - 14 * 10) / 14)

    def test_exact_frozen_boundaries(self):
        sample, context = self.sample_context()
        game = manifest()['games'][0]
        for field, value, expected in (('eta', 1, 0), ('eta', 40, 1), ('eta', 40.000001, 0),
                                       ('eventAge', 96, 1), ('eventAge', 97, 0),
                                       ('firstSeenTurn', 3, 0), ('firstSeenTurn', 2, 1)):
            changed = copy.deepcopy(context)
            changed['earliestEligible'][field] = value
            self.assertEqual(v3.inputs(sample, game, changed)[0][24], expected, (field, value))
        absent = dict(context, earliestEligible=None)
        self.assertEqual(v3.inputs(sample, game, absent)[0][24:], [0, 0])

    def test_global_earliest_recent_wave_prevents_using_other_old_wave(self):
        sample, context = self.sample_context()
        recent = dict(context['earliestEligible'], firstSeenTurn=10, eventAge=0, eta=2)
        context['earliestEligible'] = recent
        self.assertTrue(context['estimatedWaves'][0]['missingFromFirstDelivery8'])
        self.assertEqual(v3.inputs(sample, manifest()['games'][0], context)[0][24:], [0, 0])

    def test_no_wave_reduces_to_v2_and_dev_cannot_change_fit(self):
        records, spec = rows(), manifest()
        baseline, candidate = v2.fit(records, spec), v3.fit(records, spec)
        for h in v1.HORIZONS:
            old, new = baseline['models'][str(h)], candidate['models'][str(h)]
            self.assertEqual(old['mean'], new['mean'][:24])
            self.assertEqual(old['intercept'], new['intercept'])
            self.assertEqual(old['coefficients'], [c[:24] for c in new['coefficients']])
        for record in records:
            if record['gameId'] != 'A':
                record['scan']['x'] += 300
                record['confirmedShots'] = [shot(turn=record['turn'])]
        self.assertEqual(candidate, v3.fit(records, spec))

    def test_context_future_and_round_isolation_and_explicit_fresh_split(self):
        sample, before = self.sample_context()
        data = [event(t, x=500) for t in (1, 9, 10)]
        data[0]['confirmedShots'] = [shot()]
        future = event(11, x=500)
        future['confirmedShots'] = [shot(turn=11, bullet=2)]
        later = dict(contexts(data + [future], manifest()))[('A', 1, 10, 2)]
        self.assertEqual(before, later)
        withheld = event(10, game='C', x=500)
        self.assertEqual(list(contexts([withheld], manifest())), [])
        self.assertEqual(len(list(contexts([withheld], manifest(), allowed_splits=('withheld',)))), 1)
        with self.assertRaises(ValueError):
            list(contexts(data, manifest(), allowed_splits=('invalid',)))

    def test_current_ties_stale_fallback_and_same_cli(self):
        model = v3.fit(rows(), manifest())
        sample, context = self.sample_context()
        for m in model['models'].values():
            m['intercept'] = [0] * 4
            m['coefficients'] = [[0] * 26 for _ in range(4)]
        self.assertEqual(v3.predict(model, sample, manifest()['games'][0], context)[1], 'current')
        sample['anchorTurn'] += 3
        self.assertFalse(v3.predict(model, sample, manifest()['games'][0], context)[2])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'manifest.json').write_text(json.dumps(manifest()))
            (root / 'events.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows()))
            cmd = [sys.executable, str(Path(__file__).with_name('containment_forecast_v3.py')),
                '--manifest', str(root / 'manifest.json'), '--events', str(root / 'events.jsonl')]
            result = subprocess.run(cmd + ['--output', str(root / 'fit')], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((root / 'fit' / 'report.json').read_text())
            self.assertIn('contextToolSha256', report)
            result = subprocess.run(cmd + ['--stage', 'screen', '--model', str(root / 'fit' / 'model.json'),
                '--model-sha256', report['modelSha256'], '--output', str(root / 'screen')], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
