"""Synthetic nonlinear/gain/support/leakage proofs; no real fitting or retuning."""
import copy
import itertools
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import containment_tree_v5 as tree
from legal_scan_dataset import build_samples
from test_legal_scan_dataset import event, manifest


def synthetic_rows():
    return [event(t, game=g) for g in ('A', 'B', 'C', 'D') for t in range(1, 501)]


class GainTreeTests(unittest.TestCase):
    def test_first_split_floor_and_later_near_ties_are_distinct(self):
        self.assertFalse(tree.better_split(1e-12, None))
        self.assertTrue(tree.better_split(1.5e-12, None), 'do not double the initial floor')
        self.assertFalse(tree.better_split(1 + .5e-12, 1), 'within-tolerance tie keeps earlier feature')
        self.assertTrue(tree.better_split(1 + 2e-12, 1))

    def test_exact_quantiles_and_deduplication(self):
        self.assertEqual(tree.thresholds(list(reversed(range(101)))), list(range(3, 97, 3)))
        self.assertEqual(tree.thresholds([0] * 200 + [1] * 200), [0, 1])

    def test_nonlinear_three_way_context_depth_support_and_tie_order(self):
        xs, ys = [], []
        for bits in itertools.product((0, 1), repeat=3):
            for _ in range(200):
                xs.append(list(bits) + [0] * 23)
                ys.append([int(all(bits)), 0, 0])
        model = tree.build_tree(xs, ys)
        self.assertEqual(model['feature'], 0, 'equal reductions choose the first feature')
        def verify(node, depth=0):
            self.assertGreaterEqual(node['count'], 200)
            self.assertLessEqual(depth, 3)
            if 'feature' in node:
                verify(node['left'], depth + 1)
                verify(node['right'], depth + 1)
        verify(model)
        for bits in itertools.product((0, 1), repeat=3):
            self.assertEqual(tree.tree_gains(model, list(bits) + [0] * 23), [int(all(bits)), 0, 0])
        self.assertEqual(model, tree.build_tree(list(reversed(xs)), list(reversed(ys))))

    def test_zero_reduction_and_unsupported_horizon(self):
        xs = [[i] + [0] * 25 for i in range(400)]
        model = tree.build_tree(xs, [[0, -1, 0]] * 400)
        self.assertNotIn('feature', model)
        with self.assertRaises(ValueError):
            tree.build_tree(xs[:199], [[0, 0, 0]] * 199)

    def test_gain_target_counts_improvement_and_harm_not_just_hits(self):
        points = {'current': (0, 0), 'linear': (100, 0), 'circular': (0, 0), 'headon': (200, 0)}
        sample = {'label': {'x': 0, 'y': 0}}
        with patch.object(tree.v3.v1, 'baseline', side_effect=lambda s, g, name: points[name]):
            self.assertEqual(tree.gain_targets(sample, {}), [-1, 0, -1])
            sample['label']['x'] = 118  # exact radius-18 boundary is included
            self.assertEqual(tree.gain_targets(sample, {}), [1, 0, 0])
            sample['label']['x'] = 300
            self.assertEqual(tree.gain_targets(sample, {}), [0, 0, 0])

    def test_empirical_gain_margin_ties_and_stale_current_fallback(self):
        sample = list(build_samples([event(1), event(2)], manifest(), (8,), 8))[-1]
        context = {'earliestEligible': None}
        leaf = {'count': 200, 'means': [.01999999, .019, -.1]}
        model = {'trees': {'8': leaf}}
        self.assertEqual(tree.predict(model, sample, manifest()['games'][0], context)[1], 'current')
        leaf['means'] = [.02, .02, .02]
        self.assertEqual(tree.predict(model, sample, manifest()['games'][0], context)[1], 'linear')
        sample['anchorTurn'] += 3
        self.assertEqual(tree.predict(model, sample, manifest()['games'][0], context)[1], 'current')

    def test_dev_and_spent_game_values_cannot_change_training(self):
        records, spec = synthetic_rows(), manifest()
        before = tree.fit(records, spec)
        self.assertEqual(before['features'], list(tree.v3.FEATURES))
        changed = copy.deepcopy(records)
        for row in changed:
            if row['gameId'] != 'A':
                row['scan']['x'] += 300
                row['scan']['speed'] = -8
        self.assertEqual(before, tree.fit(changed, spec))

    def test_cli_freezes_model_and_scores_full_dev_masks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'manifest.json').write_text(json.dumps(manifest()))
            (root / 'events.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in synthetic_rows()))
            cmd = [sys.executable, str(Path(__file__).with_name('containment_tree_v5.py')), '--manifest',
                   str(root / 'manifest.json'), '--events', str(root / 'events.jsonl')]
            result = subprocess.run(cmd + ['--output', str(root / 'fit')], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((root / 'fit' / 'report.json').read_text())
            self.assertNotIn('cells', report)
            model_args = ['--model', str(root / 'fit' / 'model.json'), '--model-sha256', report['modelSha256']]
            result = subprocess.run(cmd + model_args + ['--stage', 'screen', '--output', str(root / 'screen')], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            screen = json.loads((root / 'screen' / 'report.json').read_text())
            self.assertEqual(set(screen['cells']), {'B'})
            cell = screen['cells']['B']['40']
            self.assertEqual((cell['anchors'], cell['observed']), (500, 460))
            self.assertLess(cell['eligible'], cell['observed'])
            self.assertEqual(sum(cell['selections'].values()), cell['observed'])
            self.assertIn('eligibleContainment', cell['methods']['scorer'])
            bad = subprocess.run(cmd + ['--stage', 'screen', '--model', str(root / 'fit' / 'model.json'),
                '--model-sha256', '0' * 64, '--output', str(root / 'bad')], capture_output=True)
            self.assertNotEqual(bad.returncode, 0)
            self.assertFalse((root / 'bad').exists())


if __name__ == '__main__':
    unittest.main()
