"""Preserve the fixed tree result; no fitting, retuning or new test evaluation."""
import gzip
import hashlib
import json
import math
from pathlib import Path
import unittest
import containment_tree_v5 as tree

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'benchmarks' / 'ai-nonlinear-gain-v5'


def read(path):
    return json.loads(path.read_bytes())


class NonlinearGainEvidenceTests(unittest.TestCase):
    def test_original_bytes_and_fixed_tree_support(self):
        record = read(EVIDENCE / 'preservation.json')
        for name, digest in record['files'].items():
            self.assertEqual(digest, hashlib.sha256((EVIDENCE / name).read_bytes()).hexdigest())
        self.assertEqual((0, False), (record['newRounds'], record['freshConfirmationRun']))
        self.assertEqual('54e5da030e334a3760a18d38310e7357ffb0a90693fef6d42fe07a6325383d16',
                         hashlib.sha256((EVIDENCE / 'model.json').read_bytes()).hexdigest())
        model = read(EVIDENCE / 'model.json')
        self.assertEqual(tree.PARAMETERS, model['parameters'])
        self.assertEqual(list(tree.v3.FEATURES), model['features'])

        def check(node, depth=0):
            self.assertLessEqual(depth, 3)
            self.assertGreaterEqual(node['count'], 200)
            self.assertEqual(3, len(node['means']))
            self.assertTrue(all(math.isfinite(value) and -1 <= value <= 1 for value in node['means']))
            if 'feature' not in node:
                return 1
            self.assertIn(node['feature'], range(26))
            self.assertTrue(math.isfinite(node['threshold']))
            self.assertGreater(node['reduction'], 1e-12)
            self.assertEqual(node['count'], node['left']['count'] + node['right']['count'])
            return check(node['left'], depth + 1) + check(node['right'], depth + 1)

        self.assertEqual([13487, 13272, 13056, 12840, 12625],
                         [model['trees'][str(h)]['count'] for h in tree.v3.v1.HORIZONS])
        for node in model['trees'].values():
            self.assertLessEqual(check(node), 8)

    def test_frozen_development_counts_and_rejection_reproduce(self):
        manifest = read(EVIDENCE / 'manifest.json')
        model = read(EVIDENCE / 'model.json')
        report = read(EVIDENCE / 'development-report.json')
        raw = gzip.decompress((ROOT / 'benchmarks' / 'ai-motion-v1' / 'events.jsonl.gz').read_bytes())
        self.assertEqual(model['eventsSha256'], hashlib.sha256(raw).hexdigest())
        dev = {g['gameId'] for g in manifest['games'] if g['split'] == 'dev'}
        records = [row for line in raw.splitlines() if (row := json.loads(line))['gameId'] in dev]
        cells = tree.screen(records, manifest, model)
        self.assertEqual(dev, set(cells))
        self.assertEqual(report['decision'], tree.v3.decision(cells))
        self.assertEqual('reject', report['decision']['status'])
        self.assertEqual(['primaryGain'], [k for k, value in report['decision']['checks'].items() if not value])
        for game, horizons in cells.items():
            for horizon, cell in horizons.items():
                saved = report['cells'][game][horizon]
                for key in ('anchors', 'observed', 'eligible', 'selections', 'coverage'):
                    self.assertEqual(saved[key], cell[key])
                for method, values in cell['methods'].items():
                    for key, value in values.items():
                        self.assertAlmostEqual(saved['methods'][method][key], value, places=8)


if __name__ == '__main__':
    unittest.main()
