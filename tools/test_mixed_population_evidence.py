"""Verify frozen research bytes/results; never refit or retune."""
import gzip
import hashlib
import json
from pathlib import Path
import unittest
import containment_forecast_v3 as scorer

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'benchmarks' / 'ai-mixed-population-v4'


def read(path):
    return json.loads(path.read_bytes())


class MixedPopulationEvidenceTests(unittest.TestCase):
    def test_original_artifact_hashes_population_and_no_confirmation(self):
        saved = read(EVIDENCE / 'preservation.json')
        for name, digest in saved['files'].items():
            self.assertEqual(digest, hashlib.sha256((EVIDENCE / name).read_bytes()).hexdigest())
        self.assertEqual((0, False), (saved['newRounds'], saved['freshConfirmationRun']))
        model = read(EVIDENCE / 'model.json')
        self.assertEqual('10599435a4fa41480efa011b6cc4a0ff63ccc61a8f3c280f31cedff0c7ec6f4b',
                         hashlib.sha256((EVIDENCE / 'model.json').read_bytes()).hexdigest())
        self.assertEqual([13487, 13272, 13056, 12840, 12625],
                         [model['models'][str(h)]['count'] for h in scorer.v1.HORIZONS])
        report = read(EVIDENCE / 'development-report.json')
        self.assertEqual('reject', report['decision']['status'])
        self.assertEqual(['primaryGain'], [k for k, v in report['decision']['checks'].items() if not v])
        self.assertEqual(hashlib.sha256((EVIDENCE / 'manifest.json').read_bytes()).hexdigest(), model['manifestSha256'])

    def test_reproduce_development_only_with_frozen_weights(self):
        raw = gzip.decompress((ROOT / 'benchmarks' / 'ai-motion-v1' / 'events.jsonl.gz').read_bytes())
        model = read(EVIDENCE / 'model.json')
        self.assertEqual(model['eventsSha256'], hashlib.sha256(raw).hexdigest())
        manifest = read(EVIDENCE / 'manifest.json')
        dev = {g['gameId'] for g in manifest['games'] if g['split'] == 'dev'}
        records = [row for line in raw.splitlines() if (row := json.loads(line))['gameId'] in dev]
        cells = scorer.screen(records, manifest, model, ('dev',))
        report = read(EVIDENCE / 'development-report.json')
        self.assertEqual(report['decision'], scorer.decision(cells))
        self.assertEqual(dev, set(cells))
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
