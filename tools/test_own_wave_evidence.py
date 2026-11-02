"""Preservation checks: never refit or retune; reproduce frozen results only."""
import gzip
import hashlib
import json
from pathlib import Path
import unittest
import containment_forecast_v3 as scorer

ROOT = Path(__file__).resolve().parents[1] / 'benchmarks' / 'ai-own-wave-v3'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


class OwnWaveEvidenceTests(unittest.TestCase):
    def test_frozen_report_reproduces_without_fitting(self):
        base = ROOT / 'confirmation'
        records = [json.loads(line) for line in gzip.decompress((base / 'events.jsonl.gz').read_bytes()).splitlines()]
        model = read(ROOT / 'model.json')
        report = read(ROOT / 'fresh-report.json')
        cells = scorer.screen(records, read(base / 'manifest.json'), model, ('withheld', 'transfer'))
        self.assertEqual(report['decision'], scorer.confirmation_decision(cells, read(base / 'manifest.json')))
        for game, horizons in cells.items():
            for horizon, cell in horizons.items():
                saved = report['cells'][game][horizon]
                for key in ('anchors', 'observed', 'eligible', 'selections', 'coverage'):
                    self.assertEqual(saved[key], cell[key])
                for method, values in cell['methods'].items():
                    for key, value in values.items():
                        if isinstance(value, int):
                            self.assertEqual(saved['methods'][method][key], value)
                        else:
                            self.assertAlmostEqual(saved['methods'][method][key], value, places=8)

    def test_complete_capture_original_bytes_and_per_game_boundaries(self):
        base = ROOT / 'confirmation'
        index = read(base / 'preservation.json')
        packed = (base / 'events.jsonl.gz').read_bytes()
        raw = gzip.decompress(packed)
        self.assertEqual(hashlib.sha256(packed).hexdigest(), index['gzipSHA256'])
        self.assertEqual(hashlib.sha256(raw).hexdigest(), index['uncompressedSHA256'])
        self.assertEqual(8243, len(raw.splitlines()))
        self.assertEqual(4953284, len(raw))
        capture = read(base / 'acquisition.json')
        self.assertEqual(('complete', 18, 6), (capture['status'], capture['validatedRounds'], capture['validatedGames']))
        self.assertEqual(6, len(index['games']))
        end = 0
        for game in index['games']:
            self.assertEqual(end, game['offset'])
            end += game['bytes']
            segment = raw[game['offset']:end]
            self.assertEqual(hashlib.sha256(segment).hexdigest(), game['sha256'])
            rows = [json.loads(line) for line in segment.splitlines()]
            self.assertEqual(game['records'], len(rows))
            self.assertEqual({game['gameId']}, {row['gameId'] for row in rows})
            status = read(base / 'games' / game['gameId'] / 'status.json')
            self.assertEqual(('complete', 'legal-scan-v1.1-closed-round'), (status['status'], status['collectorVersion']))
            self.assertEqual({1, 2, 3}, {row['round'] for row in rows})
            self.assertEqual(3, read(base / 'games' / game['gameId'] / 'battle.json')['numberOfRounds'])
        self.assertEqual(len(raw), end)
        result = read(ROOT / 'fresh-report.json')
        self.assertEqual(index['uncompressedSHA256'], result['actualEventsSha256'])
        self.assertEqual('reject', result['decision']['status'])
        self.assertEqual(['transfer'], [key for key, value in result['decision']['checks'].items() if not value])
        self.assertEqual(hashlib.sha256((ROOT / 'model.json').read_bytes()).hexdigest(), result['modelSha256'])

    def test_failed_attempt_is_preserved_as_invalid_and_unscored(self):
        base = ROOT / 'failed-attempt-41535141065'
        index = read(base / 'preservation.json')
        self.assertEqual((6, 3, False), (index['physicalRounds'], index['validatedRounds'], index['confirmationEvaluated']))
        count = 0
        for game in index['games']:
            packed = (base / 'games' / game['gameId'] / 'events.jsonl.gz').read_bytes()
            raw = gzip.decompress(packed)
            self.assertEqual(hashlib.sha256(packed).hexdigest(), game['gzipSHA256'])
            self.assertEqual(hashlib.sha256(raw).hexdigest(), game['eventsSHA256'])
            count += len(raw.splitlines())
        self.assertEqual(3136, count)
        failed = read(base / 'games' / 'w3-target-01' / 'status.json')
        self.assertEqual(('failed', 'Invalid or repeated capture turn'), (failed['status'], failed['error']))
        self.assertEqual(3, len(failed['rounds']))
        self.assertTrue(all(row['complete'] for row in failed['rounds']))


if __name__ == '__main__':
    unittest.main()
