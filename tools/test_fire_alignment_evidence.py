import hashlib
import json
from pathlib import Path
import unittest

import benchmark
from fire_alignment_screen import screen_decision

ROOT = Path(__file__).resolve().parents[1] / 'benchmarks/fire-alignment-screen'


class FireEvidenceTests(unittest.TestCase):
    def test_raw_hashes_and_exact_rejection(self):
        files = json.loads((ROOT / 'files.sha256.json').read_text())
        self.assertEqual(len(files), 52)
        for name, expected in files.items():
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), expected, name)
        screen = json.loads((ROOT / 'screen.json').read_text())
        comparison = json.loads((ROOT / 'retcon/comparison.json').read_text())
        self.assertEqual(screen_decision(comparison, screen['samples']), screen['decision'])
        self.assertEqual(screen['decision']['status'], 'reject')
        self.assertEqual(comparison['candidateSource'], '1a487c9b78976329d1d532c586fb5eb629db3f74')
        self.assertEqual(comparison['bundleSHA256']['candidate']['AISurvivor.jar'],
                         'f32dffc3096123489df64abc2e897db1ed85c11ce583e1d39bab424e2a9289b1')

    def test_every_registered_official_game_and_round_record(self):
        results = list(ROOT.rglob('battle.json'))
        self.assertEqual(len(results), 12)
        rounds = 0
        for path in results:
            official = json.loads(path.read_text())
            rows = [json.loads(line) for line in path.with_name('metrics.jsonl').read_text().splitlines()]
            count = official['numberOfRounds']
            self.assertIn(count, (3, 5))
            self.assertEqual(len(official['results']), 2)
            benchmark.validate_metrics(rows, count, 'fixed-minrisk', rows[0]['seed'], 20)
            self.assertTrue(all(row['skippedTurns'] == 0 for row in rows))
            rounds += len(rows)
        self.assertEqual(rounds, 48)


if __name__ == '__main__':
    unittest.main()
