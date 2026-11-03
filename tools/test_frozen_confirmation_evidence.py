"""Verify preserved confirmation bytes and reproduce its fixed decision, without games."""
import hashlib
import json
from pathlib import Path
import unittest
from frozen_duel_confirmation import decision

ROOT = Path(__file__).resolve().parents[1] / 'benchmarks' / 'frozen-duel-confirmation'


class FrozenEvidenceTests(unittest.TestCase):
    def test_original_evidence_hashes_and_registered_decision(self):
        manifest = json.loads((ROOT / 'files.sha256.json').read_bytes())
        self.assertEqual(1, manifest['attempt'])
        self.assertEqual(57, len(manifest['files']))
        for name, digest in manifest['files'].items():
            self.assertEqual(digest, hashlib.sha256((ROOT / name).read_bytes()).hexdigest())
        report = json.loads((ROOT / 'confirmation.json').read_bytes())
        self.assertEqual(report['decision'], decision(report['games']))
        self.assertEqual(('confirmed', 107, 6), (report['decision']['status'], report['decision']['wins'], report['decision']['gamesAboveHalf']))
        self.assertLess(report['decision']['descriptiveTLower95'], .5)
        self.assertLess(report['decision']['descriptiveWinWilson95']['lower'], .5)

    def test_all_ten_official_games_and_two_hundred_valid_fixed_rounds(self):
        report = json.loads((ROOT / 'confirmation.json').read_bytes())
        count = 0
        for game in report['games']:
            directory = ROOT / 'raw' / game['gameId']
            official = json.loads((directory / 'battle.json').read_bytes())
            self.assertEqual(20, official['numberOfRounds'])
            self.assertEqual(official['results'], game['battle']['results'])
            rows = [json.loads(line) for line in (directory / 'metrics.jsonl').read_bytes().splitlines()]
            self.assertEqual(list(range(1, 21)), [row['round'] for row in rows])
            self.assertTrue(all(row['mode'] == 'fixed' and row['action'] == 0 and row['seed'] == game['seed']
                                and row['totalRounds'] == 0 and not row['checkpointLoaded']
                                and row['skippedTurns'] == 0 for row in rows))
            count += len(rows)
        self.assertEqual(200, count)


if __name__ == '__main__':
    unittest.main()
