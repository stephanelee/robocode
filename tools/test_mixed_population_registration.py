"""Verify the immutable population-only change; no fitting or outcome evaluation."""
import collections
import copy
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]
REGISTRATION = '044e4e28265dc175213469a8b87a941c82f1cf1c'
MANIFEST = 'tools/fixtures/mixed-population-v4/manifest.json'
FRESH = 'tools/fixtures/mixed-population-v4/fresh-games.json'
ORIGINAL = 'benchmarks/ai-motion-v1/manifest.json'
RAW_SHA = 'd2cb8e055659485d89648eec6cd3aebe8cbfce8dfb4dd19f83e7ba3efa1d11e2'


def blob(path, commit='HEAD'):
    # Hash canonical Git bytes rather than newline-converted Windows worktrees.
    return subprocess.check_output(['git', 'show', commit + ':' + path], cwd=ROOT)


def document(path, commit='HEAD'):
    return json.loads(blob(path, commit))


class MixedPopulationRegistrationTests(unittest.TestCase):
    def test_only_original_three_sample_roles_and_dataset_id_change(self):
        original = document(ORIGINAL)
        mixed = document(MANIFEST)
        expected = copy.deepcopy(original)
        expected['datasetId'] = 'ais-mixed-population-v4'
        changed = []
        for game in expected['games']:
            if game['gameId'] in ('x-target-01', 'x-spinbot-01', 'x-walls-01'):
                self.assertEqual(game['split'], 'transfer')
                game['split'] = 'train'
                changed.append(game['opponent'])
        self.assertEqual(set(changed), {'Target', 'SpinBot', 'Walls'})
        self.assertEqual(mixed, expected)
        self.assertEqual(collections.Counter(g['split'] for g in mixed['games']),
                         {'train': 9, 'dev': 3, 'withheld': 3})
        self.assertTrue(all(g['opponent'] == 'Retcon' for g in mixed['games'] if g['split'] in ('dev', 'withheld')))
        self.assertEqual(blob(MANIFEST), blob(MANIFEST, REGISTRATION))

    def test_original_corpus_and_capture_provenance_remain_exact(self):
        original = document(ORIGINAL)
        self.assertEqual(hashlib.sha256(blob(ORIGINAL)).hexdigest(),
                         '1f533fc0933af8462f94eb9988a0818c6c3b18c2b7c2e981d8f1a1b89dbf8a19')
        raw = gzip.decompress(blob('benchmarks/ai-motion-v1/events.jsonl.gz'))
        self.assertEqual(hashlib.sha256(raw).hexdigest(), RAW_SHA)
        self.assertEqual(len(raw), 16215891)
        self.assertEqual(document(MANIFEST)['provenance'], original['provenance'])
        self.assertEqual(original['provenance']['collectorVersion'], 'legal-scan-v1')
        self.assertEqual(original['provenance']['botJarSha256'],
                         '1b99fa0ac303e5e5f89ed91bb3803841ded54910628f335fe1aeaeba65e34c9a')

    def test_original_scorer_and_all_feature_dependencies_are_unchanged(self):
        frozen = document('benchmarks/ai-own-wave-v3/model.json')
        for key, path in (
                ('toolSha256', 'tools/containment_forecast_v3.py'),
                ('dependencySha256', 'tools/motion_forecast.py'),
                ('contextToolSha256', 'tools/own_wave_context_diagnostic.py'),
                ('datasetToolSha256', 'tools/legal_scan_dataset.py')):
            self.assertEqual(hashlib.sha256(blob(path)).hexdigest(), frozen[key], path)

    def test_fresh_layout_is_registered_disjoint_and_exact_budget(self):
        fresh = document(FRESH)
        self.assertEqual(blob(FRESH), blob(FRESH, REGISTRATION))
        self.assertEqual((fresh['roundsPerGame'], fresh['history'], fresh['horizons']),
                         (3, 8, [8, 16, 24, 32, 40]))
        games = fresh['games']
        self.assertEqual([g['opponent'] for g in games], ['Retcon', 'Target', 'Retcon', 'SpinBot', 'Retcon', 'Walls'])
        self.assertEqual([g['gameId'] for g in games], ['mf4-retcon-01', 'mf4-target-01', 'mf4-retcon-02',
                                                       'mf4-spinbot-01', 'mf4-retcon-03', 'mf4-walls-01'])
        self.assertEqual([g['aisSeed'] for g in games], list(range(2026110601, 2026110607)))
        self.assertEqual(collections.Counter(g['split'] for g in games), {'withheld': 3, 'transfer': 3})
        prior = document(ORIGINAL)['games']
        for path in ('tools/fixtures/own-wave-v3/games.json', 'tools/fixtures/own-wave-v3/retry-games.json'):
            prior += document(path)['games']
        self.assertFalse({g['gameId'] for g in games} & {g['gameId'] for g in prior})
        self.assertFalse({g['acquisitionId'] for g in games} & {g['acquisitionId'] for g in prior})
        original_pins = {g['opponent']: g['opponentPin'] for g in document(ORIGINAL)['games']}
        self.assertTrue(all((g['arenaWidth'], g['arenaHeight']) == (800, 600) and
                            g['opponentPin'] == original_pins[g['opponent']] for g in games))


if __name__ == '__main__':
    unittest.main()
