import json
from pathlib import Path
import tempfile
import unittest

import manage


class EventStarterTests(unittest.TestCase):
    def test_new_event_is_draft_and_does_not_activate_or_overwrite_current(self):
        before = (manage.ROOT / 'current.json').read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            target = manage.scaffold('new-cup', 'a' * 40, Path(directory), 'project-for-test')
            config = json.loads((target / 'event.json').read_text())
            self.assertEqual('draft', config['status'])
            self.assertEqual('a' * 40, config['sourceCommit'])
            self.assertEqual('benchmarks/new-cup', config['paths']['evidence'])
            self.assertEqual('local-evidence/new-cup', config['paths']['privateArtifacts'])
            self.assertIsNone(config['authorization']['latestAuthorizedStage'])
            self.assertIsNone(config['execution']['practiceDrawSeed'])
            self.assertEqual([], config['entrantPolicy']['guests'])
            self.assertEqual('project-for-test', config['site']['projectId'])
        self.assertEqual(before, (manage.ROOT / 'current.json').read_bytes())

    def test_existing_event_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = manage.scaffold('saved-cup', 'b' * 40, Path(directory))
            before = (path / 'event.json').read_bytes()
            with self.assertRaises(FileExistsError):
                manage.scaffold('saved-cup', 'c' * 40, Path(directory))
            self.assertEqual(before, (path / 'event.json').read_bytes())

    def test_invalid_id_and_unpinned_source_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            for event_id, commit in [('../escape', 'a' * 40), ('new-cup', 'main'), ('A CUP', 'a' * 40)]:
                with self.subTest(event_id=event_id, commit=commit), self.assertRaises(ValueError):
                    manage.scaffold(event_id, commit, Path(directory))
            self.assertEqual([], list(Path(directory).iterdir()))

    def test_current_id_is_reserved_even_when_folder_name_differs(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError):
                manage.scaffold(manage.current_event()['id'], 'a' * 40, Path(directory))
            self.assertEqual([], list(Path(directory).iterdir()))

    def test_existing_evidence_or_media_paths_are_not_reused(self):
        for relative in ('benchmarks/new-cup', 'local-evidence/new-cup',
                         'tournament-sites/new-cup/dist'):
            with self.subTest(path=relative), tempfile.TemporaryDirectory() as directory:
                workspace = Path(directory)
                (workspace / relative).mkdir(parents=True)
                event_root = workspace / 'events'
                with self.assertRaises(FileExistsError):
                    manage.scaffold('new-cup', 'a' * 40, event_root, repository_root=workspace)
                self.assertFalse(event_root.exists())

    def test_site_url_comes_from_environment_and_preserves_base_path(self):
        config = {'site': {'baseUrlEnvironment': 'TEST_CUP_URL'}}
        environment = {'TEST_CUP_URL': 'https://example.test/cup'}
        self.assertEqual('https://example.test/cup/index.html', manage.site_url(config, environment=environment))
        self.assertEqual('https://example.test/cup/matches.html?stage=playoff-4',
                         manage.site_url(config, 'playoff-4', environment))
        with self.assertRaises(ValueError):
            manage.site_url(config, environment={})

    def test_site_url_rejects_credentials_or_non_base_url(self):
        config = {'site': {'baseUrlEnvironment': 'TEST_CUP_URL'}}
        for value in ('javascript:bad', 'https://user:secret@example.test',
                      'https://example.test?token=secret', 'https://example.test#fragment'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                manage.site_url(config, environment={'TEST_CUP_URL': value})
        with self.assertRaises(ValueError):
            manage.site_url(config, 'unknown', {'TEST_CUP_URL': 'https://example.test'})

    def test_current_snapshot_matches_published_score_evidence(self):
        config = manage.current_event()
        root = manage.ROOT.parent
        index = json.loads((root / config['paths']['officialResults']).read_text())
        self.assertEqual(index['sourceCommit'], config['sourceCommit'])
        self.assertEqual(index['validatedGames'], config['results']['validatedGames'])
        self.assertEqual(index['validatedRounds'], config['results']['competitiveRounds'])
        self.assertEqual(index['knownCompletedPhysicalRounds'], config['results']['knownPhysicalRounds'])
        actual = [(r['name'], r['losses']) for r in index['playoffStandings'] if not r['eliminated']]
        recorded = [(r['name'], r['losses']) for r in config['results']['remaining']]
        self.assertEqual(actual, recorded)
        self.assertEqual(index['stages'][-1]['stage'], config['results']['latestStage'])


if __name__ == '__main__':
    unittest.main()
