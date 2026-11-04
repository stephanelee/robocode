"""Telemetry acceptance and game-cohort checks; no battles or opponent reads."""
import json
from pathlib import Path
import tempfile
import unittest
import facet_benchmark as f


class Acceptance(unittest.TestCase):
    def test_round_identity_and_physical_transaction_invariants(self):
        row = dict(round=1, gameId='cold-1', schema=2, shotsConfirmed=2, shotsAccepted=3,
                   shotsRequested=3, neighborBytes=42)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'FACET_fixture.jsonl'
            p.write_text(json.dumps(row)+'\n')
            self.assertEqual(f.validate_metrics(p, 1), [row])
            for change in ({'round':2}, {'schema':1}, {'shotsConfirmed':4}, {'neighborBytes':134217729}):
                with self.subTest(change=change):
                    p.write_text(json.dumps(dict(row, **change))+'\n')
                    with self.assertRaises(ValueError): f.validate_metrics(p, 1)
            p.write_text(json.dumps(dict(row, invalid=float('nan')))+'\n')
            with self.assertRaises(ValueError): f.validate_metrics(p, 1)
            p.write_text(json.dumps(row)+'\n'+json.dumps(row)+'\n')
            with self.assertRaises(ValueError): f.validate_metrics(p, 2)

    def test_complete_cold_game_cohorts(self):
        games = [dict(opponent=o, scoreShare=i/10, facet=dict(firstPlaces=i), finalTelemetry={})
                 for o in f.COHORTS for i in range(1,6)]
        from unittest.mock import patch
        with patch.object(f, 'metric_summary', return_value=dict(botTurnDenominator=100,reportedSkippedTurns=0,wallCollisions=0,botCollisions=0,failuresAndFallbacks={})):
            result = f.summarize(games)
        self.assertAlmostEqual(result['Retcon']['equalGameMeanScoreShare'], .3)
        self.assertEqual(result['Retcon']['officialFirstPlaces'], 15)
        with self.assertRaises(ValueError): f.summarize(games[:-1])

    def test_freeze_rejects_changed_artifacts_and_source(self):
        from unittest.mock import patch
        import hashlib
        with tempfile.TemporaryDirectory() as d:
            root = Path(d).resolve()
            (root/'experiments/Facet').mkdir(parents=True)
            (root/'experiments/Facet/input.java').write_text('class Fixture {}')
            digest = hashlib.sha256((root/'experiments/Facet/input.java').read_bytes()).hexdigest()
            import facet_evidence
            frozen = dict(fileSHA256={'experiments/Facet/input.java': digest},
                          facetSourceFiles={'input.java': digest},
                          facetSourceSHA256=facet_evidence.tree_digest({'input.java': digest}))
            with patch.object(f, 'ROOT', root):
                self.assertEqual(f.validate_freeze(frozen), {'input.java': digest})
                (root/'experiments/Facet/input.java').write_text('class Changed {}')
                with self.assertRaises(ValueError): f.validate_freeze(frozen)

    def test_failed_black_box_preserves_log_without_exposing_its_content(self):
        with tempfile.TemporaryDirectory() as d:
            log = Path(d)/'opaque.log'
            with self.assertRaises(RuntimeError) as raised:
                f.run_black_box([f.os.sys.executable, '-c', "print('opaque-sentinel'); raise SystemExit(7)"], f.os.environ.copy(), log)
            self.assertNotIn('opaque-sentinel', str(raised.exception))
            self.assertIn('exit7', str(raised.exception))
            self.assertIn('opaque-sentinel', log.read_text())

    def test_environment_drops_other_state_and_server_settings(self):
        from unittest.mock import patch
        with patch.dict(f.os.environ, {'FACET_SEED':'1','SERVER_URL':'ws://localhost:7654',
                                     'RETCON_STATS_DIR':'private','AISURVIVOR_MODEL_PATH':'production',
                                     'JAVA_TOOL_OPTIONS':'-Dserver.url=ws://localhost:7654','PATH':'allowed'},clear=True):
            self.assertEqual(f.clean_env(), {'PATH':'allowed'})


if __name__ == '__main__': unittest.main()
