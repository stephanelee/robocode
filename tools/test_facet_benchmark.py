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

    def test_schema3_requires_successful_terminal_evidence(self):
        row=dict(round=1,gameId='new-cold',schema=3,shotsConfirmed=1,shotsAccepted=1,
                 shotsRequested=1,neighborBytes=1,snapshotStatus='success',
                 terminalObservation='success',terminalHash='success')
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'FACET_fixture.jsonl'
            path.write_text(json.dumps(row)+'\n')
            self.assertEqual(f.validate_metrics(path,1,3),[row])
            for key in ('snapshotStatus','terminalObservation','terminalHash'):
                path.write_text(json.dumps(dict(row,**{key:'failed'}))+'\n')
                with self.assertRaises(ValueError):f.validate_metrics(path,1,3)
            with self.assertRaises(ValueError):f.validate_metrics(path,1,2)

    def test_prepared_cohort_all_shot_integrity(self):
        import copy
        row=dict(round=1,gameId='fixture',schema=3,shotsConfirmed=2,shotsAccepted=2,
                 shotsRequested=2,neighborBytes=0,snapshotStatus='success',
                 terminalObservation='success',terminalHash='success',
                 counters={'gun.cohort.confirmed':2,'gun.cohort.common':1,'gun.cohort.analyticOnly':1},
                 observationsCountSumMinMax={'gun.cohort.headingErrorRadians':[2,.004,0,.004],
                                             'gun.cohort.requestAge':[2,3,1,2]})
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'FACET_fixture.jsonl'
            def validate(r):
                p.write_text(json.dumps(r)+'\n'); return f.validate_metrics(p,1,3,True)
            self.assertEqual(validate(row),[row])
            for key,val in (('gun.cohort.confirmed',1),('gun.cohort.common',2),('gun.cohort.missing',1),('gun.cohort.failure',1)):
                changed=copy.deepcopy(row); changed['counters'][key]=val
                with self.subTest(key=key),self.assertRaises(ValueError):validate(changed)
            for name,value in (('requestAge',[2,3,0,2]),('requestAge',[1,1,1,1]),
                               ('headingErrorRadians',[2,.02,0,.02]),('headingErrorRadians',[2,99,0,.001])):
                changed=copy.deepcopy(row);changed['observationsCountSumMinMax']['gun.cohort.'+name]=value
                with self.subTest(name=name,value=value),self.assertRaises(ValueError):validate(changed)
            empty=copy.deepcopy(row);empty.update(shotsConfirmed=0,counters={},observationsCountSumMinMax={})
            self.assertEqual(validate(empty),[empty])

    def test_complete_cold_game_cohorts(self):
        games = [dict(opponent=o, scoreShare=i/10, facet=dict(firstPlaces=i), finalTelemetry={})
                 for o in f.COHORTS for i in range(1,6)]
        from unittest.mock import patch
        with patch.object(f, 'metric_summary', return_value=dict(botTurnDenominator=100,reportedSkippedTurns=0,wallCollisions=0,botCollisions=0,failuresAndFallbacks={})):
            result = f.summarize(games)
        self.assertAlmostEqual(result['Retcon']['equalGameMeanScoreShare'], .3)
        self.assertEqual(result['Retcon']['officialFirstPlaces'], 15)
        with self.assertRaises(ValueError): f.summarize(games[:-1])

    def test_diamond_only_roster_keeps_five_cold_games(self):
        from unittest.mock import patch
        games = [dict(opponent='diamond-port', scoreShare=.5, facet=dict(firstPlaces=10), finalTelemetry={}) for _ in range(5)]
        with patch.object(f, 'metric_summary', return_value=dict(botTurnDenominator=100, reportedSkippedTurns=0, wallCollisions=0, botCollisions=0, failuresAndFallbacks={})):
            result = f.summarize(games, ['diamond-port'])
            self.assertEqual(list(result), ['diamond-port'])
            self.assertEqual(result['diamond-port']['rounds'], 100)
            with self.assertRaises(ValueError): f.summarize(games[:-1], ['diamond-port'])

    def test_official_outcome_survives_invalid_completion(self):
        result = dict(numberOfRounds=20, results=[dict(name='Facet', version='0.1.0', totalScore=60, firstPlaces=12), dict(name='Fixture', version='1', totalScore=40)])
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'battle.json'; p.write_text(json.dumps(result))
            record = dict(rounds=20, status='running')
            report = dict(status='running', games=[record])
            f.retain_official_result(record, p, 'Fixture', '1')
            record['stage'] = 'facet-completion'
            f.invalidate_active_game(report, ValueError('missing round12'))
            self.assertEqual(record['official'], result)
            self.assertEqual(record['officialRounds'], 20)
            self.assertEqual(record['scoreShare'], .6)
            self.assertEqual(record['status'], 'invalid')
            self.assertEqual(record['stage'], 'facet-completion')
            self.assertEqual(report['status'], 'invalid')
            with self.assertRaises(ValueError): f.retain_official_result(dict(rounds=20), p, 'Wrong', '1')

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

    def test_invalid_screen_never_grants_new_budget(self):
        with self.assertRaises(ValueError):f.authorize_registration({})
        original=dict(independentReviewEvidence='signed-source-review',premeasurementApproved=True)
        f.authorize_registration(original)
        proposed=dict(original,freshAcquisitionAfterInvalid=True,userAuthorizedFreshBudget=False)
        with self.assertRaises(ValueError):f.authorize_registration(proposed)
        f.authorize_registration(dict(proposed,userAuthorizedFreshBudget=True))

    def test_environment_drops_other_state_and_server_settings(self):
        from unittest.mock import patch
        with patch.dict(f.os.environ, {'FACET_SEED':'1','SERVER_URL':'ws://localhost:7654',
                                     'RETCON_STATS_DIR':'private','AISURVIVOR_MODEL_PATH':'production',
                                     'JAVA_TOOL_OPTIONS':'-Dserver.url=ws://localhost:7654','PATH':'allowed'},clear=True):
            self.assertEqual(f.clean_env(), {'PATH':'allowed'})


if __name__ == '__main__': unittest.main()
