"""Independent evidence-ingestion fixtures; no battles, models or opponent files."""
import json
from pathlib import Path
import tempfile
import unittest
import facet_report as report
from types import SimpleNamespace


def metric(i):
    return dict(round=i,gameId="own-fixture",schema=3,snapshotStatus="success",terminalObservation="success",terminalHash="success",shotsConfirmed=i,shotsAccepted=i,shotsRequested=i,neighborBytes=0,counters={"shot.hit":i,"turn.decisions":i},observationsCountSumMinMax={"range":[1.0,400.0,400.0,400.0]},timingLog2Nanos={},gcCount=0,gcMillis=0,surfTruncations=0,geometryFallbacks=0,unmatchedConfirmations=0)


class Evidence(unittest.TestCase):
    def test_invalid_finished_game_is_not_dropped_or_repaired(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); game=root/'diamond-port-g01';game.mkdir()
            original={'status':'invalid','scheduledGames':5,'scheduledRounds':100,'games':[{'opponent':'diamond-port','game':1,'seed':49667,'status':'running'}]}
            path=root/'results.json';path.write_text(json.dumps(original));before=path.read_bytes()
            official={'numberOfRounds':20,'results':[{'name':'Facet','totalScore':60,'firstPlaces':12},{'name':'Public fixture','totalScore':40}]}
            (game/'battle.json').write_text(json.dumps(official))
            rows=[metric(i) for i in range(1,21) if i!=12]
            (game/'FACET_metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            data,games=report.load_evidence(path)
            self.assertEqual(len(games),1)
            self.assertEqual(data['summary']['diamond-port']['rounds'],20)
            self.assertEqual(data['summary']['diamond-port']['officialFirstPlaces'],12)
            self.assertEqual(data['summary']['diamond-port']['equalGameMeanScoreShare'],.6)
            self.assertEqual(data['summary']['diamond-port']['registeredGateAssessment']['decision'],'invalid-or-incomplete-descriptive-only')
            self.assertEqual(games[0]['missingMetricRounds'],[12])
            gap=next(x for x in report.learning_intervals(games) if x['throughRound']==13)
            self.assertEqual((gap['fromRound'],gap['throughRound'],gap['shots']),(12,13,2))
            self.assertNotIn('round',gap)
            self.assertEqual(path.read_bytes(),before)
            # A report/sidecar disagreement must never silently choose favorable bytes.
            original['games'][0]['official']=dict(official,numberOfRounds=19);path.write_text(json.dumps(original))
            with self.assertRaises(ValueError):report.load_evidence(path)

    def test_failed_truncated_and_nonfinite_metrics_preserve_official_game(self):
        bad_values = [json.dumps(dict(round=1, schema=3, snapshotStatus='failed')), '{"round":1', json.dumps(dict(metric(1), neighborBytes=float('nan'))), json.dumps(dict(metric(1), shotsConfirmed=-1)), json.dumps(dict(metric(1), gcCount=float('inf'))), json.dumps(dict(metric(1), counters=[])), json.dumps(dict(metric(1), gcCount=[])), json.dumps(dict(metric(1), observationsCountSumMinMax={'range':[1,2]})), json.dumps(dict(metric(1), timingLog2Nanos={'decision':[1,-1]})), '{}']
        for bad in bad_values:
            with self.subTest(bad=bad), tempfile.TemporaryDirectory() as d:
                root=Path(d); game=root/'diamond-port-g01';game.mkdir()
                path=root/'results.json';path.write_text(json.dumps(dict(status='invalid',scheduledGames=5,scheduledRounds=100,games=[dict(opponent='diamond-port',game=1,seed=49667,status='invalid')])))
                (game/'battle.json').write_text(json.dumps(dict(numberOfRounds=20,results=[dict(name='Facet',totalScore=60,firstPlaces=12),dict(name='Public fixture',totalScore=40)])))
                (game/'FACET_metrics.jsonl').write_text(bad+'\n')
                data,games=report.load_evidence(path)
                self.assertEqual(data['summary']['diamond-port']['officialFirstPlaces'],12)
                self.assertEqual(games[0]['usableMetricRows'],[])
                self.assertTrue(games[0]['metricErrors'])
                self.assertEqual(report.learning_intervals(games),[])
                out=root/'report'
                report.main(SimpleNamespace(results=path,output=out))
                text=(out/'results.md').read_text()
                self.assertIn('official completed rounds **20/100**',text)
                self.assertIn('| diamond-port | unavailable',text)

    def test_integral_floating_observation_counts_are_native_schema(self):
        report.validate_metric_shape(metric(1))
        row=metric(1);row['observationsCountSumMinMax']={'range':[1.5,400,400,400]}
        with self.assertRaises(ValueError):report.validate_metric_shape(row)

    def test_retained_acquisition_replays_all_139_numeric_rows(self):
        path=Path(__file__).resolve().parents[1]/'benchmarks/facet-registered-300/attempt-1/results.json'
        data,games=report.load_evidence(path)
        self.assertEqual(data['status'],'invalid')
        self.assertEqual(sum(len(g['usableMetricRows']) for g in games),139)
        self.assertEqual(sum(len(g['metricErrors']) for g in games),0)
        for opponent,shots,hits in [('AISurvivor',3074,764),('Retcon',1907,186)]:
            metrics=data['summary'][opponent]['metricsByGame']
            self.assertEqual(sum(r['confirmed'] for r in metrics),shots)
            self.assertEqual(sum(r['physicalHits'] for r in metrics),hits)

    def test_counter_reset_is_not_a_learning_curve(self):
        game=dict(opponent='diamond-port',game=1,metricRows=[dict(round=1,shotsConfirmed=5,counters={}),dict(round=2,shotsConfirmed=1,counters={})])
        with self.assertRaises(ValueError):report.learning_intervals([game])


if __name__=='__main__':unittest.main()
