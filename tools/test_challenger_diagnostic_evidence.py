"""Reproduce fixed diagnostic observations without commands, games or tuning."""
import gzip,hashlib,json,math,unittest
from pathlib import Path
import benchmark,compare,challenger_diagnostic_experiment as capture
import challenger_diagnostic_analysis as analysis
ROOT=Path(__file__).resolve().parents[1]/'benchmarks/challenger-diagnostic-v5'
class DiagnosticEvidenceTests(unittest.TestCase):
    def test_original_bytes_lossless_trace_and_corrected_offline_analysis(self):
        for name,digest in json.loads((ROOT/'files.sha256.json').read_bytes())['files'].items():
            self.assertEqual(digest,hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),name)
        report=json.loads((ROOT/'experiment.json').read_bytes());self.assertIsNone(report['strengthDecision'])
        game=report['games'][0];raw=gzip.decompress((ROOT/'raw'/game['gameId']/'diagnostic.jsonl.gz').read_bytes())
        self.assertEqual(game['diagnostic']['sha256'],hashlib.sha256(raw).hexdigest());self.assertEqual(3876266,len(raw))
        records=[json.loads(line) for line in raw.splitlines()];self.assertEqual(12891,len(records))
        expected=json.loads((ROOT/'analysis.json').read_bytes());actual=analysis.analyze(records)
        def same(a,b):
            if isinstance(a,dict):
                self.assertEqual(set(a),set(b))
                for k in a:same(a[k],b[k])
            elif isinstance(a,list):
                self.assertEqual(len(a),len(b))
                for x,y in zip(a,b):same(x,y)
            elif isinstance(a,float):self.assertTrue(math.isclose(a,b,abs_tol=1e-9,rel_tol=1e-12),(a,b))
            else:self.assertEqual(a,b)
        same(expected,actual)
        self.assertEqual((237,231,13),(actual['confirmedShotsRecorded'],actual['joinedPassages'],actual['actualHitEvents']))
        m=actual['chosenMovementNextState'];self.assertEqual((3624,5,0),(m['joined'],m['masks']['disabled'],m['transitionsAboveTolerance']))
        self.assertEqual(0,actual['decisionToCreationGunPhaseDegrees']['mean'])
        self.assertEqual((ROOT/'source/reviewed-workflow.yml').read_bytes().replace(b'\r\n',b'\n'),(ROOT/'source/executed-workflow.yml').read_bytes())
    def test_all_five_cold_rounds_endpoint_and_actual_hit_coverage(self):
        report=json.loads((ROOT/'experiment.json').read_bytes());self.assertEqual(1,len(report['games']));g=report['games'][0]
        d=ROOT/'raw'/g['gameId'];rows=[json.loads(line) for line in (d/'metrics.jsonl').read_bytes().splitlines()]
        capture.check_rows(rows,5,20263501);self.assertEqual(g['metrics'],benchmark.summarize_metrics(rows))
        self.assertEqual(g['endpoint'],capture.endpoint(d/'model.json',5,20263501));self.assertEqual(231,g['endpoint']['state']['contextMemorySize'])
        status=json.loads((d/'diagnostic.jsonl.status.json').read_bytes());self.assertEqual((5,0,False),(status['roundsCompleted'],status['droppedRecords'],status['failed']))
        official=json.loads((d/'battle.json').read_bytes());self.assertEqual(5,official['numberOfRounds'])
        self.assertEqual(g['battle']['results'],official['results']);self.assertEqual(g['scoreShare'],compare.score_share(g['battle']))
        self.assertEqual((0,0,237,13),tuple(sum(r[k] for r in rows) for k in ('won','skippedTurns','shotsFired','bulletHits')))
        result=json.loads((ROOT/'analysis.json').read_bytes());self.assertEqual([1,2,3,4,5],result['completeActualHitCoverageRounds'])
if __name__=='__main__':unittest.main()
