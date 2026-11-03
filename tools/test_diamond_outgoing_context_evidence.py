"""Preserve the invalid attempt; never repair rows, waive integrity, replay or fit."""
import hashlib,json,unittest
from pathlib import Path
import benchmark,compare,diamond_outgoing_context_experiment as e
ROOT=Path(__file__).resolve().parents[1]/'benchmarks/diamond-outgoing-context-v7'
class OutgoingInvalidEvidenceTests(unittest.TestCase):
    def test_original_invalid_report_and_all_evidence_bytes_remain_intact(self):
        for name,digest in json.loads((ROOT/'files.sha256.json').read_bytes())['files'].items():
            self.assertEqual(digest,hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),name)
        r=json.loads((ROOT/'experiment.json').read_bytes())
        self.assertEqual(('invalid','Shared cohort totals differ',[]),(r['status'],r['failure'],r['games']))
        self.assertNotIn('decision',r);self.assertNotIn('pilotDecision',r);self.assertNotIn('confirmationDecision',r)
        with self.assertRaises(ValueError):e.decision(r['games'],'pilot')
        self.assertEqual(r['planSHA256'],hashlib.sha256((ROOT/'registered-plan.md').read_bytes()).hexdigest())
        self.assertEqual((ROOT/'source/reviewed-workflow.yml').read_bytes().replace(b'\r\n',b'\n'),(ROOT/'source/executed-workflow.yml').read_bytes())
        self.assertIn('difference=1',(ROOT/'lifecycle-probe/result.txt').read_text())
        self.assertIn('ACTUAL_PR65_CALLBACK_CAUSE_UNPROVEN',(ROOT/'lifecycle-probe/result.txt').read_text())
    def test_all20_physical_rows_and_endpoint_preserve_the_failure_without_reclassification(self):
        d=ROOT/'raw/diamond-outgoing-context-v7-pilot';rows=[json.loads(l) for l in (d/'metrics.jsonl').read_bytes().splitlines()]
        e.check_rows(rows,20,20263701);endpoint=e.endpoint(d/'model.json',20,20263701);m=endpoint['state']
        self.assertEqual((20,13,512,683),(m['roundsCompleted'],m['contextFeatureCount'],m['contextMemorySize'],m['contextAccepted']))
        self.assertEqual([531]*4,m['pairedAttempts']);self.assertEqual([46,31,35,50],m['pairedHits'])
        self.assertEqual((551,530,0,21),tuple(sum(r[k] for r in rows) for k in ('pairedCreated','pairedCompleted','pairedDiscarded','pairedPending')))
        self.assertNotEqual(sum(r['pairedCompleted'] for r in rows),m['pairedAttempts'][0])
        self.assertEqual((1,1),(rows[-1]['pairedPending'],m['pairedPending']))
        self.assertEqual((0,0,703,37),tuple(sum(r[k] for r in rows) for k in ('won','skippedTurns','shotsFired','bulletHits')))
        b=json.loads((d/'battle.json').read_bytes());self.assertEqual(20,b['numberOfRounds'])
        own=next(v for v in b['results'] if v['name']==e.NAME)
        o=json.loads((ROOT/'descriptive-observations.json').read_bytes());self.assertEqual('invalid',o['canonicalStatus'])
        self.assertEqual(o['metrics'],benchmark.summarize_metrics(rows));self.assertEqual(0,own['firstPlaces'])
        self.assertEqual(o['scoreShare'],compare.score_share(dict(**own,results=b['results'])))
        self.assertEqual((15438,10287,7347),tuple(sum(r[k] for r in rows) for k in ('waveContextQueries','waveContextOne','waveContextTwo')))
if __name__=='__main__':unittest.main()
