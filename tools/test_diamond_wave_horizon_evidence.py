"""Reproduce frozen movement-pilot evidence; never refit, retune or replay games."""
import hashlib,json,unittest
from pathlib import Path
import benchmark,compare,diamond_wave_horizon_experiment as e
ROOT=Path(__file__).resolve().parents[1]/'benchmarks/diamond-wave-horizon-v4'
class WaveHorizonEvidenceTests(unittest.TestCase):
    def test_original_bytes_plan_and_negative_decision(self):
        for name,digest in json.loads((ROOT/'files.sha256.json').read_bytes())['files'].items():
            self.assertEqual(digest,hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),name)
        r=json.loads((ROOT/'experiment.json').read_bytes())
        self.assertEqual(r['decision'],e.decision(r['games'],'pilot'))
        self.assertEqual(['progressShare'],[k for k,v in r['decision']['checks'].items() if not v])
        self.assertNotIn('confirmationDecision',r)
        self.assertEqual(r['planSHA256'],hashlib.sha256((ROOT/'registered-plan.md').read_bytes()).hexdigest())
        self.assertEqual((ROOT/'source/reviewed-workflow.yml').read_bytes().replace(b'\r\n',b'\n'),(ROOT/'source/executed-workflow.yml').read_bytes())
    def test_all20_cold_rows_own_memory_and_actual_wave_activation(self):
        r=json.loads((ROOT/'experiment.json').read_bytes());self.assertEqual(1,len(r['games']));g=r['games'][0]
        d=ROOT/'raw'/g['gameId'];rows=[json.loads(line) for line in (d/'metrics.jsonl').read_bytes().splitlines()]
        e.check_rows(rows,20,20263401);self.assertEqual(g['metrics'],benchmark.summarize_metrics(rows))
        self.assertEqual(g['endpoint'],e.endpoint(d/'model.json',20,20263401))
        for k,v in g['waveActivity'].items():self.assertEqual(v,sum(row[k] for row in rows))
        self.assertEqual((1228,220,24,17485,17164,3819),tuple(g['waveActivity'][k] for k in ('inferredWaves','matchedHits','unmatchedHits','plannerDecisions','profileDecisions','stopDecisions')))
        self.assertEqual((1,0,80,1149),tuple(sum(row[k] for row in rows) for k in ('won','skippedTurns','bulletHits','shotsFired')))
        state=g['endpoint']['state'];self.assertEqual((20,512,1228,220),(state['roundsCompleted'],state['contextMemorySize'],state['inferredWavesTotal'],state['matchedHitsTotal']))
        official=json.loads((d/'battle.json').read_bytes());self.assertEqual(20,official['numberOfRounds'])
        self.assertEqual(g['battle']['results'],official['results']);self.assertEqual(g['scoreShare'],compare.score_share(g['battle']))
        self.assertEqual({('DiamondChallenger','0.4.0'),('Diamond Port','1.8.28')},{(v['name'],v['version']) for v in official['results']})
if __name__=='__main__':unittest.main()
