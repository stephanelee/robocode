"""Reproduce the original rejected challenger pilot without running or tuning it."""
import hashlib
import json
from pathlib import Path
import unittest
import benchmark
import compare
import diamond_challenger_experiment as e

ROOT=Path(__file__).resolve().parents[1]/'benchmarks/diamond-pressure-v1'

class DiamondPressureEvidenceTests(unittest.TestCase):
    def test_original_bytes_plan_and_exact_pilot_rejection(self):
        for name,digest in json.loads((ROOT/'files.sha256.json').read_bytes())['files'].items():
            self.assertEqual(digest,hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),name)
        r=json.loads((ROOT/'experiment.json').read_bytes())
        self.assertEqual(r['pilotDecision'],e.decision(r['games'],'pilot'));self.assertEqual(r['pilotDecision'],r['decision'])
        self.assertEqual(['progressShare','oneWin'],[k for k,v in r['decision']['checks'].items() if not v])
        self.assertEqual(r['planSHA256'],hashlib.sha256((ROOT/'registered-plan.md').read_bytes()).hexdigest())
        self.assertEqual((ROOT/'source/reviewed-workflow.yml').read_bytes().replace(b'\r\n',b'\n'),(ROOT/'source/executed-workflow.yml').read_bytes())

    def test_all20_cold_rows_private_endpoint_and_both_official_identities(self):
        r=json.loads((ROOT/'experiment.json').read_bytes());self.assertEqual(1,len(r['games']));game=r['games'][0]
        path=ROOT/'raw'/game['gameId'];rows=[json.loads(line) for line in (path/'metrics.jsonl').read_bytes().splitlines()]
        e.check_rows(rows,20,20263101);self.assertEqual(game['metrics'],benchmark.summarize_metrics(rows))
        self.assertEqual(game['endpoint'],e.endpoint(path/'model.json',20,20263101));self.assertEqual(452.5,sum(row['energySpent'] for row in rows))
        official=json.loads((path/'battle.json').read_bytes());self.assertEqual(20,official['numberOfRounds'])
        self.assertEqual(game['battle']['results'],official['results']);self.assertEqual(game['scoreShare'],compare.score_share(game['battle']))
        self.assertEqual({('DiamondChallenger','0.1.0'),('Diamond Port','1.8.28')},{(v['name'],v['version']) for v in official['results']})
        self.assertEqual((0,0,11,274),(sum(row['won'] for row in rows),sum(row['skippedTurns'] for row in rows),sum(row['bulletHits'] for row in rows),sum(row['shotsFired'] for row in rows)))

if __name__=='__main__':unittest.main()
