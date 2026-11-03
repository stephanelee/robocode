"""Preserve the invalid attempt; never replace games or score a favorable subset."""
import hashlib
import json
from pathlib import Path
import unittest
import benchmark
import compare
import coherent_duel_v2_experiment as harness
import final_default_confirmation as protocol

ROOT=Path(__file__).resolve().parents[1]/'benchmarks/final-default-confirmation'


class FinalDefaultEvidenceTests(unittest.TestCase):
    def test_original_hashes_reports_and_invalid_stopping_boundary(self):
        manifest=json.loads((ROOT/'files.sha256.json').read_bytes())
        self.assertEqual((41575285157,1,72),(manifest['runId'],manifest['attempt'],len(manifest['files'])))
        for name,digest in manifest['files'].items():
            self.assertEqual(digest,hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),name)
        first=json.loads((ROOT/'batch-1.json').read_bytes());second=json.loads((ROOT/'batch-2.json').read_bytes())
        self.assertEqual(('complete',5),(first['status'],len(first['games'])))
        self.assertEqual(('invalid',1,'dc946-b2-g02'),(second['status'],len(second['games']),second['inProgressGame']))
        self.assertEqual(protocol.SOURCE,first['botSource']);self.assertEqual(protocol.JAR,second['botJarSHA256'])
        self.assertEqual(first['planSHA256'],hashlib.sha256((ROOT/'registered-plan.md').read_bytes()).hexdigest())
        self.assertEqual(protocol.expected_layout(),json.loads((ROOT/'registered-games.json').read_bytes())['games'])
        self.assertFalse((ROOT/'confirmation.json').exists())

    def test_all140_raw_rounds_seven_models_and_single_skipped_turn_are_retained(self):
        summary=json.loads((ROOT/'attempt-summary.json').read_bytes());count=0;validated=0;skips=[];models=[]
        self.assertEqual(('not-confirmed','invalid',140,120,60),(summary['status'],summary['acquisitionStatus'],summary['physicalRounds'],summary['validatedRounds'],summary['unplayedRounds']))
        self.assertFalse(summary['aggregateWasRun']);self.assertFalse(summary['completeConfirmationDecisionWasEvaluated'])
        for game in summary['games']:
            directory=ROOT/'raw'/game['gameId'];bot=directory/'candidate/AISurvivor'
            rows=[json.loads(line) for line in (bot/'metrics.jsonl').read_bytes().splitlines()]
            benchmark.validate_metrics(rows,20,'train',game['seed'],20)
            self.assertEqual(game['metrics'],benchmark.summarize_metrics(rows))
            self.assertEqual(game['endpoint'],harness.validate_endpoint(bot/'model.dat',20))
            self.assertEqual(game['mechanism'],harness.policy.mechanism(rows))
            official=json.loads((directory/'battle.json').read_bytes())
            identity=next(r for r in official['results'] if r['name']=='AISurvivor-Candidate')
            self.assertEqual(20,official['numberOfRounds'])
            self.assertEqual(game['observedRoundWins'],identity['firstPlaces'])
            self.assertEqual(game['observedScoreShare'],compare.score_share(dict(**identity,results=official['results'])))
            count+=len(rows);validated+=len(rows) if game['strictlyValid'] else 0
            skips.extend((game['gameId'],r['round'],r['skippedTurns']) for r in rows if r['skippedTurns'])
            models.append(game['endpoint']['sha256'])
        self.assertEqual((140,120,7,7),(count,validated,len(models),len(set(models))))
        self.assertEqual([('dc946-b2-g02',4,1)],skips)
        for game in protocol.expected_layout()[7:]:self.assertFalse((ROOT/'raw'/game['gameId']).exists())
        with self.assertRaises(ValueError):protocol.decision(json.loads((ROOT/'batch-1.json').read_bytes())['games'])


if __name__=='__main__':unittest.main()
