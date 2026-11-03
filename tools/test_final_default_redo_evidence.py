"""Verify the separately authorized redo without pooling or replacing prior outcomes."""
import hashlib
import json
from pathlib import Path
import unittest
import benchmark
import coherent_duel_v2_experiment as harness
import final_default_confirmation as previous
import final_default_confirmation_redo as protocol

ROOT=Path(__file__).resolve().parents[1]/'benchmarks/final-default-confirmation-redo'


class FinalDefaultRedoEvidenceTests(unittest.TestCase):
    def test_original_bytes_registered_confirmation_and_execution_link(self):
        manifest=json.loads((ROOT/'files.sha256.json').read_bytes())
        self.assertEqual((41584002511,1,101),(manifest['runId'],manifest['attempt'],len(manifest['files'])))
        for name,digest in manifest['files'].items():
            self.assertEqual(digest,hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),name)
        report=json.loads((ROOT/'confirmation.json').read_bytes())
        self.assertEqual(report['decision'],protocol.decision(report['games']))
        self.assertEqual(('confirmed',187,200,10),(report['decision']['status'],report['decision']['wins'],report['decision']['rounds'],report['decision']['gamesAboveHalf']))
        self.assertGreater(report['decision']['gameMeanTLower95'],.5)
        first=json.loads((ROOT/'batch-1.json').read_bytes());second=json.loads((ROOT/'batch-2.json').read_bytes())
        self.assertEqual(('complete','complete'),(first['status'],second['status']))
        self.assertEqual(first['planSHA256'],hashlib.sha256((ROOT/'registered-plan.md').read_bytes()).hexdigest())
        self.assertEqual(protocol.expected_layout(),json.loads((ROOT/'registered-games.json').read_bytes())['games'])
        provenance=json.loads((ROOT/'execution-provenance.json').read_bytes())
        self.assertEqual(report['executionSource'],provenance['executionHead'])
        reviewed=(ROOT/'source/reviewed-workflow-windows.yml').read_bytes()
        executed=(ROOT/'source/final-default-confirmation-redo.yml').read_bytes()
        self.assertEqual(provenance['reviewedWorkflowSha256'],hashlib.sha256(reviewed).hexdigest())
        self.assertEqual(provenance['executedGitWorkflowSha256'],hashlib.sha256(executed).hexdigest())
        self.assertEqual(reviewed.replace(b'\r\n',b'\n'),executed)
        for key in ('gameId','seed'):
            self.assertFalse({g[key] for g in protocol.expected_layout()}&{g[key] for g in previous.expected_layout()})
        self.assertEqual('invalid',json.loads((ROOT.parent/'final-default-confirmation/attempt-summary.json').read_bytes())['acquisitionStatus'])

    def test_all200_official_raw_rounds_and_ten_independent_cold_models(self):
        report=json.loads((ROOT/'confirmation.json').read_bytes());count=0;models=[]
        jars=json.loads((ROOT/'bundle-jar-hashes.json').read_bytes())
        for game in report['games']:
            directory=ROOT/'raw'/game['gameId'];bot=directory/'candidate/AISurvivor';battle=game['battles']['candidate']
            rows=[json.loads(line) for line in (bot/'metrics.jsonl').read_bytes().splitlines()]
            benchmark.validate_metrics(rows,20,'train',game['seed'],20)
            self.assertEqual(battle['metrics'],benchmark.summarize_metrics(rows))
            self.assertEqual(0,battle['metrics']['skippedTurns']);self.assertFalse(battle['metrics']['checkpointLoaded'])
            self.assertEqual(battle['endpoint'],harness.validate_endpoint(bot/'model.dat',20))
            self.assertEqual(battle['mechanism'],harness.policy.mechanism(rows))
            official=json.loads((directory/'battle.json').read_bytes());self.assertEqual(20,official['numberOfRounds'])
            self.assertEqual(official['results'],battle['results'])
            identity=next(r for r in official['results'] if r['name']==battle['bot']['name'])
            for key,value in identity.items():self.assertEqual(value,battle[key])
            self.assertEqual(protocol.JAR,jars[f"{game['gameId']}/candidate/AISurvivor/AISurvivor.jar"])
            count+=len(rows);models.append(battle['endpoint']['sha256'])
        self.assertEqual((200,10,10),(count,len(models),len(set(models))))


if __name__=='__main__':unittest.main()
