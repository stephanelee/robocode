"""Verify original bytes and reproduce the frozen result; never launch extra games."""
import hashlib
import json
from pathlib import Path
import unittest
import benchmark
import coherent_duel_experiment as e

ROOT=Path(__file__).resolve().parents[1]/'benchmarks/coherent-duel-v1'


class CoherentEvidenceTests(unittest.TestCase):
    def test_archived_bytes_original_registration_and_exact_decision(self):
        manifest=json.loads((ROOT/'files.sha256.json').read_bytes())
        self.assertEqual((41564742819,13198895411,1,139),(manifest['runId'],manifest['artifactId'],manifest['attempt'],len(manifest['files'])))
        for name,digest in manifest['files'].items():
            self.assertEqual(digest,hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),name)
        report=json.loads((ROOT/'experiment.json').read_bytes())
        self.assertEqual(report['planSHA256'],hashlib.sha256((ROOT/'registered-plan.md').read_bytes()).hexdigest())
        self.assertEqual(e.PLAN_COMMIT,report['registrationCommit'])
        self.assertEqual(e.layout(),json.loads((ROOT/'registered-games.json').read_bytes()))
        self.assertEqual(report['decision'],e.decision(report['games']))
        self.assertEqual('retain',report['decision']['status'])

    def test_all198_records_18_independent_endpoints_and_official_identities(self):
        report=json.loads((ROOT/'experiment.json').read_bytes())
        jars=json.loads((ROOT/'bundle-jar-hashes.json').read_bytes())
        records_count=0;models=[];physical=0
        for game in report['games']:
            directory=ROOT/'raw'/game['gameId']
            official=json.loads((directory/'battle.json').read_bytes())
            self.assertEqual(game['rounds'],official['numberOfRounds']);physical+=game['rounds']
            for arm,battle in game['battles'].items():
                bot=directory/arm/'AISurvivor'
                records=[json.loads(line) for line in (bot/'metrics.jsonl').read_bytes().splitlines()]
                benchmark.validate_metrics(records,game['rounds'],'train',game['seeds'][arm],game['rounds'])
                self.assertEqual(battle['metrics'],benchmark.summarize_metrics(records))
                self.assertEqual(0,battle['metrics']['skippedTurns'])
                self.assertFalse(battle['metrics']['checkpointLoaded'])
                self.assertEqual(battle['endpoint'],e.validate_endpoint(bot/'model.dat',game['rounds']))
                self.assertEqual(official['results'],battle['results'])
                identity=[r for r in official['results'] if r['name']==battle['bot']['name']]
                self.assertEqual(1,len(identity));self.assertEqual(battle['id'],identity[0]['id'])
                self.assertEqual(e.JARS[arm],jars[f"{game['gameId']}/{arm}/AISurvivor/AISurvivor.jar"])
                if arm=='candidate':self.assertEqual(battle['mechanism'],e.policy.mechanism(records))
                models.append(battle['endpoint']['sha256']);records_count+=len(records)
        self.assertEqual((168,198,18,18),(physical,records_count,len(models),len(set(models))))


if __name__=='__main__':unittest.main()
