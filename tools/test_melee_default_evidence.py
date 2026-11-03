"""Reproduce the frozen descriptive sample-melee measurement without new games."""
import hashlib
import json
from pathlib import Path
import unittest
import benchmark
import compare
import melee_default_baseline as m

ROOT=Path(__file__).resolve().parents[1]/'benchmarks/melee-default-baseline'


class MeleeEvidenceTests(unittest.TestCase):
    def test_original_bytes_registration_and_descriptive_summary(self):
        manifest=json.loads((ROOT/'files.sha256.json').read_bytes())
        self.assertEqual((41643133213,13232269909,1,39),(manifest['runId'],manifest['artifactId'],manifest['attempt'],len(manifest['files'])))
        for name,digest in manifest['files'].items():self.assertEqual(digest,hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),name)
        report=json.loads((ROOT/'baseline.json').read_bytes())
        self.assertEqual('complete',report['status']);self.assertEqual(report['summary'],m.summarize(report['games']))
        self.assertEqual((60,[1,1,1],60),(report['summary']['rounds'],report['summary']['officialGameRanks'],report['summary']['officialFirstPlaces']))
        self.assertEqual(report['planSHA256'],hashlib.sha256((ROOT/'registered-plan.md').read_bytes()).hexdigest())
        self.assertEqual(m.expected_layout(),json.loads((ROOT/'registered-games.json').read_bytes())['games'])
        provenance=json.loads((ROOT/'execution-provenance.json').read_bytes())
        self.assertEqual(provenance['executionHead'],report['executionSource'])
        self.assertEqual(provenance['executedWorkflowSHA256'],hashlib.sha256((ROOT/'source/melee-default-baseline.yml').read_bytes()).hexdigest())

    def test_all60_rows_three_cold_models_and_complete_four_bot_identities(self):
        report=json.loads((ROOT/'baseline.json').read_bytes());count=0;models=[]
        jars=json.loads((ROOT/'bundle-jar-hashes.json').read_bytes())
        for game in report['games']:
            directory=ROOT/'raw'/game['gameId'];battle=game['battle']
            rows=[json.loads(line) for line in (directory/'metrics.jsonl').read_bytes().splitlines()]
            benchmark.validate_metrics(rows,20,'train',game['seed'],20)
            self.assertEqual(battle['metrics'],benchmark.summarize_metrics(rows));self.assertEqual(game['selection'],m.selection(rows))
            self.assertEqual(game['endpoint'],m.reusable.validate_endpoint(directory/'model.dat',20))
            self.assertEqual(game['endpoint']['sha256'],battle['modelSHA256']);self.assertEqual(m.JAR,jars[game['gameId']])
            official=json.loads((directory/'battle.json').read_bytes());self.assertEqual(20,official['numberOfRounds'])
            self.assertEqual(official['results'],battle['results']);self.assertEqual(game['scoreShare'],compare.score_share(battle))
            expected=[json.loads((directory/'AISurvivor/AISurvivor.json').read_bytes())]
            expected += [json.loads((directory/'opponents'/name/(name+'.json')).read_bytes()) for name in m.OPPONENTS]
            self.assertEqual(sorted((d['name'],d['version']) for d in expected),sorted((d['name'],d['version']) for d in official['results']))
            self.assertEqual(4,len({r['id'] for r in official['results']}));count+=len(rows);models.append(game['endpoint']['sha256'])
        self.assertEqual((60,3,3),(count,len(models),len(set(models))))


if __name__=='__main__':unittest.main()
