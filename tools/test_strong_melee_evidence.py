"""Preserve every negative melee outcome and the separate zero-game acquisition failure."""
import hashlib
import json
from pathlib import Path
import unittest
import benchmark
import compare
import strong_melee_baseline as m

ROOT=Path(__file__).resolve().parents[1]/'benchmarks/strong-melee-baseline'


class StrongMeleeEvidenceTests(unittest.TestCase):
    def test_original_bytes_negative_summary_and_zero_game_failure(self):
        manifest=json.loads((ROOT/'files.sha256.json').read_bytes());self.assertEqual(67,len(manifest['files']))
        for name,digest in manifest['files'].items():self.assertEqual(digest,hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),name)
        report=json.loads((ROOT/'baseline.json').read_bytes());self.assertEqual('complete',report['status'])
        self.assertEqual(report['summary'],m.summarize(report['games']))
        self.assertEqual([4,4,4],report['summary']['opponents']['AISurvivor']['officialGameRanks'])
        self.assertEqual(1,report['summary']['opponents']['AISurvivor']['officialFirstPlaces'])
        self.assertEqual(report['planSHA256'],hashlib.sha256((ROOT/'registered-plan.md').read_bytes()).hexdigest())
        self.assertEqual(m.expected_layout(),json.loads((ROOT/'registered-games.json').read_bytes())['games'])
        failed=json.loads((ROOT/'failed-preflight/original-artifact/build/strong-melee-baseline/baseline.json').read_bytes())
        self.assertEqual(('invalid',[]),(failed['status'],failed['games']))
        self.assertFalse(list((ROOT/'failed-preflight').rglob('battle.json')))
        lock=json.loads((ROOT/'python-opponents.lock.json').read_bytes())
        normalize=lambda s:s.lower().replace('_','-')
        self.assertEqual({(normalize(w['name']),w['version']) for w in lock['runtime']['wheels']},
            {(normalize(n),v) for n,v in report['pythonOpponents']['runtime']['packages']})

    def test_all60_raw_rows_three_cold_models_four_identities_and_native_captures(self):
        report=json.loads((ROOT/'baseline.json').read_bytes());count=0;models=[]
        jars=json.loads((ROOT/'bundle-jar-hashes.json').read_bytes())
        for game in report['games']:
            directory=ROOT/'raw'/game['gameId'];battle=game['battle']
            rows=[json.loads(line) for line in (directory/'metrics.jsonl').read_bytes().splitlines()]
            benchmark.validate_metrics(rows,20,'train',game['seed'],20)
            self.assertEqual(battle['metrics'],benchmark.summarize_metrics(rows));self.assertEqual(game['selection'],m.measured.selection(rows))
            self.assertEqual(game['endpoint'],m.measured.reusable.validate_endpoint(directory/'model.dat',20))
            self.assertEqual(game['endpoint']['sha256'],battle['modelSHA256']);self.assertEqual(m.JAR,jars[game['gameId']])
            official=json.loads((directory/'battle.json').read_bytes());self.assertEqual(20,official['numberOfRounds'])
            self.assertEqual(official['results'],battle['results']);self.assertEqual(game['scoreShare'],compare.score_share(battle))
            expected=[json.loads((directory/'AISurvivor/AISurvivor.json').read_bytes())]
            expected += [json.loads((directory/'opponents'/name/(name+'.json')).read_bytes()) for name in m.OPPONENTS]
            self.assertEqual(sorted((d['name'],d['version']) for d in expected),sorted((d['name'],d['version']) for d in official['results']))
            self.assertEqual(4,len({r['id'] for r in official['results']}))
            capture=json.loads((directory/'retcon-stats/capture.json').read_bytes());self.assertEqual(('complete',20),(capture['status'],capture['data']['roundsRecorded']))
            for raw in capture['rawArtifacts']:self.assertEqual(raw['sha256'],hashlib.sha256((directory/'retcon-stats'/raw['path']).read_bytes()).hexdigest())
            self.assertEqual(109,len(game['executedPythonSourceHashes']));count+=len(rows);models.append(game['endpoint']['sha256'])
        self.assertEqual((60,3,3),(count,len(models),len(set(models))))


if __name__=='__main__':unittest.main()
