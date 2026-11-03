"""Reproduce the frozen rejected screen; never run games or tune its rules."""
import hashlib
import json
from pathlib import Path
import unittest
import benchmark
import learned_danger_experiment as m

ROOT=Path(__file__).resolve().parents[1]/'benchmarks/learned-duel-danger'

class LearnedDangerEvidenceTests(unittest.TestCase):
    def test_preserved_bytes_and_exact_registered_rejection(self):
        manifest=json.loads((ROOT/'files.sha256.json').read_bytes())
        for name,digest in manifest['files'].items():
            self.assertEqual(digest,hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),name)
        report=json.loads((ROOT/'experiment.json').read_bytes())
        self.assertEqual(report['screenDecision'],m.screen_decision(report['games']))
        self.assertEqual(['retconRelative'],[k for k,v in report['screenDecision']['checks'].items() if not v])
        self.assertEqual(dict(status='reject-at-screen',confirmationRounds=0),report['decision'])
        self.assertEqual(report['planSHA256'],hashlib.sha256((ROOT/'registered-plan.md').read_bytes()).hexdigest())
        self.assertEqual(report['protocolSHA256'],hashlib.sha256((ROOT/'registered-protocol.json').read_bytes()).hexdigest())
        self.assertEqual((ROOT/'source/reviewed-workflow.yml').read_bytes().replace(b'\r\n',b'\n'),(ROOT/'source/executed-workflow.yml').read_bytes())

    def test_all60_rows_six_cold_models_official_identities_and_native_bytes(self):
        report=json.loads((ROOT/'experiment.json').read_bytes());count=0;models=[]
        jars=json.loads((ROOT/'bundle-jar-hashes.json').read_bytes())
        m.validate_layout(report['games'],['screen'])
        for game in report['games']:
            directory=ROOT/'raw'/game['gameId'];battle=game['battle']
            rows=[json.loads(line) for line in (directory/'metrics.jsonl').read_bytes().splitlines()]
            benchmark.validate_metrics(rows,10,'train',game['seed'],10)
            self.assertEqual(battle['metrics'],benchmark.summarize_metrics(rows))
            self.assertEqual(game['selection'],m.measured.selection(rows))
            self.assertEqual(game['endpoint'],m.measured.reusable.validate_endpoint(directory/'model.dat',10))
            self.assertEqual(game['endpoint']['sha256'],battle['modelSHA256'])
            self.assertEqual(m.JARS[game['arm']],jars[game['gameId']])
            self.assertEqual(game['botAssetHashes']['AISurvivor.jar'],jars[game['gameId']])
            official=json.loads((directory/'battle.json').read_bytes())
            self.assertEqual(10,official['numberOfRounds']);self.assertEqual(official['results'],battle['results'])
            expected=[json.loads((directory/'AISurvivor/AISurvivor.json').read_bytes()),json.loads((directory/'opponents'/game['opponent']/(game['opponent']+'.json')).read_bytes())]
            self.assertEqual(sorted((d['name'],d['version']) for d in expected),sorted((d['name'],d['version']) for d in official['results']))
            self.assertEqual(2,len({r['id'] for r in official['results']}))
            if game['opponent']=='Retcon':
                self.assertEqual('complete',battle['retconStats']['status'])
                for file in battle['retconStats']['rawArtifacts']:
                    self.assertEqual(file['sha256'],hashlib.sha256((directory/'retcon-stats'/Path(file['path']).name).read_bytes()).hexdigest())
            count+=len(rows);models.append(game['endpoint']['sha256'])
        self.assertEqual((60,6,6),(count,len(models),len(set(models))))

if __name__=='__main__':unittest.main()
