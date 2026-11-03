"""Reproduce frozen fresh parity evidence without games or refitting."""
import hashlib
import json
from pathlib import Path
import unittest
import benchmark
import block_experiment as b
from coherent_duel_v2_experiment import validate_endpoint

ROOT=Path(__file__).resolve().parents[1]/'benchmarks/replay-harness-parity-redo'

class ReplayRedoEvidenceTests(unittest.TestCase):
    def test_all_preserved_bytes_and_exact_noninferiority_rules(self):
        for name,digest in json.loads((ROOT/'files.sha256.json').read_bytes())['files'].items():
            self.assertEqual(digest,hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),name)
        r=json.loads((ROOT/'raw/experiment.json').read_bytes())
        self.assertEqual(r['primary'],b.noninferiority_decision(r['primaryBlocks'],30,'t'))
        self.assertEqual(r['decision'],b.final_decision(r['primary'],r['samples'],None,False))
        self.assertEqual('retain',r['decision']['status']);self.assertEqual([],r['onlineBlocks'])
        self.assertEqual('2edd072b365bfc2b3e50f66adf970b6de37fd8b7',r['planCommit'])

    def test1260_cold_rows_66_finite_endpoints_and_60_native_exports(self):
        r=json.loads((ROOT/'raw/experiment.json').read_bytes());count=0;models=[];native=0
        games=[]
        for i,block in enumerate(r['primaryBlocks'],1):
            self.assertEqual(20262800+i,block['seed'])
            for arm in ('previous','candidate'):games.append((ROOT/'raw/primary'/f'block-{i:02d}'/arm,'Retcon',block['seed'],20,block[arm]))
        for sample in r['samples']:
            for arm in ('previous','candidate'):games.append((ROOT/'raw/samples'/sample['opponent']/arm,sample['opponent'],20262899,10,sample[arm]))
        for path,op,seed,n,battle in games:
            run=path/(op+'-train');rows=[json.loads(line) for line in (run/'metrics.jsonl').read_bytes().splitlines()]
            benchmark.validate_metrics(rows,n,'train',seed,20)
            self.assertEqual(battle['metrics'],benchmark.summarize_metrics(rows));self.assertEqual(0,battle['metrics']['skippedTurns'])
            endpoint=validate_endpoint(path/'model.dat',n);self.assertEqual(battle['modelSHA256'],endpoint['sha256'])
            official=json.loads((run/'battle.json').read_bytes());self.assertEqual(n,official['numberOfRounds']);self.assertEqual(battle['results'],official['results'])
            self.assertEqual(2,len({result['id'] for result in official['results']}))
            self.assertEqual(battle['scoreShare'],sum(result['totalScore'] for result in official['results'] if result['name']=='AISurvivor')/sum(result['totalScore'] for result in official['results']))
            if op=='Retcon':
                self.assertEqual('complete',battle['retconStats']['status']);native+=1
                for file in battle['retconStats']['rawArtifacts']:
                    self.assertEqual(file['sha256'],hashlib.sha256((run/'retcon-stats'/Path(file['path']).name).read_bytes()).hexdigest())
            count+=len(rows);models.append(endpoint['sha256'])
        self.assertEqual((1260,66,66,60),(count,len(models),len(set(models)),native))

if __name__=='__main__':unittest.main()
