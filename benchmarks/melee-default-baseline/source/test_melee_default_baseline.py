import argparse
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import melee_default_baseline as m


class MeleeBaselineTests(unittest.TestCase):
    def test_descriptive_summary_keeps_skips_and_zero_duel_eligibility(self):
        rows=[dict(duelPolicy='controller',actionCounts=[0,3,0,0,0,0],duelEligibleTurns=0,forcedDuelTurns=0)]
        self.assertEqual(0,m.selection(rows)['forcedDuelTurns'])
        games=[dict(**g,scoreShare=.2,battle=dict(rank=3,firstPlaces=2,metrics=dict(turns=100,shotsFired=10,bulletHits=1,damageDealt=4,damageTaken=10,wallCollisions=2,botCollisions=1,skippedTurns=1)),selection=m.selection(rows)) for g in m.expected_layout()]
        result=m.summarize(games)
        self.assertEqual(('descriptive-baseline',60,[3,3,3],3),(result['disposition'],result['rounds'],result['officialGameRanks'],result['metrics']['skippedTurns']))
        with self.assertRaises(ValueError):m.summarize(games[:-1])

    def test_exact_single_train_call_and_environment_cleanup_restoration(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp).resolve();run=directory/'Melee-train'
            descriptors=[('AISurvivor','AISurvivor'),('Target','opponents/Target'),('SpinBot','opponents/SpinBot'),('Walls','opponents/Walls')]
            official=[]
            for i,(name,folder) in enumerate(descriptors,1):
                p=run/folder;p.mkdir(parents=True)
                (p/(name+'.json')).write_text(json.dumps(dict(name=name,version='1.0')))
                official.append(dict(id=i,name=name,version='1.0',totalScore=100))
            rows=[dict(duelPolicy='controller',actionCounts=[0,1,0,0,0,0],duelEligibleTurns=0,forcedDuelTurns=0)]*20
            (run/'metrics.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
            def battle(*args):
                self.assertNotIn('AISURVIVOR_DUEL_POLICY',os.environ);self.assertNotIn('AISURVIVOR_DATASET_PATH',os.environ)
                self.assertEqual((['Target','SpinBot','Walls'],'train',20),(args[5],args[6],args[7]))
                return dict(totalScore=100,results=official)
            with patch.dict(os.environ,dict(AISURVIVOR_DUEL_POLICY='learned',AISURVIVOR_DATASET_PATH='unwanted')):
                with patch.object(m.benchmark,'battle',side_effect=battle) as called,patch.object(m.reusable,'validate_endpoint',return_value=dict(totalRounds=20)):
                    result=m.play(argparse.Namespace(),directory,Path('runner'),Path('samples'),m.expected_layout()[0])
                self.assertEqual('learned',os.environ['AISURVIVOR_DUEL_POLICY']);self.assertEqual(1,called.call_count)
                self.assertEqual(.25,result['scoreShare'])


if __name__=='__main__':unittest.main()
