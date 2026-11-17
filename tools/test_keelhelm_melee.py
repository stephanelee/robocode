import copy
import gzip
import json
from pathlib import Path
import tempfile
import unittest
import keelhelm_melee as m

class ComparisonContracts(unittest.TestCase):
    def test_schedule_cannot_spend_more_than_registered_and_covers_four_launch_slots(self):
        rows=m.schedule()
        self.assertEqual((16,220),(len(rows),sum(g['rounds'] for g in rows)))
        for arm in ('control','candidate'):
            melee=[g for g in rows if g['arm']==arm and g['scenario']=='melee']
            self.assertEqual([0,3,2,1],[g['order'].index('KeelHelm') for g in melee])
            self.assertEqual(80,sum(g['rounds'] for g in melee))
            self.assertTrue(all(set(g['order'])=={'KeelHelm','Diamond Port','Keel','Prism'} for g in melee))

    def test_acceptance_uses_exact_score_ratios_at_percentage_point_boundary(self):
        rows=[]
        for scenario in ('melee','Retcon','Walls','SpinBot'):
            for arm,score in [('control',20),('candidate',25)]:
                official=dict(results=[dict(totalScore=score),dict(totalScore=100-score)])
                rows.append(dict(scenario=scenario,arm=arm,scoreShare=score/100,
                    helm=dict(totalScore=score,firstPlaces=1),official=official))
        self.assertEqual('accept-measured-melee-improvement',m.summary(rows)['decision'])
        rows[1]['helm']['totalScore']=24;rows[1]['official']['results'][0]['totalScore']=24
        rows[1]['official']['results'][1]['totalScore']=76;rows[1]['scoreShare']=.24
        self.assertFalse(m.summary(rows)['gates']['meleeGain'])

    def test_audit_rejects_missing_ticks_wrong_final_and_corrupted_recording(self):
        participants=[dict(id=1,name='KeelHelm',version='0.2.0'),dict(id=2,name='Walls',version='1')]
        official=dict(numberOfRounds=1,results=[dict(**p,totalScore=100) for p in participants])
        rows=[dict(type='GameStartedEventForObserver',participants=participants,
              gameSetup=dict(arenaWidth=800,arenaHeight=600,numberOfRounds=1,turnTimeout=100000)),
              dict(type='TickEventForObserver',roundNumber=1,turnNumber=1,events=[],
                   botStates=[dict(id=1,enemyCount=1),dict(id=2,enemyCount=1)]),
              dict(type='RoundEndedEventForObserver',roundNumber=1,turnNumber=1),
              dict(type='GameEndedEventForObserver',**official)]
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'recording.gz'
            def write(value):
                with gzip.open(path,'wt') as stream:
                    for row in value:stream.write(json.dumps(row)+'\n')
            def audit():return m.audit(path,official,dict(scenario='Walls',rounds=1),[(p['name'],p['version']) for p in participants])
            write(rows);self.assertEqual(1,audit()['roundTurns'][1])
            bad=copy.deepcopy(rows);bad[1]['turnNumber']=2;write(bad)
            with self.assertRaises(ValueError):audit()
            bad=copy.deepcopy(rows);bad[-1]['results'][0]['totalScore']=101;write(bad)
            with self.assertRaises(ValueError):audit()
            write(rows);raw=bytearray(path.read_bytes());raw[-8]^=1;path.write_bytes(raw)
            with self.assertRaises((gzip.BadGzipFile,EOFError)):audit()

if __name__=='__main__':unittest.main()
