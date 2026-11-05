import json
import unittest
from unittest.mock import patch
from pathlib import Path
import tempfile
from fractions import Fraction
import keelhelm_followups as f
import keelhelm_resources as resources

class Contracts(unittest.TestCase):
    def test_setup_failure_is_retained_as_a_failed_slot(self):
        game=f.schedule(1)[0];freeze={'stageConfig':{'controlMask':0,'candidateMask':1}}
        with tempfile.TemporaryDirectory() as folder,patch.object(f,'_play',side_effect=OSError('staging failed')):
            result=f.play(game,freeze,Path(folder),None)
        self.assertEqual('setup-failed',result['status']);self.assertEqual(game['id'],result['id'])
    def test_paging_gate_uses_elapsed_time_and_three_consecutive_breaches(self):
        previous=dict(monotonic=0,swapouts=0,pageSize=16384);streak=0
        sample=dict(availablePercent=40,swapouts=256,pageSize=16384)
        streak,error=resources.annotate_sample(previous,sample,4,streak)
        self.assertEqual(128,sample['swapoutPagesPer2Seconds']);self.assertIsNone(error)
        for t in (6,8,10):
            previous=sample;sample=dict(availablePercent=40,swapouts=previous['swapouts']+129,pageSize=16384)
            streak,error=resources.annotate_sample(previous,sample,t,streak)
        self.assertIsNotNone(error);self.assertEqual(3,streak)
        previous=sample;sample=dict(availablePercent=40,swapouts=0,pageSize=16384)
        with self.assertRaises(ValueError):resources.annotate_sample(previous,sample,12,streak)
    def test_every_stage_has_twenty_games_and_balanced_parallel_waves(self):
        for stage in range(1,5):
            games=f.schedule(stage)
            self.assertEqual((20,260),(len(games),sum(g['rounds'] for g in games)))
            for i in range(0,20,2):
                a,b=games[i:i+2];self.assertEqual({'control','candidate'},{a['arm'],b['arm']})
                self.assertEqual((a['scenario'],a['seed'],a['order']),(b['scenario'],b['seed'],b['order']))
    def test_resource_parser_counts_only_owned_groups(self):
        r=resources.parse_resources('System-wide memory free percentage: 47%',
            'Mach Virtual Memory Statistics: (page size of 16384 bytes)\nSwapouts: 2912494.\n',
            '10 10 200 30.5\n11 10 100 15.0\n20 20 9999 100.0\n',[10])
        self.assertEqual(47,r['availablePercent']);self.assertEqual(16384,r['pageSize'])
        self.assertEqual(300,r['ownedGroups'][10]['rssKiB']);self.assertEqual(45.5,r['ownedGroups'][10]['cpuLifetimePercent'])
        with self.assertRaises((TypeError,AttributeError)):resources.parse_resources('missing','','',[])
    def test_exact_two_point_gate_and_guard_failures_are_not_rounded(self):
        games=[]
        for scenario in ('melee','Diamond Port','Retcon','Walls','SpinBot'):
            for arm,score in [('control',25),('candidate',27)]:
                games.append(dict(scenario=scenario,arm=arm,helm=dict(totalScore=score,firstPlaces=1),
                    official=dict(results=[dict(totalScore=score),dict(totalScore=100-score)])))
        self.assertEqual('accept',f.summary(games)['decision'])
        games[1]['helm']['totalScore']=26;games[1]['official']['results'][0]['totalScore']=26
        games[1]['official']['results'][1]['totalScore']=74
        self.assertEqual('reject',f.summary(games)['decision'])

if __name__=='__main__':unittest.main()
