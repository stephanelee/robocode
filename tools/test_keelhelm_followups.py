import json
import unittest
from unittest.mock import patch
from pathlib import Path
import tempfile
import io
from types import SimpleNamespace
import keelhelm_melee as core
from fractions import Fraction
import keelhelm_followups as f
import keelhelm_resources as resources

class Contracts(unittest.TestCase):
    def test_invalid_predecessor_cannot_prepare_the_next_serial_stage(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);previous=root/'invalid.json';previous.write_text(json.dumps(dict(stage=2,experimentStatus='invalid',decision='invalid-evidence-no-improvement-claim',nextAcceptedMask=0)))
            args=SimpleNamespace(evidence=root/'new-stage',stage=3,control_mask=0,concurrency=1,load_plan='serial-continuation',previous_audit=previous)
            with patch.object(core,'collect') as collect,self.assertRaisesRegex(ValueError,'invalid stopped predecessor'):
                f.prepare(args)
            collect.assert_not_called()
    def test_serial_schedule_is_fresh_balanced_and_unchanged_in_budget(self):
        serial=f.schedule(2,True);parallel=f.schedule(2)
        self.assertEqual((20,260),(len(serial),sum(g['rounds'] for g in serial)))
        self.assertNotEqual(serial[0]['seed'],parallel[0]['seed'])
        self.assertEqual(2026111100,serial[0]['seed'])
        for i in range(0,20,2):self.assertEqual(serial[i]['seed'],serial[i+1]['seed'])
    def test_group_inspection_failure_still_terminates_and_reaps_known_child(self):
        class Process:
            pid=12345;returncode=None;terminated=False;waited=False
            def poll(self):return self.returncode
            def terminate(self):self.terminated=True;self.returncode=-15
            def kill(self):self.returncode=-9
            def wait(self,timeout):self.waited=True;return self.returncode
        process=Process()
        with tempfile.TemporaryDirectory() as folder:
            run=Path(folder);(run/'helm-metrics.txt').write_text('13 skip 1\n')
            with patch.object(core.subprocess,'Popen',return_value=process),patch.object(core,'owned_group_live',side_effect=PermissionError('ps denied')):
                with self.assertRaisesRegex(ValueError,'Own skipped turn'):core.execute(['unused'],{},run)
            cleanup=json.loads((run/'cleanup.json').read_text())
        self.assertTrue(process.terminated);self.assertTrue(process.waited)
        self.assertFalse(cleanup['verified']);self.assertIn('ps denied',cleanup['validationError'])
    def test_signal_permission_error_does_not_mask_primary_skipped_turn(self):
        class Process:
            pid=12345;returncode=None
            def poll(self):return self.returncode
            def terminate(self):self.returncode=-15
            def kill(self):self.returncode=-9
            def wait(self,timeout):return self.returncode
        process=Process();calls=[]
        def live(group):
            calls.append(group);return [process.pid] if len(calls)==1 else []
        with tempfile.TemporaryDirectory() as folder:
            run=Path(folder);(run/'helm-metrics.txt').write_text('13 skip 1\n')
            with patch.object(core.subprocess,'Popen',return_value=process),patch.object(core,'owned_group_live',side_effect=live),patch.object(core.os,'killpg',side_effect=PermissionError(1,'Operation not permitted')):
                with self.assertRaisesRegex(ValueError,'Own skipped turn'):core.execute(['unused'],{},run)
            cleanup=json.loads((run/'cleanup.json').read_text())
            self.assertTrue(cleanup['verified']);self.assertEqual([],cleanup['remainingPids']);self.assertEqual(1,len(cleanup['signalErrors']))
    def test_setup_failure_is_retained_as_a_failed_slot(self):
        game=f.schedule(1)[0];freeze={'stageConfig':{'controlMask':0,'candidateMask':1}}
        with tempfile.TemporaryDirectory() as folder,patch.object(f,'_play',side_effect=OSError('staging failed')),patch('sys.stdout',new_callable=io.StringIO):
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
