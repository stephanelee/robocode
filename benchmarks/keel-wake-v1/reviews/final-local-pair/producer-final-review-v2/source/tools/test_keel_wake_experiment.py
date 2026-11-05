"""Synthetic transport/gate/deadline contracts only: no runtime prep, bots or games."""
import copy
import datetime
import json
import gzip
import io
import signal
import tempfile
import time
from pathlib import Path
import unittest
from unittest import mock
import keel_wake_experiment as k


def marker(candidate=True, values=None):
    values = values or dict(matchedDangerLabels=1, refreshEligiblePlans=1, refreshedWaves=1, densityChangedWaves=1)
    extra = ''.join(' '+name+'='+str(values[name]) for name in k.EXPOSURE) if candidate else ''
    return ['1 start','1 stats shots=1 hitsTaken=0 gun=0 score=0.000/0.000 maxMs=13'+extra,'1 end']


def rows(stage):
    result=[]
    for game in k.stage_plan(stage):
        candidate=game['bot']=='candidate';opp=game['opponent']
        score=(54 if candidate else 51) if opp=='diamond' else (91 if candidate else 92)
        wins=(6 if stage=='screen' else 10) if opp=='diamond' else (8 if opp=='retcon' else 4)
        exposure=dict.fromkeys(k.EXPOSURE,0)
        if candidate and opp=='diamond':exposure=dict.fromkeys(k.EXPOSURE,1)
        result.append(dict(**game,status='valid',skips=0,maxMs=13,ownScore=score,totalScore=100,wins=wins,
                           exposure=exposure if candidate else None))
    return result


class CounterContracts(unittest.TestCase):
    def test_exact_four_and_baseline_na(self):
        self.assertEqual(len(k.EXPOSURE),4)
        self.assertEqual(k.parse_markers(marker(),1,True)[1]['exposure']['refreshedWaves'],1)
        self.assertIsNone(k.parse_markers(marker(False),1,False)[1]['exposure'])
        with self.assertRaises(ValueError):k.parse_markers(marker(),1,False)

    def test_every_missing_duplicate_unknown_or_noninteger_counter_invalid(self):
        for name in k.EXPOSURE:
            for value in ('-1','1.0','true','1000001'):
                bad=[x.replace(name+'=1',name+'='+value) for x in marker()]
                with self.subTest(name=name,value=value),self.assertRaises(ValueError):k.parse_markers(bad,1,True)
            missing=[x.replace(' '+name+'=1','') for x in marker()]
            duplicate=[x+' '+name+'=1' if ' stats ' in x else x for x in marker()]
            for bad in (missing,duplicate):
                with self.assertRaises(ValueError):k.parse_markers(bad,1,True)
        with self.assertRaises(ValueError):k.parse_markers([x+' bogus=0' if ' stats ' in x else x for x in marker()],1,True)

    def test_invariants_and_bool_not_integer(self):
        good=dict(matchedDangerLabels=1,refreshEligiblePlans=2,refreshedWaves=4,densityChangedWaves=3)
        self.assertTrue(k.valid_exposure(good))
        for patch in ({'matchedDangerLabels':0},{'refreshEligiblePlans':0},{'refreshedWaves':5},{'densityChangedWaves':5},{'matchedDangerLabels':True}):
            self.assertFalse(k.valid_exposure(dict(good,**patch)))
        self.assertTrue(k.valid_exposure(dict.fromkeys(k.EXPOSURE,0)))

    def test_live_first_overflow_skip_and_missing_timing_reject(self):
        for bad in (['1 start','1 diagnosticOverflow'],['1 start','1 skip 7'],
                    [x.replace('maxMs=13','maxMs=75') for x in marker()],marker()[:1]+['1 end']):
            with self.assertRaises(ValueError):k.parse_markers(bad,1,True,complete=False)


class ExactGates(unittest.TestCase):
    def decision(self,stage,data):return k.decide(stage,data,[g['id'] for g in k.stage_plan(stage)])
    def test_registered_populations_and_fresh_ids(self):
        for stage,count,budget,seed,port in [('screen',14,120,930001,33711),('confirmation',18,300,940001,33811)]:
            data=k.stage_plan(stage);self.assertEqual((len(data),sum(x['rounds'] for x in data)),(count,budget))
            self.assertEqual((data[0]['seed'],data[0]['port']),(seed,port))
    def test_screen_and_confirmation_exact_gates(self):
        self.assertEqual(self.decision('screen',rows('screen'))['decision'],'progress; confirmation open')
        result=self.decision('confirmation',rows('confirmation'))
        self.assertEqual(result['decision'],'confirmed')
        self.assertTrue(result['checks']['diamondWinsAtLeast60'])
    def test_plus2pp_boundary_is_inclusive_and_less_fails(self):
        data=rows('screen')
        for row in data:
            if row['bot']=='candidate' and row['opponent']=='diamond':row['ownScore']=53
        self.assertTrue(self.decision('screen',data)['checks']['diamondDeltaAtLeast2pp'])
        data[1]['ownScore']=52
        self.assertFalse(self.decision('screen',data)['checks']['diamondDeltaAtLeast2pp'])
    def test_half_is_not_above_half_and_consistency_not_mean(self):
        data=rows('screen');d=[r for r in data if r['bot']=='candidate' and r['opponent']=='diamond']
        for r,s in zip(d,[90,50,50]):r['ownScore']=s
        result=self.decision('screen',data);self.assertEqual(result['decision'],'reject')
        self.assertFalse(result['checks']['diamondGamesAbove50'])
    def test_only_stage_diamond_changed_density_supplies_exposure(self):
        data=rows('screen')
        for r in data:
            if r['bot']=='candidate' and r['opponent']=='diamond':r['exposure']['densityChangedWaves']=0
            elif r['bot']=='candidate':r['exposure']=dict.fromkeys(k.EXPOSURE,1)
        result=self.decision('screen',data);self.assertEqual(result['decision'],'inconclusive')
        self.assertEqual(result['candidateDiamondDensityChangedWaves'],0)
    def test_integrity_precedes_exposure_or_score(self):
        data=rows('screen');data[1]['exposure']['matchedDangerLabels']=0
        self.assertEqual(self.decision('screen',data)['decision'],'invalid')
        data=rows('screen');data[-1]['status']='invalid'
        self.assertEqual(self.decision('screen',data)['decision'],'invalid')


class Capsule(unittest.TestCase):
    def test_canonical_required_and_only_equal_aliases(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder);(out/'profile.json').write_text('{}')
            with self.assertRaises(ValueError):k.profile_path(out)
            (out/'provenance').mkdir();(out/'provenance/profile.json').write_text('{}')
            self.assertEqual(k.profile_path(out),out/'provenance/profile.json')
            (out/'profile.json').write_text('{"validLooking":"different"}')
            with self.assertRaises(ValueError):k.profile_path(out)
            (out/'profile.json').write_text('{}');(out/'freeze.json').write_text('{}')
            with self.assertRaises(ValueError):k.profile_path(out)
            (out/'provenance/freeze.json').write_text('{}');self.assertTrue(k.profile_path(out).is_file())


class Deadline(unittest.TestCase):
    def test_io_minimum_remaining_complete_job_and_existing_caps(self):
        b=k.RevalidationBudget('unused',100,1000,{'fetchObservations':[]},Path('/tmp'),'x')
        with mock.patch.object(k.time,'monotonic',return_value=270),mock.patch.object(k,'now',return_value=995):
            self.assertEqual(b.timeout(20),5)
        with mock.patch.object(k.time,'monotonic',return_value=281),self.assertRaises(TimeoutError):b.timeout(20)
    def test_real_guard_interrupts_blocking_local_work_not_just_after_return(self):
        @k.complete_revalidation_guard
        def blocked():time.sleep(.4)
        start=time.monotonic()
        with mock.patch.object(k,'REVALIDATION_SECONDS',.03),self.assertRaises(TimeoutError):blocked()
        self.assertLess(time.monotonic()-start,.3)
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL)[0],0)
    def test_end_record_is_after_work_and_includes_reserve(self):
        started=(datetime.datetime.now(datetime.timezone.utc)-datetime.timedelta(seconds=1)).isoformat()
        state={'status':'checks-complete','jobDeadlineUTC':(datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(seconds=2200)).isoformat()}
        result=k.finish_revalidation(state,started,time.monotonic()-1)
        self.assertEqual(result['status'],'valid');self.assertGreater(k.utc(result['checkedAtUTC']),k.utc(started))
        self.assertGreaterEqual(result['remainingJobSecondsAtEnd'],2100)
        for deadline in (-1,2099):
            state={'status':'checks-complete','jobDeadlineUTC':(datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(seconds=deadline)).isoformat()}
            with self.assertRaises(TimeoutError):k.finish_revalidation(state,started,time.monotonic()-1)
    def test_guard_uses_owned_group_and_kills_on_timeout(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder);process=mock.Mock(pid=999999)
            process.wait.side_effect=[k.subprocess.TimeoutExpired('synthetic-only',179.5),-9]
            with mock.patch.object(k.subprocess,'Popen',return_value=process) as launch,mock.patch.object(k.os,'killpg') as kill:
                with self.assertRaises(TimeoutError):k.revalidate_authority('screen',out,'before-receipt')
            self.assertTrue(launch.call_args.kwargs['start_new_session']);kill.assert_called_once_with(999999,signal.SIGKILL)
            record=k.load(out/'approval-revalidations/before-receipt.json');self.assertEqual(record['status'],'invalid')
    def test_metadata_and_game_environment_credentials_do_not_mix(self):
        with mock.patch.dict(k.os.environ,{'GH_TOKEN':'fixture-secret','GITHUB_TOKEN':'never-child','JAVA_TOOL_OPTIONS':'bad','PYTHONPATH':'bad'}):
            self.assertNotIn('GH_TOKEN',k.clean_env());self.assertNotIn('GITHUB_TOKEN',k.metadata_worker_environment())
            self.assertNotIn('PYTHONPATH',k.metadata_worker_environment());self.assertNotIn('JAVA_TOOL_OPTIONS',k.metadata_worker_environment())


class Recording(unittest.TestCase):
    def fixture(self, data, duplicate=False, complete=True):
        setup=dict(gameType='1v1', numberOfRounds=1, turnTimeout=100000, arenaWidth=800, arenaHeight=600)
        participants=[dict(id=1,name='Keel',version='0.1.0'),dict(id=2,name='Diamond Port',version='1.8.28')]
        result=[dict(**participants[0], totalScore=60,firstPlaces=1),dict(**participants[1],totalScore=40,firstPlaces=0)]
        tick=dict(type='TickEventForObserver',roundNumber=1,turnNumber=1,events=[
            dict(type='BulletFiredEvent',bullet=dict(ownerId=1,power=.1)),
            dict(type='BulletFiredEvent',bullet=dict(ownerId=2,power=1.2)),
            dict(type='BulletHitBotEvent',bullet=dict(ownerId=1),victimId=2,damage=.4)])
        events=[dict(type='GameStartedEventForObserver',gameSetup=setup,participants=participants),tick]
        if duplicate:events.append(tick)
        if complete:events += [dict(type='RoundEndedEventForObserver',roundNumber=1,turnNumber=1,results=result),
            dict(type='GameEndedEventForObserver',numberOfRounds=1,results=result)]
        with gzip.open(data/'fixture.battle.gz','wt') as f:
            for e in events:f.write(json.dumps(e)+'\n')
        (data/'markers.txt').write_text('\n'.join(marker(False))+'\n')

    def test_official_counts_and_partial_physical_budget(self):
        g=dict(rounds=1,bot='baseline',opponent='diamond',seed=1)
        p=dict(name='Keel',version='0.1.0',opponentName='Diamond Port',opponentVersion='1.8.28')
        with tempfile.TemporaryDirectory() as tmp:
            data=Path(tmp);self.fixture(data)
            m=k.recording_metrics(data,g,p)
            self.assertEqual((m['wins'],m['shots'],m['hits'],m['enemyShots']), (1,1,1,1))
            self.assertEqual(m['share'],.6);self.assertEqual(m['perRound'][0]['damageDealt'],.4)
            self.fixture(data,duplicate=True)
            with self.assertRaises(ValueError):k.recording_metrics(data,g,p)
            self.fixture(data,complete=False)
            partial=k.partial_observation(data)
            self.assertEqual(partial['observedPhysicalRounds'],1)
            self.assertEqual(partial['completedPhysicalRounds'],0)
            with self.assertRaises(ValueError):k.recording_metrics(data,g,p)



class HTTPResponses(unittest.TestCase):
    class Response(io.BytesIO):
        def __init__(self, body, status=200):super().__init__(body);self.code=status
        def getcode(self):return self.code
    def perform(self,status,body=b'{"message":"Not Found"}',allow=False):
        folder=tempfile.TemporaryDirectory();self.addCleanup(folder.cleanup);out=Path(folder.name)
        state={'fetchObservations':[]}
        budget=k.RevalidationBudget(datetime.datetime.now(datetime.timezone.utc).isoformat(),time.monotonic(),time.time()+4000,state,out,'before-receipt')
        with mock.patch.object(k,'_ACTIVE_CHECK',budget),mock.patch.dict(k.os.environ,{'GH_TOKEN':'fixture-only'}),mock.patch.object(k.urllib.request,'urlopen',return_value=self.Response(body,status)) as request:
            if status==404 and not allow:
                with self.assertRaises(ValueError):k.api('git/ref/tags/experiment-receipts/keel-wake-v1-screen')
            else:k.api('git/ref/tags/experiment-receipts/keel-wake-v1-screen',allow_missing=allow)
        return out,state,request
    def test_expected_404_retains_complete_body_and_true_transport(self):
        out,state,request=self.perform(404,allow=True);row=state['fetchObservations'][0]
        self.assertEqual(set(row),{'endpoint','method','fetchStartedAtUTC','fetchCompletedAtUTC','httpStatus','responseFile','responseBytes','responseSHA256','transportComplete','error'})
        self.assertEqual((row['httpStatus'],row['transportComplete'],row['error']),(404,True,None))
        self.assertEqual((out/row['responseFile']).read_bytes(),b'{"message":"Not Found"}')
        self.assertLessEqual(request.call_args.kwargs['timeout'],20)
    def test_404_after_consumption_is_never_accepted(self):
        _,state,_=self.perform(404,allow=False);self.assertEqual(state['fetchObservations'][0]['httpStatus'],404)
    def test_response_cap_failure_keeps_partial_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            state={'fetchObservations':[]};out=Path(folder)
            budget=k.RevalidationBudget('fixture',time.monotonic(),time.time()+4000,state,out,'before-receipt')
            with mock.patch.object(k,'_ACTIVE_CHECK',budget),mock.patch.dict(k.os.environ,{'GH_TOKEN':'fixture-only'}),mock.patch.object(k,'API_MAX_BYTES',3),mock.patch.object(k.urllib.request,'urlopen',return_value=self.Response(b'12345')):
                with self.assertRaises(ValueError):k.api('pulls/119')
            row=state['fetchObservations'][0];self.assertFalse(row['transportComplete']);self.assertIsNotNone(row['error'])
            self.assertEqual((out/row['responseFile']).read_bytes(),b'1234')


class ApprovalWindow(unittest.TestCase):
    def test_initial_cutoff_retains_one_hundred_ns_exact_fractional_authority(self):
        expected={name:'a'*64 if name.endswith('SHA256') else 'fixture' for name in k.APPROVAL_KEYS}
        expected.update(schemaVersion=1,repository=k.REPOSITORY,issue=118,stage='screen',verdict='CLEAR',headSHA='a'*40,runId=9,runAttempt=1,artifactId=20)
        finalized='2027-01-15T08:00:00.0000000Z'
        for suffix,wanted in [('0000000',True),('0000001',False),('0000009',False)]:
            stamp='2027-01-15T08:30:00.'+suffix+'Z'
            row=dict(user={'login':'stephanelee'},body=k.APPROVAL_PREFIX+'\n'+json.dumps(expected),created_at=stamp,updated_at=stamp)
            with mock.patch.object(k,'now',return_value=k.utc('2027-01-15T09:00:00Z')):
                self.assertIs(k.validate_approval(row,expected,'stephanelee',finalized),wanted)

    def test_future_created_and_storage_hour_bounds_do_not_round_away_one_hundred_ns(self):
        expected={name:'a'*64 if name.endswith('SHA256') else 'fixture' for name in k.APPROVAL_KEYS}
        expected.update(schemaVersion=1,repository=k.REPOSITORY,issue=118,stage='screen',verdict='CLEAR',headSHA='a'*40,runId=9,runAttempt=1,artifactId=20)
        stamp='2027-01-15T08:10:00.0000001Z'
        row=dict(user={'login':'stephanelee'},body=k.APPROVAL_PREFIX+'\n'+json.dumps(expected),created_at=stamp,updated_at=stamp)
        with mock.patch.object(k,'now',return_value=k.resource_lower('2027-01-15T08:10:00.0000000Z')):
            self.assertFalse(k.validate_approval(row,expected,'stephanelee','2027-01-15T08:00:00.0000000Z'))
        record=dict(host='fixture',volume='/fixture',freeBytes=8*k.GIB,toolEvidence={'tool':'synthetic'},observedAtUTC='2027-01-15T08:00:00.0000000Z')
        self.assertEqual(k.storage_preflight(record,'2027-01-15T08:00:01.0000000Z','2027-01-15T09:00:00.0000000Z'),record)
        record['observedAtUTC']='2027-01-15T07:59:59.9999999Z'
        with self.assertRaises(ValueError):k.storage_preflight(record,'2027-01-15T08:00:01.0000000Z','2027-01-15T09:00:00.0000000Z')
        with mock.patch.object(k.time,'time_ns',return_value=1_800_000_000_000_000_001):
            self.assertEqual(k.now(),k.Fraction(1_800_000_000_000_000_001,1_000_000_000))

    def test_initial_wait_uses_exact_absolute_cutoff_before_any_poll(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);out=root/'screen';out.mkdir();(out/'provenance').mkdir()
            k.save(out/'provenance/profile.json',{'identity':{'runId':9}});k.save(out/'dispatch-assignment.json',{'reviewerAccount':'stephanelee'})
            artifact=dict(name='keel-wake-v1-screen-provenance-9',updated_at='2027-01-15T08:00:00.0000001Z')
            expected={key:'fixture' for key in k.APPROVAL_KEYS}
            with mock.patch.object(k,'BUILD',root),mock.patch.object(k,'artifact_identity',return_value=artifact),mock.patch.object(k,'expected_approval',return_value=expected),mock.patch.object(k,'now',return_value=k.resource_lower('2027-01-15T08:30:00.0000000Z')),mock.patch.object(k,'comments',side_effect=RuntimeError('entered exact pre-cutoff poll')) as poll:
                with self.assertRaisesRegex(RuntimeError,'entered exact pre-cutoff'):k.wait_approval('screen',1,'a'*64)
                self.assertEqual(poll.call_args.kwargs['deadline'],k.resource_lower('2027-01-15T08:30:00.0000001Z'))
            artifact['updated_at']='2027-01-15T08:00:00.0000000Z'
            with mock.patch.object(k,'BUILD',root),mock.patch.object(k,'artifact_identity',return_value=artifact),mock.patch.object(k,'expected_approval',return_value=expected),mock.patch.object(k,'now',return_value=k.resource_lower('2027-01-15T08:30:00.0000001Z')),mock.patch.object(k,'comments') as poll:
                with self.assertRaises(TimeoutError):k.wait_approval('screen',1,'a'*64)
                poll.assert_not_called()

    def test_initial_window_not_renewed_or_reapplied_to_late_stage(self):
        expected={name:'a'*64 if name.endswith('SHA256') else 'fixture' for name in k.APPROVAL_KEYS}
        expected.update(schemaVersion=1,repository=k.REPOSITORY,issue=118,stage='screen',verdict='CLEAR',headSHA='a'*40,runId=9,runAttempt=1,artifactId=20)
        row=dict(user={'login':'stephanelee'},body=k.APPROVAL_PREFIX+'\n'+json.dumps(expected),created_at='2026-11-04T10:02:00Z',updated_at='2026-11-04T10:02:00Z')
        finalized=k.utc('2026-11-04T10:00:00Z')
        with mock.patch.object(k,'now',return_value=k.utc('2026-11-04T13:00:00Z')):
            self.assertTrue(k.validate_approval(row,expected,'stephanelee',finalized))
            late=dict(row,created_at='2026-11-04T10:30:01Z',updated_at='2026-11-04T10:30:01Z')
            self.assertFalse(k.validate_approval(late,expected,'stephanelee',finalized))
        with mock.patch.object(k,'now',return_value=k.utc('2026-11-04T10:01:00Z')):
            self.assertFalse(k.validate_approval(row,expected,'stephanelee',finalized))




class CompilerProof(unittest.TestCase):
    def test_compile_only_command_is_frozen_and_readonly_recheck_detects_mutation(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);out=root/'stage';out.mkdir();(root/'tools').mkdir();(root/'tools/KeelWakeBattle.java').write_text('fixture source')
            java=root/'jdk';(java/'bin').mkdir(parents=True);(java/'bin/javac').write_bytes(b'compiler fixture');runner=root/'runner.jar';runner.write_bytes(b'classpath fixture')
            locations=dict(java25=str(java),runner=str(runner))
            def compile_only(argv,**kwargs):
                self.assertEqual(argv[1],'-proc:none');self.assertEqual(Path(argv[0]).name,'javac')
                (Path(argv[argv.index('-d')+1])/'KeelWakeBattle.class').write_bytes(b'diagnostic class fixture')
                return type('Result',(),{'returncode':0})()
            with mock.patch.object(k,'ROOT',root),mock.patch.object(k.subprocess,'check_output',return_value=b'javac 25.0.5\n'),mock.patch.object(k.subprocess,'run',side_effect=compile_only) as run:
                proof=k.launcher_compilation(out,locations,collect=True)
                self.assertEqual(k.launcher_compilation(out,locations),proof)
                self.assertEqual(run.call_count,1)
                (out/'launcher-compilation/classes/KeelWakeBattle.class').write_bytes(b'mutated')
                with self.assertRaisesRegex(ValueError,'compile proof'):k.launcher_compilation(out,locations)
                self.assertEqual(run.call_count,1)

class ReviewRegressions(unittest.TestCase):
    def test_remaining_seconds_keeps_actual_api_wrapper_for_revalidation(self):
        jobs=dict(total_count=1,jobs=[dict(id=7,name='registered-stage',status='in_progress',started_at='2026-11-04T10:00:00Z')])
        with mock.patch.dict(k.os.environ,{'GITHUB_RUN_ID':'9'}),mock.patch.object(k,'api',return_value=jobs),mock.patch.object(k,'now',return_value=k.utc('2026-11-04T10:01:00Z')):
            remain,current_jobs=k.remaining_seconds()
        self.assertEqual(remain,340*60-60)
        self.assertEqual([j['id'] for j in current_jobs['jobs']],[7])
    def test_counter_wire_requires_canonical_ascii_unsigned(self):
        for token in ('00','01','+0','-0','１','١'):
            rows=[row.replace('densityChangedWaves=1','densityChangedWaves='+token) for row in marker()]
            with self.subTest(token=token),self.assertRaises(ValueError):k.parse_markers(rows,1,True)
    def test_drip_body_cannot_reset_registered_total_request_cap(self):
        clock=[0.0]
        class Drip:
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def getcode(self):return 200
            def read(self,size):clock[0]+=7;return b' '
        with mock.patch.dict(k.os.environ,{'GH_TOKEN':'synthetic'}),mock.patch.object(k.urllib.request,'urlopen',return_value=Drip()),mock.patch.object(k.time,'monotonic',side_effect=lambda:clock[0]):
            with self.assertRaisesRegex(TimeoutError,'Absolute request body'):k.api('synthetic')
        self.assertEqual(clock[0],21)
    def test_cold_home_or_existing_markers_refused_before_process_creation(self):
        for mutation in ('home','marker','model'):
            with tempfile.TemporaryDirectory() as folder:
                out=Path(folder);data=out/'games'/'wake-s01';data.mkdir(parents=True);home=out/'home';home.mkdir()
                target=home/'model' if mutation=='home' else data/('markers.txt' if mutation=='marker' else 'ais-model.dat');target.write_bytes(b'existing')
                with mock.patch.object(k,'approved_game',return_value={'ownWritableHome':str(home)}),mock.patch.object(k.subprocess,'Popen') as launch:
                    with self.assertRaisesRegex(ValueError,'Cold'):k.execute_game({'id':'wake-s01'},out)
                    launch.assert_not_called()
    def test_targeted_revocation_removes_authority_and_never_grants_it(self):
        expected=dict(repository=k.REPOSITORY,issue=118,stage='screen',headSHA='a'*40,runId=9,runAttempt=1,bootId='boot',artifactId=4,reviewerTask='/independent')
        approval=dict(id=7,created_at='2026-11-04T10:00:00Z')
        obj=dict(schemaVersion=1,verdict='REVOKE',**expected,approvalCommentId=7,reason='Changed source')
        row=dict(user={'login':'stephanelee'},body=k.REVOCATION_PREFIX+'\n'+json.dumps(obj),created_at='2026-11-04T10:00:02Z')
        with self.assertRaisesRegex(ValueError,'explicitly revoked'):k.revocation([row],expected,approval,'stephanelee')
        self.assertIsNone(k.revocation([dict(row,user={'login':'other'})],expected,approval,'stephanelee'))
        obj['unknown']=1;row['body']=k.REVOCATION_PREFIX+'\n'+json.dumps(obj)
        with self.assertRaisesRegex(ValueError,'Malformed'):k.revocation([row],expected,approval,'stephanelee')

if __name__=='__main__':unittest.main()
