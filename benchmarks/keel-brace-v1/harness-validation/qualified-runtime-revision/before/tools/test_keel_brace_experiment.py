"""Pure transport/recording/gate fixtures; no server, bots or runtime preparation."""
import copy
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import keel_brace_experiment as k


def markers(candidate=False):
    suffix = ' ' + ' '.join(n + '=1' for n in k.EXPOSURE) if candidate else ''
    return ['1 start', '1 stats shots=1 hitsTaken=0 gun=0 score=0.000/0.000 maxMs=13' + suffix, '1 end']


class Markers(unittest.TestCase):
    def test_complete_integer_timing_and_candidate_exposure(self):
        value = k.parse_markers(markers(True), 1, True)
        self.assertEqual(value[1]['maxMs'], 13)
        self.assertEqual(value[1]['exposure']['rawRamEvents'], 1)
        self.assertIsNone(k.parse_markers(markers(), 1, False)[1]['exposure'])

    def test_all_missing_duplicate_skip_and_boundary_records_fail_closed(self):
        cases = [markers()[:1] + ['1 end'], markers() + [markers()[1]],
            [s.replace('maxMs=13', 'maxMs=75') for s in markers()],
            [s.replace('maxMs=13', 'maxMs=74.9') for s in markers()],
            [s.replace('maxMs=13', 'maxMs=-1') for s in markers()],
            ['1 start', '1 skip 3'], markers(True)]
        for rows in cases:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                k.parse_markers(rows, 1, False)
        with self.assertRaises(ValueError):
            k.parse_markers(markers(), 1, True)
        self.assertTrue(k.parse_markers(['1 start'], 1, False, complete=False) == {})


class Approval(unittest.TestCase):
    def setUp(self):
        self.expected = {name: 'a' * 64 if name.endswith('SHA256') else 'value' for name in k.APPROVAL_KEYS}
        self.expected.update(schemaVersion=1, verdict='CLEAR', repository=k.REPOSITORY, issue=114,
            stage='screen', headSHA='a' * 40, runId=20, runAttempt=1, artifactId=30)
        self.row = dict(body=k.APPROVAL_PREFIX + '\n' + json.dumps(self.expected),
            user={'login': 'stephanelee'}, created_at='2026-11-04T10:01:00Z', updated_at='2026-11-04T10:01:00Z', id=40)
        self.finalized = k.utc('2026-11-04T10:00:00Z')

    def test_actual_expected_payload_has_exact20keys(self):
        identity = dict(headSHA='a' * 40, runId=20, bootId='boot', workspace='/work', pullNumber=115)
        def fake_load(path):
            if path.name == 'profile.json':return {'identity': identity}
            if path.name == 'dispatch-assignment.json':return {'reviewerTask': '/root/reviewer'}
            return {'body': 'exact dispatch assignment body'}
        with mock.patch.object(k, 'load', side_effect=fake_load), mock.patch.object(k.benchmark, 'digest', return_value='a' * 64):
            result = k.expected_approval('screen', Path('/tmp/out'), {'id':30,'digest':'sha256:'+'b'*64})
        self.assertEqual(set(result), k.APPROVAL_KEYS)
        self.assertEqual(len(result),20)

    def test_exact20key_bound_approval(self):
        self.assertEqual(len(self.expected), 20)
        self.assertTrue(k.validate_approval(self.row, self.expected, 'stephanelee', self.finalized))

    def test_stale_wrong_identity_author_extra_keys_and_bool_fail(self):
        for field in ('headSHA', 'bootId', 'runId', 'stage', 'artifactDigestSHA256'):
            row = copy.deepcopy(self.row);value = copy.deepcopy(self.expected);value[field] = 'wrong'
            row['body'] = k.APPROVAL_PREFIX + '\n' + json.dumps(value)
            self.assertFalse(k.validate_approval(row, self.expected, 'stephanelee', self.finalized))
        row = copy.deepcopy(self.row);row['user']['login'] = 'other'
        self.assertFalse(k.validate_approval(row, self.expected, 'stephanelee', self.finalized))
        self.assertFalse(k.validate_approval(self.row, self.expected, 'stephanelee', k.utc(self.row['created_at'])))
        value = dict(self.expected, unknown=1);row['body'] = k.APPROVAL_PREFIX + '\n' + json.dumps(value)
        self.assertFalse(k.validate_approval(row, self.expected, 'other', self.finalized))
        value = dict(self.expected, schemaVersion=True);row['body'] = k.APPROVAL_PREFIX + '\n' + json.dumps(value)
        self.assertFalse(k.validate_approval(row, value, 'other', self.finalized))

    def test_duplicate_json_key_nonfinite_and_trailing_text_rejected(self):
        for text in ('{"x":1,"x":2}', '{"x":NaN}', '{} trailing'):
            with self.assertRaises(ValueError):
                k.prefixed(k.APPROVAL_PREFIX + '\n' + text, k.APPROVAL_PREFIX)

    def test_report_and_separate_agent_provenance_are_mandatory(self):
        report = dict(id=39, user={'login': 'stephanelee'}, created_at='2026-11-04T10:00:30Z', updated_at='2026-11-04T10:00:30Z',
            body=k.REPORT_PREFIX + '\n' + json.dumps(dict(approvalPayloadSHA256=k.sha(k.canonical(self.expected)),
                reviewerTask=self.expected['reviewerTask'], reviewReport='Independently downloaded and checked original bytes.',
                agentToolProvenance=dict(reviewerTask=self.expected['reviewerTask'], reportGeneratedBeforeGames=True))))
        self.assertEqual(k.review_report([report], self.row, self.expected, 'stephanelee', self.finalized), report)
        with self.assertRaises(ValueError):
            k.review_report([], self.row, self.expected, 'stephanelee', self.finalized)
        with self.assertRaises(ValueError):
            k.review_report([report, report], self.row, self.expected, 'stephanelee', self.finalized)


class AuthorityActorFiltering(unittest.TestCase):
    def test_foreign_malformed_approval_is_ignored(self):
        fixture=Approval();fixture.setUp()
        row=dict(fixture.row,body=k.APPROVAL_PREFIX+'\n{',user={'login':'unassigned'})
        self.assertFalse(k.validate_approval(row,fixture.expected,'stephanelee',fixture.finalized))

    def test_foreign_malformed_report_does_not_poison_valid_report(self):
        fixture=Approval();fixture.setUp()
        valid=dict(id=39,user={'login':'stephanelee'},created_at='2026-11-04T10:00:30Z',updated_at='2026-11-04T10:00:30Z',
            body=k.REPORT_PREFIX+'\n'+json.dumps(dict(approvalPayloadSHA256=k.sha(k.canonical(fixture.expected)),
                reviewerTask=fixture.expected['reviewerTask'],reviewReport='Synthetic independently checked.',
                agentToolProvenance=dict(reviewerTask=fixture.expected['reviewerTask'],reportGeneratedBeforeGames=True))))
        foreign=dict(valid,body=k.REPORT_PREFIX+'\n{',user={'login':'unassigned'})
        self.assertEqual(k.review_report([foreign,valid],fixture.row,fixture.expected,'stephanelee',fixture.finalized),valid)

    def test_foreign_malformed_dispatch_is_ignored_before_parse(self):
        foreign=dict(body=k.DISPATCH_PREFIX+'\n{',user={'login':'unassigned'})
        with self.assertRaisesRegex(ValueError,'Need exactly one prospective dispatch'):
            k.dispatch_assignment('screen',{'headSHA':'a'*40},[foreign])

    def test_authorized_malformed_authority_still_fails_closed(self):
        fixture=Approval();fixture.setUp()
        for prefix,invoke in (
                (k.APPROVAL_PREFIX,lambda row:k.validate_approval(row,fixture.expected,'stephanelee',fixture.finalized)),
                (k.REPORT_PREFIX,lambda row:k.review_report([row],fixture.row,fixture.expected,'stephanelee',fixture.finalized)),
                (k.DISPATCH_PREFIX,lambda row:k.dispatch_assignment('screen',{'headSHA':'a'*40},[row]))):
            with self.subTest(prefix=prefix),self.assertRaises(json.JSONDecodeError):
                invoke(dict(fixture.row,body=prefix+'\n{'))


class Decision(unittest.TestCase):
    def rows(self, stage):
        rows = []
        for g in k.stage_plan(stage):
            share = .60 if g['opponent'] == 'diamond' and g['bot'] == 'candidate' else .55 if g['opponent'] == 'diamond' else .96 if g['bot'] == 'candidate' else .95
            rows.append(dict(**g, share=share, wins=g['rounds'], maxMs=13, skips=0, status='valid', exposure={name:1 for name in k.EXPOSURE} if g['bot']=='candidate' else None))
        return rows

    def test_complete_screen_and_confirmation_progress(self):
        for stage, expected in [('screen', 'progress; confirmation open'), ('confirmation', 'confirmed')]:
            rows = self.rows(stage)
            self.assertEqual(k.decide(stage, rows, [r['id'] for r in rows])['decision'], expected)

    def test_no_favorable_subset_or_incomplete_budget(self):
        rows = self.rows('screen');ids=[r['id'] for r in rows]
        self.assertEqual(k.decide('screen', rows[:-1], ids)['decision'], 'invalid')
        rows[0]['status'] = 'invalid'
        self.assertEqual(k.decide('screen', rows, ids)['decision'], 'invalid')

    def test_exact_percentage_boundaries_and_strict_majorities(self):
        rows = self.rows('screen');ids = [r['id'] for r in rows]
        for r in rows:
            if r['opponent'] == 'diamond':r['share'] = .53 if r['bot'] == 'candidate' else .50
            if r['opponent'] == 'retcon':r['share'] = .90 if r['bot'] == 'candidate' else .89
        self.assertEqual(k.decide('screen', rows, ids)['decision'], 'progress; confirmation open')
        for r in rows:
            if r['bot'] == 'candidate' and r['opponent'] == 'diamond':r['share'] = .50
        self.assertEqual(k.decide('screen', rows, ids)['decision'], 'reject')
        rows=self.rows('confirmation');ids=[r['id'] for r in rows]
        for r in rows:
            if r['bot']=='candidate' and r['opponent']=='diamond':r['wins']=10
        self.assertEqual(k.decide('confirmation', rows, ids)['decision'], 'not confirmed')


class FailClosedStage(unittest.TestCase):
    def test_first_invalid_game_stops_and_retains_partial_physical_budget(self):
        plan = [dict(id='s01', rounds=10, bot='baseline', opponent='diamond'),
                dict(id='s02', rounds=10, bot='candidate', opponent='diamond')]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'screen';out.mkdir();(out/'approval.json').write_text('{}')
            for g in plan:(out/'games'/g['id']).mkdir(parents=True)
            with mock.patch.object(k,'BUILD',root), mock.patch.object(k,'stage_plan',return_value=plan), \
                    mock.patch.object(k,'head',return_value='a'*40), mock.patch.object(k,'verify_worker'), \
                    mock.patch.object(k,'consume_receipt'), mock.patch.object(k,'revalidate_authority'), mock.patch.object(k,'worker_storage',return_value={}), mock.patch.object(k,'remaining_seconds',return_value=(3000,[])), \
                    mock.patch.object(k,'execute_game',side_effect=TimeoutError('fixture timeout')) as child, \
                    mock.patch.object(k,'partial_observation',return_value=dict(observedPhysicalRounds=1,completedPhysicalRounds=0)):
                k.run('screen')
            report=json.loads((out/'decision.json').read_text())
            self.assertEqual(child.call_count,1)
            self.assertEqual(report['unplayedIDs'],['s02'])
            self.assertEqual(report['observedPhysicalRounds'],1)
            self.assertEqual(report['completedPhysicalRounds'],0)
            self.assertEqual(report['decision'],'invalid')
            self.assertTrue((out/'games/s01/failure.json').is_file())


class LiveAuthority(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.out = self.root / 'out'
        self.reg = self.root / 'reg'
        self.out.mkdir();self.reg.mkdir()
        self.patches = [mock.patch.object(k, 'ROOT', self.root), mock.patch.object(k, 'REG', self.reg)]
        for patch in self.patches:patch.start();self.addCleanup(patch.stop)
        self.ident = dict(repository=k.REPOSITORY, headSHA='a'*40, runId=9, runAttempt=1,
            bootId='boot', workspace=str(self.root), pullNumber=115)
        self.assignment = dict(schemaVersion=1, repository=k.REPOSITORY, issue=114, stage='screen',
            headSHA='a'*40, reviewerTask='/root/independent', reviewerAccount='stephanelee',
            implementationTasks=['/root/implementation'],
            assignmentToolRecord=dict(reviewerTask='/root/independent', activeBeforeDispatch=True, storagePreflight=dict(host='fixturehost',volume='/fixture',observedAtUTC='2026-11-04T09:00:00Z',freeBytes=8*k.GIB,toolEvidence='synthetic fixture only')))
        self.assignment['assignmentToolRecordSHA256'] = k.sha(k.canonical(self.assignment['assignmentToolRecord']))
        self.dispatch = self.comment(10,k.DISPATCH_PREFIX+'\n'+json.dumps(self.assignment),'2026-11-04T09:00:00Z')
        self.artifact = dict(id=50,name='provenance',digest='sha256:'+'d'*64,size_in_bytes=1000,
            created_at='2026-11-04T10:00:00Z',updated_at='2026-11-04T10:00:00Z',expired=False,
            workflow_run=dict(id=9,head_sha='a'*40))
        k.save(self.out/'profile.json', {'identity':self.ident})
        k.save(self.out/'participant-manifest.json',{});k.save(self.out/'opponent-manifest.json',{})
        k.save(self.reg/'freeze.json',{});k.save(self.reg/'games.json',{})
        k.save(self.out/'dispatch-assignment.json',self.assignment)
        k.save(self.out/'dispatch-assignment-comment.json',self.dispatch)
        self.prepared = dict(game={'id':'s01'},command=['java','pinned-main'],cwd='/work',
            environment={'HOME':'/empty'},inputFiles={},name='Keel',version='0.1.0')
        prepared_path = self.out/'games/s01/prepared-game.json'
        k.save(prepared_path,self.prepared)
        k.save(self.out/'commands.json',[dict(id='s01',preparedGameFile=prepared_path.relative_to(self.root).as_posix(),
            preparedGameSHA256=k.benchmark.digest(prepared_path),preparedGame=self.prepared)])
        k.save(self.out/'artifact.json',self.artifact)
        self.expected = k.expected_approval('screen',self.out,self.artifact)
        self.approval = self.comment(30,k.APPROVAL_PREFIX+'\n'+json.dumps(self.expected),'2026-11-04T10:02:00Z')
        report = dict(approvalPayloadSHA256=k.sha(k.canonical(self.expected)),reviewerTask='/root/independent',
            reviewReport='Checked actual immutable artifacts before games.',
            agentToolProvenance=dict(reviewerTask='/root/independent',reportGeneratedBeforeGames=True))
        self.report = self.comment(20,k.REPORT_PREFIX+'\n'+json.dumps(report),'2026-11-04T10:01:00Z')
        for name,value in [('approval.json',self.expected),('approval-comment.json',self.approval),
                           ('review-report-comment.json',self.report)]:k.save(self.out/name,value)
        k.save(self.out/'accepted-authority.json',k.accepted_authority(self.out,self.artifact,self.expected))
        self.current={10:copy.deepcopy(self.dispatch),20:copy.deepcopy(self.report),30:copy.deepcopy(self.approval)}
        self.pr=dict(state='open',head={'sha':'a'*40},body='KEEL-BRACE-SOURCE-CLEARED '+'a'*40)
        self.run=dict(created_at='2026-11-04T09:30:00Z')
        self.remote_tag=None
        for patch in [mock.patch.object(k,'api',side_effect=self.api),mock.patch.object(k,'comments',side_effect=lambda:list(self.current.values())),
                      mock.patch.object(k,'identity',return_value=self.ident)]:
            patch.start();self.addCleanup(patch.stop)

    def comment(self, ident, body, timestamp):
        return dict(id=ident,node_id='node'+str(ident),url='https://api/comments/'+str(ident),
            issue_url='https://api/issues/114',html_url='https://github/comment/'+str(ident),
            created_at=timestamp,updated_at=timestamp,body=body,user=dict(id=30698997,login='stephanelee',type='User'))

    def api(self,path,*args,**kwargs):
        if path=='actions/artifacts/50':return copy.deepcopy(self.artifact)
        if path.startswith('issues/comments/'):
            value=self.current.get(int(path.rsplit('/',1)[1]))
            if value is None:
                raise k.urllib.error.HTTPError('https://api/'+path,404,'Not Found',{},None)
            return copy.deepcopy(value)
        if path=='actions/runs/9':return copy.deepcopy(self.run)
        if path=='pulls/115':return copy.deepcopy(self.pr)
        if path=='git/ref/tags/experiment-receipts/keel-brace-v1-screen':return {'object':{'type':'tag','sha':'e'*40}}
        if path=='git/tags/'+'e'*40:return copy.deepcopy(self.remote_tag)
        raise AssertionError('Unexpected network fixture path '+path)

    def check(self,phase='before-receipt'):
        return k.revalidate_authority('screen',self.out,phase)

    def test_unchanged_current_authority_and_receipt_chain_pass(self):
        authority=self.check()
        payload=dict(acceptedAuthority=authority,acceptedAuthoritySHA256=k.sha(k.canonical(authority)),approval=self.expected)
        k.save(self.out/'receipt.json',dict(payload=payload,annotatedTag={'sha':'e'*40}))
        self.remote_tag={'message':json.dumps(payload)}
        self.assertEqual(self.check('before-s01'),authority)
        state=k.load(self.out/'approval-revalidations/before-s01.json')
        self.assertEqual(state['status'],'valid');self.assertEqual(len(state['currentComments']),3)

    def test_deleted_edited_wrong_author_and_timestamp_comments_fail(self):
        original = copy.deepcopy(self.current)
        mutations = [lambda: self.current.pop(30),
            lambda: self.current[30].update(body=self.current[30]['body']+' '),
            lambda: self.current[20].update(updated_at='2026-11-04T10:03:00Z'),
            lambda: self.current[10]['user'].update(login='other')]
        for index, mutate in enumerate(mutations):
            self.current=copy.deepcopy(original);mutate()
            with self.subTest(index=index),self.assertRaises((ValueError,k.urllib.error.HTTPError)):
                self.check('mutation-'+str(index))
            self.assertEqual(k.load(self.out/'approval-comment.json'),self.approval)
            self.assertEqual(k.load(self.out/('approval-revalidations/mutation-'+str(index)+'.json'))['status'],'invalid')

    def test_revoked_duplicate_approval_and_changed_assignment_fail(self):
        revoke = {name:self.expected[name] for name in ('schemaVersion','repository','issue','stage','headSHA','runId','runAttempt','bootId','artifactId','reviewerTask')}
        revoke.update(verdict='REVOKE',approvalCommentId=30,reason='Runtime approval withdrawn')
        self.current[40]=self.comment(40,k.REVOCATION_PREFIX+'\n'+json.dumps(revoke),'2026-11-04T10:03:00Z')
        with self.assertRaisesRegex(ValueError,'revoked'):self.check('revoked')
        self.current.pop(40)
        self.current[31]=self.comment(31,self.approval['body'],'2026-11-04T10:03:00Z')
        with self.assertRaisesRegex(ValueError,'ambiguous'):self.check('duplicate-approval')
        self.current.pop(31)
        changed=copy.deepcopy(self.assignment);changed['reviewerTask']='/root/replacement'
        self.current[10]['body']=k.DISPATCH_PREFIX+'\n'+json.dumps(changed)
        with self.assertRaises(ValueError):self.check('changed-assignment')

    def test_artifact_and_current_head_mismatch_fail(self):
        self.artifact['digest']='sha256:'+'f'*64
        with self.assertRaises(ValueError):self.check('artifact-digest')
        self.artifact['digest']='sha256:'+'d'*64;self.artifact['name']='replacement'
        with self.assertRaises(ValueError):self.check('artifact-identity')
        self.artifact['name']='provenance';self.pr['head']['sha']='b'*40
        with self.assertRaises(ValueError):self.check('current-head')

    def test_local_profile_approval_anchor_and_command_mutations_fail(self):
        mutations=[('profile.json',lambda v:v.update(extra='changed')),
            ('approval.json',lambda v:v.update(verdict='REVOKE')),
            ('participant-manifest.json',lambda v:v.update(extra='changed')),
            ('opponent-manifest.json',lambda v:v.update(extra='changed')),
            ('commands.json',lambda v:v[0]['preparedGame']['environment'].update(GH_TOKEN='forbidden')),
            ('accepted-authority.json',lambda v:v.update(approvalPayloadSHA256='f'*64))]
        for index,(name,mutate) in enumerate(mutations):
            path=self.out/name;original=path.read_bytes();value=k.load(path);mutate(value);k.save(path,value)
            with self.subTest(name=name),self.assertRaises(ValueError):self.check('local-'+str(index))
            path.write_bytes(original)
        p=self.out/'games/s01/prepared-game.json';original=p.read_bytes()
        for index,field in enumerate(('command','environment','opponentName')):
            value=k.load(p)
            if field=='command':value[field]=['java','changed-main']
            elif field=='environment':value[field]['HOME']='/foreign-home'
            else:value[field]='unapproved opponent'
            k.save(p,value)
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,'per-game'):
                self.check('prepared-'+str(index))
            p.write_bytes(original)

    def test_changed_prepared_command_cannot_redefine_local_approved_manifest(self):
        path=self.out/'games/s01/prepared-game.json'
        value=k.load(path);value['command']=['java','unapproved-main'];k.save(path,value)
        bindings=k.load(self.out/'commands.json');bindings[0]['preparedGame']=value
        bindings[0]['preparedGameSHA256']=k.benchmark.digest(path);k.save(self.out/'commands.json',bindings)
        with self.assertRaisesRegex(ValueError,'original accepted API approval'):
            k.approved_game(self.out,'s01')

    def test_mutated_receipt_authority_chain_fails(self):
        authority=self.check()
        payload=dict(acceptedAuthority=authority,acceptedAuthoritySHA256=k.sha(k.canonical(authority)),approval=self.expected)
        k.save(self.out/'receipt.json',dict(payload=payload,annotatedTag={'sha':'e'*40}))
        changed=copy.deepcopy(payload);changed['acceptedAuthoritySHA256']='f'*64
        self.remote_tag={'message':json.dumps(changed)}
        with self.assertRaisesRegex(ValueError,'receipt-bound'):self.check('before-s01')


class OwnedTeardown(unittest.TestCase):
    def test_exited_parent_still_signals_live_owned_process_group(self):
        parent=mock.Mock(pid=12345);parent.poll.return_value=0
        calls=[]
        def killpg(pgid,sig):
            calls.append((pgid,sig))
            if sig==0:raise ProcessLookupError
        with mock.patch.object(k.os,'killpg',side_effect=killpg):
            evidence=k.stop_child(parent)
        self.assertTrue(evidence['termSent']);self.assertEqual(calls[0],(12345,15))
        parent.wait.assert_called_once_with(timeout=5)

    def test_owned_descendants_are_killed_after_bounded_term_drain(self):
        parent=mock.Mock(pid=12345);parent.poll.return_value=0
        with mock.patch.object(k.os,'killpg') as kill, mock.patch.object(k.time,'monotonic',side_effect=[0,11]):
            evidence=k.stop_child(parent)
        self.assertTrue(evidence['killSent'])
        self.assertIn(mock.call(12345,9),kill.call_args_list)
        self.assertTrue(all(call.args[0]==12345 for call in kill.call_args_list))


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
        (data/'markers.txt').write_text('\n'.join(markers())+'\n')

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


class NewBraceGuards(unittest.TestCase):
    def rows(self, stage):return Decision().rows(stage)

    def test_exposure_is_stage_local_and_zero_is_inconclusive_without_filtering(self):
        for stage in ('screen', 'confirmation'):
            rows=self.rows(stage);ids=[r['id'] for r in rows]
            for row in rows:
                if row['bot']=='candidate' and row['opponent']=='retcon':
                    row['exposure']={name:0 for name in k.EXPOSURE}
                if row['bot']=='candidate' and row['opponent']=='diamond':row['share']=.1
            result=k.decide(stage,rows,ids)
            self.assertEqual(result['decision'],'inconclusive')
            self.assertFalse(result['checks']['candidateRetconEligibleRamExposure'])
            self.assertFalse(result['checks']['diamondAbove50'])
            self.assertEqual(len(rows),14 if stage=='screen' else 18)
            rows[0]['status']='invalid'
            self.assertEqual(k.decide(stage,rows,ids)['decision'],'invalid')

    def test_retcon_ceiling_cannot_relax_positive_gain(self):
        rows=self.rows('screen')
        for row in rows:
            if row['opponent']=='retcon':row['share']=.995 if row['bot']=='baseline' else 1.0
        result=k.decide('screen',rows,[r['id'] for r in rows])
        self.assertEqual(result['decision'],'reject')
        self.assertFalse(result['checks']['retconDeltaAtLeast1pp'])

    def test_exact_negative_diamond_guards_and_retcon_positive_boundary(self):
        for stage,control,candidate in [('screen',.54,.51),('confirmation',.55,.53)]:
            rows=self.rows(stage)
            for row in rows:
                if row['opponent']=='diamond':row['share']=candidate if row['bot']=='candidate' else control
                if row['opponent']=='retcon':row['share']=.90 if row['bot']=='candidate' else .89
            expected='progress; confirmation open' if stage=='screen' else 'confirmed'
            self.assertEqual(k.decide(stage,rows,[r['id'] for r in rows])['decision'],expected)
            for row in rows:
                if row['opponent']=='diamond' and row['bot']=='candidate':row['share']-=.000001
            self.assertNotEqual(k.decide(stage,rows,[r['id'] for r in rows])['decision'],expected)

    def test_counters_overflow_bad_chain_and_baseline_leak_fail(self):
        for old,new in [('rawRamEvents=1','rawRamEvents=1000001'),
                        ('eligibleRamCorrectedScans=1','eligibleRamCorrectedScans=2')]:
            with self.assertRaises(ValueError):k.parse_markers([s.replace(old,new) for s in markers(True)],1,True)
        with self.assertRaisesRegex(ValueError,'overflow'):
            k.parse_markers(['1 start','1 diagnosticOverflow'],1,True,complete=False)
        with self.assertRaises(ValueError):k.parse_markers(markers(True),1,False)
        rows=self.rows('screen');rows[1]['exposure']['rawRamEvents']=True
        self.assertEqual(k.decide('screen',rows,[r['id'] for r in rows])['decision'],'invalid')

    def test_prospective_storage_timestamp_size_and_real_evidence_guards(self):
        base=dict(host='fixture',volume='/fixture',observedAtUTC='2026-11-04T09:00:00Z',
            freeBytes=8*k.GIB,toolEvidence={'tool':'synthetic df fixture'})
        self.assertEqual(k.storage_preflight(base,'2026-11-04T09:00:01Z','2026-11-04T10:00:00Z'),base)
        for change in ({'freeBytes':8*k.GIB-1},{'freeBytes':True},{'host':''},{'toolEvidence':{}},
                       {'observedAtUTC':'2026-11-04T08:59:59Z'}, {'observedAtUTC':'2026-11-04T09:00:02Z'},
                       {'observedAtUTC':'2026-11-04T09:00:00'}):
            with self.subTest(change=change),self.assertRaises(ValueError):
                k.storage_preflight(dict(base,**change),'2026-11-04T09:00:01Z','2026-11-04T10:00:00Z')
        for timestamp in ('2026-11-04T09:00:00','2026-11-04T09:00:00-04:00'):
            with self.assertRaises(ValueError):k.utc(timestamp)

    def test_worker_storage_aggregate_cap_is_sticky_and_floor_is_exact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'screen';out.mkdir()
            free=mock.Mock(free=20*k.GIB)
            with mock.patch.object(k,'ROOT',root),mock.patch.object(k.shutil,'disk_usage',return_value=free), \
                    mock.patch.object(k,'owned_output_bytes',side_effect=[10,20,5,k.STAGE_MAX_BYTES+1,1]):
                first=k.worker_storage(out,20*k.GIB,'before-preparation')
                self.assertEqual(first['freeBytes'],20*k.GIB)
                k.worker_storage(out);k.worker_storage(out)
                monitor=k.load(out/'storage-monitor.json')
                self.assertEqual(monitor['maximumStageOutputBytes'],20)
                self.assertEqual(monitor['observations'],3)
                with self.assertRaises(ValueError):k.worker_storage(out)
                with self.assertRaises(ValueError):k.worker_storage(out)
                self.assertTrue(k.load(out/'storage-monitor.json')['limitExceeded'])
            with mock.patch.object(k,'ROOT',root),mock.patch.object(k.shutil,'disk_usage',return_value=mock.Mock(free=8*k.GIB-1)), \
                    mock.patch.object(k,'owned_output_bytes',return_value=0):
                with self.assertRaisesRegex(ValueError,'storage admission'):k.worker_storage(out,8*k.GIB)

    def test_runtime_report_unknown_keys_or_edit_are_not_authority(self):
        fixture=Approval();fixture.setUp()
        report=dict(id=39,user={'login':'stephanelee'},created_at='2026-11-04T10:00:30Z',updated_at='2026-11-04T10:00:30Z',
            body=k.REPORT_PREFIX+'\n'+json.dumps(dict(approvalPayloadSHA256=k.sha(k.canonical(fixture.expected)),
                reviewerTask=fixture.expected['reviewerTask'],reviewReport='Synthetic only.',
                agentToolProvenance=dict(reviewerTask=fixture.expected['reviewerTask'],reportGeneratedBeforeGames=True),extra=True)))
        with self.assertRaises(ValueError):k.review_report([report],fixture.row,fixture.expected,'stephanelee',fixture.finalized)
        row=copy.deepcopy(fixture.row);row['updated_at']='2026-11-04T10:02:00Z'
        self.assertFalse(k.validate_approval(row,fixture.expected,'stephanelee',fixture.finalized))

    def test_export_original_api_is_external_and_raw_decision_unchanged(self):
        for size,expected in [(k.ZIP_MAX_BYTES,'valid'),(k.ZIP_MAX_BYTES+1,'incomplete')]:
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);out=root/'screen';out.mkdir();raw=b'{"decision":"reject"}\n';(out/'decision.json').write_bytes(raw)
                ident=dict(runId=9,headSHA='a'*40)
                artifact=dict(id=30,name='keel-brace-v1-screen-evidence-9',digest='sha256:'+'b'*64,
                    size_in_bytes=size,expired=False,workflow_run=dict(id=9,head_sha='a'*40))
                with mock.patch.object(k,'BUILD',root),mock.patch.object(k,'identity',return_value=ident),mock.patch.object(k,'api',return_value=artifact):
                    if expected=='valid':k.finalize_evidence('screen',30,'b'*64)
                    else:
                        with self.assertRaises(ValueError):k.finalize_evidence('screen',30,'b'*64)
                self.assertEqual((out/'decision.json').read_bytes(),raw)
                record=k.load(out/'finalprovider/export-validation.json')
                self.assertEqual(record['status'],expected)
                self.assertEqual(record['rawDecisionSHA256'],k.sha(raw))
                if expected=='valid':self.assertEqual(k.load(out/'finalprovider/evidence-artifact.json'),artifact)

    def test_missing_upload_retains_incomplete_custody_without_new_games(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'screen').mkdir()
            with mock.patch.object(k,'BUILD',root),self.assertRaises(ValueError):k.finalize_evidence('screen')
            self.assertEqual(k.load(root/'screen/finalprovider/export-validation.json')['status'],'incomplete')


class VirtualCustody(unittest.TestCase):
    def setUp(self):
        import io,zipfile,hashlib
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.reg=self.root/'reg';self.reg.mkdir();self.out=self.root/'out';self.out.mkdir()
        self.plan=k.stage_plan('screen')
        self.patch=[mock.patch.object(k,'ROOT',self.root),mock.patch.object(k,'REG',self.reg),
                    mock.patch.object(k,'stage_plan',return_value=self.plan)]
        for p in self.patch:p.start();self.addCleanup(p.stop)
        self.source=b'fixture-source-only\n';self.source_hash=k.sha(self.source)
        self.source_git=hashlib.sha1(b'blob '+str(len(self.source)).encode()+b'\0'+self.source).hexdigest()
        self.freeze=dict(assets={'tools/fake.py':self.source_hash});k.save(self.reg/'freeze.json',self.freeze)
        self.prefix='build/keel-brace/screen/';self.decision=getattr(self,'decision_bytes',b'{"decision":"fixture-not-a-game"}\n')
        payloads={self.prefix+'decision.json':self.decision,self.prefix+'provenance/freeze.json':(self.reg/'freeze.json').read_bytes(),
            self.prefix+'provenance/source/tools/fake.py':self.source,self.prefix+'unrelated-same-bytes.py':self.source}
        self.recordings={g['id']:self.prefix+'games/'+g['id']+'/raw.battle.gz' for g in self.plan}
        payloads.update({p:('synthetic-recording-'+g).encode() for g,p in self.recordings.items()})
        stream=io.BytesIO()
        with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_DEFLATED) as archive:
            for name,b in payloads.items():archive.writestr(name,b)
        raw=stream.getvalue();members={}
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            for info in archive.infolist():
                b=archive.read(info);meta=k.stream_identity(io.BytesIO(b),len(b));meta.update(crc32=info.CRC,
                    compression=info.compress_type,compressedBytes=info.compress_size,flags=info.flag_bits,
                    externalAttr=info.external_attr,timestamp=list(info.date_time),directory=info.is_dir());members[info.filename]=meta
        self.retained={};parts=[];cut=len(raw)//2
        for i,(offset,b) in enumerate([(0,raw[:cut]),(cut,raw[cut:])]):
            relative='kept/part-'+str(i)+'.bin';p=self.root/relative;p.parent.mkdir(exist_ok=True);p.write_bytes(b)
            meta=k.stream_identity(io.BytesIO(b),len(b));parts.append(dict(path=relative,offset=offset,**meta));self.retained[relative]=meta['sha256']
        self.custody=dict(schemaVersion=1,headSHA='a'*40,sourceFreezeSHA256=k.benchmark.digest(self.reg/'freeze.json'),
            originalZIP=dict(artifactId=30,bytes=len(raw),sha256=k.sha(raw),parts=parts),members=members,
            decisionMember=self.prefix+'decision.json',freezeMember=self.prefix+'provenance/freeze.json',
            gameRecordingMembers=self.recordings,sourceBindings={'tools/fake.py':dict(member=self.prefix+'provenance/source/tools/fake.py',
                commit='a'*40,path='tools/fake.py',gitBlobSHA1=self.source_git,sha256=self.source_hash)})
        self.git=mock.patch.object(k.subprocess,'check_output',return_value=self.source_git+'\n');self.git.start();self.addCleanup(self.git.stop)

    def check(self,custody=None):return k.verify_custody(custody or self.custody,self.retained,'a'*40,k.sha(self.decision))

    def test_all_original_members_and_whole_recordings_verify_without_expansion(self):
        members=self.check()
        self.assertEqual(len([n for n in members if n.endswith('.battle.gz')]),14)
        self.assertFalse(list(self.root.rglob('*.battle.gz')))
        reader=k.PartsReader(self.custody['originalZIP']['parts'],self.custody['originalZIP']['bytes'])
        boundary=self.custody['originalZIP']['parts'][0]['bytes'];reader.seek(boundary-3)
        self.assertEqual(len(reader.read(10)),10)
        reader.seek(-7,2);self.assertEqual(len(reader.read()),7)

    def test_gap_corruption_omission_and_metadata_changes_fail(self):
        mutations=[lambda c:c['originalZIP']['parts'][1].update(offset=0),
            lambda c:c['originalZIP'].update(sha256='f'*64),lambda c:c['members'].pop(self.prefix+'decision.json'),
            lambda c:c['members'][self.prefix+'decision.json'].update(crc32=0),
            lambda c:c['originalZIP'].update(bytes=k.ZIP_MAX_BYTES+1)]
        for mutate in mutations:
            c=copy.deepcopy(self.custody);mutate(c)
            with self.subTest(mutation=mutate),self.assertRaises(ValueError):self.check(c)
        path=self.root/self.custody['originalZIP']['parts'][0]['path'];b=bytearray(path.read_bytes());b[0]^=1;path.write_bytes(b)
        with self.assertRaises(ValueError):self.check()

    def test_source_path_identity_and_embedded_freeze_are_exact(self):
        changes=[dict(path='tools/other.py'),dict(member=self.prefix+'unrelated-same-bytes.py'),dict(commit='b'*40)]
        for change in changes:
            c=copy.deepcopy(self.custody);c['sourceBindings']['tools/fake.py'].update(change)
            with self.subTest(change=change),self.assertRaises(ValueError):self.check(c)
        c=copy.deepcopy(self.custody);c['sourceBindings']['extra.py']=copy.deepcopy(c['sourceBindings']['tools/fake.py'])
        with self.assertRaises(ValueError):self.check(c)
        c=copy.deepcopy(self.custody);c['freezeMember']=self.prefix+'decision.json'
        with self.assertRaises(ValueError):self.check(c)

    def test_recording_path_game_binding_cannot_be_reused(self):
        c=copy.deepcopy(self.custody);ids=list(c['gameRecordingMembers']);c['gameRecordingMembers'][ids[1]]=c['gameRecordingMembers'][ids[0]]
        with self.assertRaises(ValueError):self.check(c)
        c=copy.deepcopy(self.custody);c['gameRecordingMembers'].pop(ids[0])
        with self.assertRaises(ValueError):self.check(c)



class ConditionalGate(VirtualCustody):
    def setUp(self):
        rows=Decision().rows('screen')
        for row in rows:row['recordingSHA256']=k.sha(('synthetic-recording-'+row['id']).encode())
        self.decision_bytes=json.dumps(dict(stage='screen',headSHA='a'*40,decision='progress; confirmation open',games=rows)).encode()
        super().setUp()
        direct='kept/decision.json';manifest='kept/custody.json'
        (self.root/direct).write_bytes(self.decision);k.save(self.root/manifest,self.custody)
        self.retained[direct]=k.sha(self.decision);self.retained[manifest]=k.benchmark.digest(self.root/manifest)
        self.gate=dict(schemaVersion=1,sourceFreezeSHA256=k.benchmark.digest(self.reg/'freeze.json'),screenRunId=5,
            receiptTagSHA='e'*40,decisionFile=direct,decisionSHA256=k.sha(self.decision),retainedEvidenceFiles=self.retained,
            custodyManifestFile=manifest,independentOutcomeReviewCommentId=8)
        k.save(self.reg/'screen-evidence.json',self.gate)
        k.save(self.out/'dispatch-assignment.json',dict(implementationTasks=['/root/implementation']))
        self.artifact=dict(id=30,name='keel-brace-v1-screen-evidence-5',digest='sha256:'+self.custody['originalZIP']['sha256'],
            size_in_bytes=self.custody['originalZIP']['bytes'],expired=False,workflow_run=dict(id=5,head_sha='a'*40))
        self.run=dict(status='completed',conclusion='success',head_sha='a'*40,run_attempt=1)
        self.jobs=[dict(name='registered-stage',conclusion='success',completed_at='2026-11-04T11:00:00Z',
            steps=[dict(name='Validate finalized original artifact custody',conclusion='success')])]
        report=dict(verdict='SCREEN-PASS',stage='screen',sourceFreezeSHA256=self.gate['sourceFreezeSHA256'],
            runId=5,headSHA='a'*40,decisionSHA256=self.gate['decisionSHA256'],reviewerTask='/root/independent-outcome',
            agentToolProvenance=dict(reviewerTask='/root/independent-outcome',reviewReport='Synthetic fixture only; no real games.'))
        self.review=dict(user={'login':'stephanelee'},created_at='2026-11-04T12:00:00Z',updated_at='2026-11-04T12:00:00Z',
            body='KEEL_BRACE_OUTCOME_REVIEW_V1\n'+json.dumps(report))
        approval=dict(freezeSHA256=self.gate['sourceFreezeSHA256'],runId=5,runAttempt=1,headSHA='a'*40,issue=k.ISSUE,repository=k.REPOSITORY)
        self.tag=dict(message=json.dumps(dict(stage='screen',approval=approval)))
        def api(path,*args,**kwargs):
            if path=='git/ref/tags/experiment-receipts/keel-brace-v1-screen':return dict(object=dict(sha='e'*40,type='tag'))
            if path=='git/tags/'+'e'*40:return copy.deepcopy(self.tag)
            if path=='actions/runs/5':return copy.deepcopy(self.run)
            if path=='actions/artifacts/30':return copy.deepcopy(self.artifact)
            if path=='actions/runs/5/jobs?filter=latest&per_page=100':return dict(jobs=copy.deepcopy(self.jobs))
            if path=='issues/comments/8':return copy.deepcopy(self.review)
            raise AssertionError('Unexpected network fixture '+path)
        patch=mock.patch.object(k,'api',side_effect=api);patch.start();self.addCleanup(patch.stop)

    def test_full_confirmation_gate_accepts_lossless_virtual_custody(self):
        k.verify_confirmation_gate(self.out)
        retained=k.load(self.out/'screen-gate-verified.json')
        self.assertEqual(retained['originalArtifact']['id'],30)
        self.assertFalse(list(self.root.rglob('*.battle.gz')))

    def test_no_success_without_original_export_and_independent_postcompletion_review(self):
        base_artifact=copy.deepcopy(self.artifact);base_jobs=copy.deepcopy(self.jobs);base_review=copy.deepcopy(self.review)
        changes=[lambda:self.artifact.update(size_in_bytes=self.artifact['size_in_bytes']+1),
            lambda:self.artifact.update(name='provenance-not-final-evidence'),
            lambda:self.jobs[0]['steps'].clear(),lambda:self.jobs[0].update(conclusion='failure'),
            lambda:self.review.update(updated_at='2026-11-04T12:01:00Z'),
            lambda:self.review.update(created_at='2026-11-04T10:59:00Z',updated_at='2026-11-04T10:59:00Z')]
        for change in changes:
            self.artifact=copy.deepcopy(base_artifact);self.jobs=copy.deepcopy(base_jobs);self.review=copy.deepcopy(base_review);change()
            with self.subTest(change=change),self.assertRaises(ValueError):k.verify_confirmation_gate(self.out)
        self.review=base_review;self.jobs=base_jobs;self.artifact=base_artifact;self.run['status']='in_progress'
        with self.assertRaises(ValueError):k.verify_confirmation_gate(self.out)

    def test_inconclusive_original_screen_never_opens_confirmation(self):
        value=k.load(self.root/'kept/decision.json')
        for row in value['games']:
            if row['bot']=='candidate' and row['opponent']=='retcon':row['exposure']={n:0 for n in k.EXPOSURE}
        k.save(self.root/'kept/decision.json',value);self.gate['decisionSHA256']=k.benchmark.digest(self.root/'kept/decision.json')
        self.gate['retainedEvidenceFiles']['kept/decision.json']=self.gate['decisionSHA256'];k.save(self.reg/'screen-evidence.json',self.gate)
        with self.assertRaisesRegex(ValueError,'did not pass'):k.verify_confirmation_gate(self.out)



class ImageProvenanceHold(unittest.TestCase):
    def test_environment_alone_never_establishes_allowed_image(self):
        with self.assertRaisesRegex(ValueError,'environment alone'):
            k.image_consensus('20261026.180.1',{})

    def test_raw_setup_header_conflict_is_fatal_even_when_environment_is_allowed(self):
        with self.assertRaisesRegex(ValueError,'Conflicting'):
            k.image_consensus('20261026.180.1',{'fixtureImageFile':'20261026.180.1'},'20261025.179.1')
        with self.assertRaisesRegex(ValueError,'Conflicting'):
            k.image_consensus('20261026.180.1',{'fixtureImageFile':'20261101.186.1'})

    def test_consistent_unapproved_image_does_not_expand_allowlist(self):
        with self.assertRaisesRegex(ValueError,'Unregistered'):
            k.image_consensus('20261025.179.1',{'fixtureImageFile':'20261025.179.1'},'20261025.179.1')
        self.assertEqual(k.image_consensus('20261026.180.1',{'fixtureImageFile':'20261026.180.1'},'20261026.180.1'),'20261026.180.1')

    def test_real_identity_is_blocked_until_authority_schema_review(self):
        with self.assertRaisesRegex(ValueError,'UNRESOLVED IMAGE PROVENANCE'):k.identity()


class RemainingBoundaries(unittest.TestCase):
    def test_teardown_respects_remaining_hard_deadline_and_retains_reap_error(self):
        parent=mock.Mock(pid=12345)
        with mock.patch.object(k.os,'killpg'),mock.patch.object(k.time,'monotonic',side_effect=[95,95,96,97,98]):
            result=k.stop_child(parent,100)
        parent.wait.assert_called_once_with(timeout=3)
        self.assertTrue(result['killSent']);self.assertTrue(result['hardDeadlineRespected'])
        parent.wait.side_effect=k.subprocess.TimeoutExpired('fixture',5)
        with mock.patch.object(k.os,'killpg',side_effect=ProcessLookupError):
            result=k.stop_child(parent)
        self.assertIn('reapError',result)

    def test_private_contact_counter_is_only_an_upper_bounded_callback_count(self):
        fixture=Recording();g=dict(rounds=1,bot='candidate',opponent='diamond',seed=1)
        prepared=dict(name='Keel',version='0.1.0',opponentName='Diamond Port',opponentVersion='1.8.28')
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);fixture.fixture(p);(p/'markers.txt').write_text('\n'.join(markers(True))+'\n')
            with self.assertRaisesRegex(ValueError,'callbacks exceed'):k.recording_metrics(p,g,prepared)
            raw=p/'fixture.battle.gz'
            with gzip.open(raw,'rt') as stream:events=[json.loads(v) for v in stream]
            events[1]['events'].append(dict(type='BotHitBotEvent',botId=1))
            with gzip.open(raw,'wt') as stream:
                for event in events:stream.write(json.dumps(event)+'\n')
            result=k.recording_metrics(p,g,prepared)
            self.assertEqual(result['exposure']['rawRamEvents'],1)
            self.assertEqual(result['botContacts'],1)
            # A delivered-callback count of0 is permitted for an official contact.
            zero=markers(True)
            for name in k.EXPOSURE:zero=[line.replace(name+'=1',name+'=0') for line in zero]
            (p/'markers.txt').write_text('\n'.join(zero)+'\n')
            result=k.recording_metrics(p,g,prepared)
            self.assertEqual(result['exposure']['rawRamEvents'],0)



if __name__ == '__main__':
    unittest.main()
