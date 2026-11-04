"""Pure transport/recording/gate fixtures; no server, bots or runtime preparation."""
import copy
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import keel_helm_experiment as k


def markers(candidate=False):
    suffix = ' ' + ' '.join(n + '=1' for n in k.EXPOSURE) if candidate else ''
    return ['1 start', '1 stats shots=1 hitsTaken=0 gun=0 score=0.000/0.000 maxMs=13' + suffix, '1 end']


class Markers(unittest.TestCase):
    def test_complete_integer_timing_and_candidate_exposure(self):
        value = k.parse_markers(markers(True), 1, True)
        self.assertEqual(value[1]['maxMs'], 13)
        self.assertEqual(value[1]['exposure']['stoppedRolloutSteps'], 1)
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
        self.expected.update(schemaVersion=1, verdict='CLEAR', repository=k.REPOSITORY, issue=107,
            stage='screen', headSHA='a' * 40, runId=20, runAttempt=1, artifactId=30)
        self.row = dict(body=k.APPROVAL_PREFIX + '\n' + json.dumps(self.expected),
            user={'login': 'stephanelee'}, created_at='2026-11-04T10:01:00Z', id=40)
        self.finalized = k.utc('2026-11-04T10:00:00Z')

    def test_actual_expected_payload_has_exact20keys(self):
        identity = dict(headSHA='a' * 40, runId=20, bootId='boot', workspace='/work', pullNumber=108)
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
        report = dict(id=39, user={'login': 'stephanelee'}, created_at='2026-11-04T10:00:30Z',
            body=k.REPORT_PREFIX + '\n' + json.dumps(dict(approvalPayloadSHA256=k.sha(k.canonical(self.expected)),
                reviewerTask=self.expected['reviewerTask'], reviewReport='Independently downloaded and checked original bytes.',
                agentToolProvenance=dict(reviewerTask=self.expected['reviewerTask'], reportGeneratedBeforeGames=True))))
        self.assertEqual(k.review_report([report], self.row, self.expected, 'stephanelee', self.finalized), report)
        with self.assertRaises(ValueError):
            k.review_report([], self.row, self.expected, 'stephanelee', self.finalized)
        with self.assertRaises(ValueError):
            k.review_report([report, report], self.row, self.expected, 'stephanelee', self.finalized)


class Decision(unittest.TestCase):
    def rows(self, stage):
        rows = []
        for g in k.stage_plan(stage):
            share = .60 if g['opponent'] == 'diamond' and g['bot'] == 'candidate' else .55 if g['opponent'] == 'diamond' else .95
            rows.append(dict(**g, share=share, wins=g['rounds'], maxMs=13, skips=0, status='valid'))
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
            if r['opponent'] == 'retcon':r['share'] = .90 if r['bot'] == 'candidate' else .93
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
                    mock.patch.object(k,'consume_receipt'), mock.patch.object(k,'revalidate_authority'), mock.patch.object(k,'remaining_seconds',return_value=(3000,[])), \
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
            bootId='boot', workspace=str(self.root), pullNumber=108)
        self.assignment = dict(schemaVersion=1, repository=k.REPOSITORY, issue=107, stage='screen',
            headSHA='a'*40, reviewerTask='/root/independent', reviewerAccount='stephanelee',
            implementationTasks=['/root/implementation'],
            assignmentToolRecord=dict(reviewerTask='/root/independent', activeBeforeDispatch=True))
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
        self.pr=dict(state='open',head={'sha':'a'*40},body='KEEL-HELM-SOURCE-CLEARED '+'a'*40)
        self.run=dict(created_at='2026-11-04T09:30:00Z')
        self.remote_tag=None
        for patch in [mock.patch.object(k,'api',side_effect=self.api),mock.patch.object(k,'comments',side_effect=lambda:list(self.current.values())),
                      mock.patch.object(k,'identity',return_value=self.ident)]:
            patch.start();self.addCleanup(patch.stop)

    def comment(self, ident, body, timestamp):
        return dict(id=ident,node_id='node'+str(ident),url='https://api/comments/'+str(ident),
            issue_url='https://api/issues/107',html_url='https://github/comment/'+str(ident),
            created_at=timestamp,updated_at=timestamp,body=body,user=dict(id=30698997,login='stephanelee',type='User'))

    def api(self,path,*args,**kwargs):
        if path=='actions/artifacts/50':return copy.deepcopy(self.artifact)
        if path.startswith('issues/comments/'):
            value=self.current.get(int(path.rsplit('/',1)[1]))
            if value is None:
                raise k.urllib.error.HTTPError('https://api/'+path,404,'Not Found',{},None)
            return copy.deepcopy(value)
        if path=='actions/runs/9':return copy.deepcopy(self.run)
        if path=='pulls/108':return copy.deepcopy(self.pr)
        if path=='git/ref/tags/experiment-receipts/keel-helm-v1-screen':return {'object':{'type':'tag','sha':'e'*40}}
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


if __name__ == '__main__':
    unittest.main()
