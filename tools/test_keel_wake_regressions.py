"""Retained authority/transport/teardown regression fixtures; no engine/runtime preparation."""
import copy
import gzip
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import keel_wake_experiment as k

class Approval(unittest.TestCase):
    def setUp(self):
        self.expected = {name: 'a' * 64 if name.endswith('SHA256') else 'value' for name in k.APPROVAL_KEYS}
        self.expected.update(schemaVersion=1, verdict='CLEAR', repository=k.REPOSITORY, issue=118,
            stage='screen', headSHA='a' * 40, runId=20, runAttempt=1, artifactId=30)
        self.row = dict(body=k.APPROVAL_PREFIX + '\n' + json.dumps(self.expected),
            user={'login': 'stephanelee'}, created_at='2026-11-04T10:01:00Z', updated_at='2026-11-04T10:01:00Z', id=40)
        self.finalized = k.utc('2026-11-04T10:00:00Z')


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
        self.prefix='build/keel-wake/screen/';self.decision=getattr(self,'decision_bytes',b'{"decision":"fixture-not-a-game"}\n')
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


class StorageAndExport(unittest.TestCase):
    def test_owned_transport_paths_are_regular_files_not_symlink_aliases(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'part').write_bytes(b'original');(root/'alias').symlink_to(root/'part')
            with mock.patch.object(k,'ROOT',root):
                self.assertEqual(k.retained_path('part'),(root/'part').resolve())
                with self.assertRaisesRegex(ValueError,'symlink'):k.retained_path('alias')

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
                artifact=dict(id=30,name='keel-wake-v1-screen-evidence-9',digest='sha256:'+'b'*64,
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

if __name__=="__main__":unittest.main()
