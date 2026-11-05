"""Pure raw metadata/custody/qualified-identity corruption fixtures; no collection or games."""
import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock
import keel_brace_image_identity as i

HISTORY=i.ROOT/'benchmarks/keel-brace-v1/image-probe-evidence-41940635533'


def anchor():
    return dict(policyID=i.POLICY_ID,policySHA256=i.POLICY_SHA,beforeToolsSnapshotSHA256='1'*64,
        preparedSnapshotSHA256='2'*64,preparedWrapperSHA256='3'*64,preparedNumericJobId=7)


def review(ident=None,observed='2026-11-04T10:00:30Z'):
    ident=ident or dict(runId=9,headSHA='a'*40)
    return dict(schemaVersion=1,**{k:v for k,v in anchor().items() if k!='preparedNumericJobId'},
        primaryPregameAuthority='native-only-qualified-observation',finalExternalReconciliationRequired=True,
        earlyHeaderEvidence=dict(status='unavailable-404-deferred',endpoint='https://api.github.com/repos/stephanelee/robocode/actions/jobs/7/logs',
            runId=ident['runId'],runAttempt=1,jobId=7,headSHA=ident['headSHA'],observedAtUTC=observed,httpStatus=404,
            responseBytes=10,responseSHA256='4'*64,responseFile='early-header-response.bin',scope='actual-job-log-request',
            runAccessVerified=True,jobsAccessVerified=True,artifactAccessVerified=True,jobInProgress=True,nativeClaim=None))


def rewrite(path,value):path.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n')


def reseal(folder,report):
    rewrite(folder/'report.json',report)
    status=i.load(folder/'capture-status.json');data=(folder/'report.json').read_bytes()
    row=dict(status['report'],bytes=len(data),suppliedBytes=len(data),sha256=i.sha(data))
    status['report']=row;status['files']=dict(report['rawArtifactIndex'],**{'report.json':row})
    rewrite(folder/'capture-status.json',status)


class Captures(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.base=Path(self.tmp.name)
        self.folder=self.base/'capture';shutil.copytree(HISTORY/'original-members/before-tools',self.folder)

    def test_original_observations_remain_distinct_and_noncanonical(self):
        value=i.verify_capture(self.folder)
        self.assertEqual(value['claims']['setupInfoNative']['version'],'20261025.179.1')
        self.assertEqual(value['claims']['processEnvironment']['ImageVersion'],'20261026.180.1')

    def test_stat_size_spoof_and_missing_index_rejected_even_resealed(self):
        original=i.load(self.folder/'report.json')
        for change in ('stat','index'):
            report=copy.deepcopy(original)
            if change=='stat':
                for key in ('statBefore','openedStat','statAfter'):report['captures']['imagedata'][key]['size']=123456
            else:report['rawArtifactIndex'].pop('raw/imagedata.bin')
            reseal(self.folder,report)
            with self.subTest(change=change),self.assertRaises(ValueError):i.verify_capture(self.folder)

    def test_rehashed_parsed_artifact_cannot_disagree_with_actual_raw(self):
        report=i.load(self.folder/'report.json');parsed=self.folder/'parsed/imagedata.json'
        value=i.load(parsed);value['claim']['version']='unapproved';rewrite(parsed,value);data=parsed.read_bytes()
        row=report['captures']['imagedata']['parsedArtifact'];row.update(bytes=len(data),suppliedBytes=len(data),sha256=i.sha(data))
        report['rawArtifactIndex']['parsed/imagedata.json']=dict(row);reseal(self.folder,report)
        with self.assertRaisesRegex(ValueError,'parser report'):i.verify_capture(self.folder)

    def test_mutation_errors_truncation_discovery_and_parse_faults_rejected(self):
        original=i.load(self.folder/'report.json')
        changes=[lambda r:r['captures']['imagedata'].update(bytesComplete=False),
            lambda r:r['runnerDiscovery'].update(roots=['/x','/y']),
            lambda r:r['runnerDiscovery'].update(truncated=True),
            lambda r:r['runnerDiscovery']['matches'][0].update(errors=[{'fault':True}]),
            lambda r:r['captures']['etc_environment'].update(parseErrors=[{'bad':True}]),
            lambda r:r.update(durationNs=120*10**9+1)]
        for change in changes:
            report=copy.deepcopy(original);change(report);reseal(self.folder,report)
            with self.subTest(change=change),self.assertRaises(ValueError):i.verify_capture(self.folder)

    def test_missing_optional_lsb_must_be_clean_and_appearance_changes_stable(self):
        original=i.verify_capture(self.folder);r=i.load(self.folder/'report.json')
        r['captures']['etc_lsb_release']=dict(requestedPath='/etc/lsb-release',exists=False,
            errors=[dict(operation='lstat',path='/etc/lsb-release',type='FileNotFoundError',message='No such file',messageWasTruncated=False)])
        reseal(self.folder,r);status=i.load(self.folder/'capture-status.json');status['diagnosticErrorCount']=1;rewrite(self.folder/'capture-status.json',status);absent=i.verify_capture(self.folder)
        self.assertEqual(absent['rawSources']['etc_lsb_release'],dict(sourcePath='/etc/lsb-release',absent=True))
        with self.assertRaises(ValueError):i.compare({'stable':original},{'stable':absent})
        r['captures']['etc_lsb_release']['errors'][0]['type']='PermissionError';reseal(self.folder,r)
        with self.assertRaises(ValueError):i.verify_capture(self.folder)

    def test_raw_prefix_or_unknown_native_tuple_rejected(self):
        p=self.folder/'raw/imagedata.bin';data=p.read_bytes();p.write_bytes(data.replace(b'20261025.179.1',b'20261026.180.1'))
        with self.assertRaises(ValueError):i.verify_capture(self.folder)


class Anchors(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.base=Path(self.tmp.name)
        self.context=dict(stage='screen',runId=9,runAttempt=1,headSHA='a'*40,workspace='/actual/work',logicalJobName='registered-stage')
        for phase,collector_phase in [('before-tools','before-tools'),('prepared','after-tools')]:
            folder=self.base/phase/'collector'/collector_phase
            shutil.copytree(HISTORY/'original-members'/collector_phase,folder)
            r=i.load(folder/'report.json');r['context'].update(GITHUB_REPOSITORY='stephanelee/robocode',GITHUB_RUN_ID='9',
                GITHUB_RUN_ATTEMPT='1',GITHUB_JOB='registered-stage',KEEL_BRACE_IMAGE_PROBE_HEAD_SHA='a'*40)
            reseal(folder,r);stable=i.verify_capture(folder);stable['sourceIdentity']=self.context
            value=dict(schemaVersion=1,policyID=i.POLICY_ID,policySHA256=i.POLICY_SHA,phase=phase,**self.context,
                **{k:stable[k] for k in ('bootId','architecture','kernel','kernelVersion','collectorSourceSHA256')},
                snapshotManifestSHA256=i.sha((folder/'capture-status.json').read_bytes()),reportSHA256=i.sha((folder/'report.json').read_bytes()),stable=stable)
            i.write(self.base/phase/'snapshot.json',value)
        self.jobs=dict(jobs=[dict(id=7,name='registered-stage',status='in_progress',run_id=9,run_attempt=1,head_sha='a'*40)])
        i.bind_prepared(self.base,self.jobs)

    def test_all_snapshot_phases_invoke_exact_stock_collector_bounded(self):
        output=self.base/'fresh'
        def collect(command,**kwargs):
            target=Path(command[command.index('--output-dir')+1]);phase=command[-1]
            shutil.copytree(HISTORY/'original-members'/phase,target)
        with mock.patch.object(i,'source_context',return_value=self.context),mock.patch.object(i.subprocess,'run',side_effect=collect) as child:
            value=i.snapshot(output,'screen','before-tools')
        child.assert_called_once_with(['/usr/bin/python3','-I','-B','-S',str(i.ROOT/'tools/keel_brace_image_probe.py'),
            '--output-dir',str(output/'before-tools/collector/before-tools'),'--phase','before-tools'],check=False,timeout=120)
        self.assertEqual(value['phase'],'before-tools')

    def test_actual_own_snapshot_wrapper_graph_identity(self):
        self.assertEqual(i.profile(self.base)['preparedNumericJobId'],7)
        self.assertNotEqual(i.original(self.base,'before-tools')['reportSHA256'],i.original(self.base,'prepared')['reportSHA256'])

    def test_wrapper_numeric_job_policy_extra_key_and_path_mutations_fail(self):
        p=self.base/'prepared-wrapper.json';original=i.load(p)
        for change in ({'preparedNumericJobId':8},{'preparedNumericJobId':True},{'policyID':'other'},
                       {'logicalJobName':'other'},{'jobsAPIFile':'../../other'},{'extra':1}):
            rewrite(p,dict(original,**change))
            with self.subTest(change=change),self.assertRaises(ValueError):i.profile(self.base)

    def test_snapshot_phase_context_and_top_level_disagreement_fail(self):
        p=self.base/'before-tools/snapshot.json';original=i.load(p)
        for change in ({'phase':'prepared'},{'bootId':'other'},{'collectorSourceSHA256':'f'*64},
                       {'runId':True},{'runAttempt':2},{'headSHA':'b'*40},{'logicalJobName':'other'}):
            rewrite(p,dict(original,**change))
            with self.subTest(change=change),self.assertRaises(ValueError):i.original(self.base,'before-tools')

    def test_read_time_pid_and_mtime_are_preserved_not_stable_equality(self):
        a=i.original(self.base,'before-tools');b=i.original(self.base,'prepared');i.compare(a,b)
        b['stable']['rawSources']['imagedata']['mode']+=1
        with self.assertRaises(ValueError):i.compare(a,b)


class Authority(unittest.TestCase):
    def test_only_exact_actual_job_404_deferral_or_matched_header(self):
        ident=dict(runId=9,headSHA='a'*40);value=review();self.assertTrue(i.review(value,anchor(),ident))
        early=value['earlyHeaderEvidence']
        for change in ({'httpStatus':403},{'jobInProgress':False},{'jobId':8},{'runId':True},{'scope':'unknown'},
                       {'nativeClaim':i.NATIVE},{'responseSHA256':'bad'},{'responseFile':'other'}, {'observedAtUTC':'2026-11-04T10:00:00'}):
            mutated=copy.deepcopy(value);mutated['earlyHeaderEvidence'].update(change)
            with self.subTest(change=change),self.assertRaises(ValueError):i.review(mutated,anchor(),ident)
        early.update(status='available-matched',httpStatus=200,scope='complete-startup-header',nativeClaim=i.NATIVE)
        self.assertTrue(i.review(value,anchor(),ident))

    def test_qualification_and_all_snapshot_bindings_mandatory(self):
        value=review();ident=dict(runId=9,headSHA='a'*40)
        for change in ({'schemaVersion':True},{'policyID':'other'},{'finalExternalReconciliationRequired':False},
                       {'primaryPregameAuthority':'canonical-vm'},{'preparedSnapshotSHA256':'f'*64},{'extra':True}):
            with self.subTest(change=change),self.assertRaises(ValueError):i.review(dict(value,**change),anchor(),ident)


class Provider(unittest.TestCase):
    def test_real_original_full_provider_log_not_echoed_workflow(self):
        raw=(HISTORY/'job.log').read_bytes();self.assertEqual(i.sha(raw),'cc9a989d1d4d98914dc331354e7aae3f6cfaea3ef51ec537b4ddf9cbf463f1ce')
        self.assertEqual(i.parse_provider_header(raw),i.NATIVE)
        # A later echoed synthetic Runner Image cannot rescue missing actual startup.
        text=raw.decode('utf-8-sig');header='##[group]Runner Image'
        for changed in (text.replace(header+'\n','##[group]Wrong Image\n',1),
                        text[:text.index('##[endgroup]',text.index(header))],
                        text.replace(header,header+'\nImage: duplicate',1),
                        text.replace('Version: 20261025.179.1','Version: 20261026.180.1',1),
                        text.replace(header,header+'\n##[endgroup]\n'+header,1)):
            with self.subTest(change=changed[:30]),self.assertRaises(ValueError):i.parse_provider_header(changed.encode())

    def test_naked_header_or_missing_real_startup_boundary_rejected(self):
        raw=(HISTORY/'job.log').read_bytes();text=raw.decode('utf-8-sig')
        lines=text.splitlines();start=next(n for n,line in enumerate(lines) if line.endswith('##[group]Runner Image'))
        end=next(n for n in range(start,len(lines)) if lines[n].endswith('##[endgroup]'))
        for bad in ('\n'.join(lines[start:end+1]),'\n'.join(lines[:end+1]),
                    text.replace('Current runner version: ','Invented startup: ',1)):
            with self.assertRaisesRegex(ValueError,'startup'):i.parse_provider_header(bad.encode())

    def fixture(self):
        raw=(HISTORY/'job.log').read_bytes();ident=dict(runId=9,headSHA='a'*40)
        run=dict(id=9,head_sha='a'*40,run_attempt=1,status='completed',conclusion='success')
        steps=[dict(name='Set up job',conclusion='success',started_at='2026-11-05T00:20:51Z',completed_at='2026-11-05T00:20:53Z')]
        for n,name in enumerate(['Capture original qualified image observations before tools',
                'Prepare exact sources/runtime/commands on this worker (zero games)',
                'Finalize immutable zero-game provenance before independent review']):
            steps.append(dict(name=name,conclusion='success',started_at=f'2026-11-05T00:2{1+n}:00Z',completed_at=f'2026-11-05T00:2{1+n}:30Z'))
        job=dict(id=7,run_id=9,run_attempt=1,head_sha='a'*40,name='registered-stage',status='completed',conclusion='success',completed_at='2026-11-05T01:00:00Z',steps=steps)
        jobs=dict(jobs=[job]);evidence={'run':json.dumps(run).encode(),'jobs':json.dumps(jobs).encode(),'log':raw}
        result=dict(schemaVersion=1,status='PASS',**{k:v for k,v in anchor().items() if k!='preparedNumericJobId'},
            **ident,runAttempt=1,jobId=7,nativeClaim=i.NATIVE,completedAtUTC=job['completed_at'],reviewerTask='/root/independent')
        for name,path in [('runAPI','run'),('jobsAPI','jobs'),('jobLog','log')]:
            data=evidence[path];result[name]=dict(path=path,bytes=len(data),sha256=i.sha(data))
        return result,ident,evidence,run,jobs

    @staticmethod
    def windows():
        return {'Capture original qualified image observations before tools':('2026-11-05T00:21:01Z','2026-11-05T00:21:02Z'),
            'Prepare exact sources/runtime/commands on this worker (zero games)':('2026-11-05T00:22:01Z','2026-11-05T00:22:02Z')}

    def test_final_provider_exact_original_and_current_job_bindings(self):
        value,ident,evidence,run,jobs=self.fixture()
        self.assertTrue(i.final_reconciliation(value,anchor(),ident,evidence.__getitem__,run,jobs,Provider.windows()))
        for change in ({'status':'PROVENANCE-PENDING'},{'jobId':8},{'runId':True},{'preparedWrapperSHA256':'f'*64}, {'nativeClaim':{}}):
            with self.subTest(change=change),self.assertRaises(ValueError):i.final_reconciliation(dict(value,**change),anchor(),ident,evidence.__getitem__,run,jobs,Provider.windows())
        bad=copy.deepcopy(jobs);bad['jobs'][0]['steps'][0]['started_at']='2026-11-05T00:20:52Z'
        evidence['jobs']=json.dumps(bad).encode();value['jobsAPI'].update(bytes=len(evidence['jobs']),sha256=i.sha(evidence['jobs']))
        with self.assertRaisesRegex(ValueError,'outside bound'):i.final_reconciliation(value,anchor(),ident,evidence.__getitem__,run,bad,Provider.windows())

    def test_provider_second_resolution_has_exact_recorded_second_interval(self):
        step=dict(started_at='2026-11-05T00:20:52Z',completed_at='2026-11-05T00:20:52Z')
        self.assertTrue(i.within_api_step('2026-11-05T00:20:52.200000Z','2026-11-05T00:20:52.999999Z',step))
        self.assertFalse(i.within_api_step('2026-11-05T00:20:52.200000Z','2026-11-05T00:20:53Z',step))
        step['completed_at']='2026-11-05T00:20:52.100000Z'
        self.assertFalse(i.within_api_step('2026-11-05T00:20:52.200000Z','2026-11-05T00:20:52.300000Z',step))

    def test_missing_full_final_provider_original_cannot_be_waived(self):
        value,ident,evidence,run,jobs=self.fixture();evidence['log']=b'prefix'
        with self.assertRaises(ValueError):i.final_reconciliation(value,anchor(),ident,evidence.__getitem__,run,jobs,Provider.windows())
        value,ident,evidence,run,jobs=self.fixture();run['conclusion']='failure'
        with self.assertRaises(ValueError):i.final_reconciliation(value,anchor(),ident,evidence.__getitem__,run,jobs,Provider.windows())


class ControllerIntegration(unittest.TestCase):
    def test_receipt_child_revalidation_binds_actual_capture_and_prepared_graph(self):
        import keel_brace_experiment as k
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'screen';out.mkdir();profile=dict(imageIdentity=anchor(),identity=dict(runId=9))
            k.save(out/'profile.json',profile)
            def fresh(base,stage,output):
                path=output/'revalidation-123/snapshot.json';k.save(path,{'actual':'fresh synthetic observation'})
                return {'phase':'revalidation-123'}
            with mock.patch.object(k,'ROOT',root),mock.patch.object(k.image_identity,'live',side_effect=fresh),mock.patch.object(k,'runtime_metadata',return_value=profile):
                result=k.revalidate_image_worker('screen',out,profile)
            self.assertEqual(result['snapshotFile'],'image-identity-revalidations/revalidation-123/snapshot.json')
            self.assertEqual(result['snapshotSHA256'],k.benchmark.digest(out/result['snapshotFile']))
            self.assertEqual(result['runtimeProfileSHA256'],k.benchmark.digest(out/'profile.json'))
            with mock.patch.object(k,'ROOT',root),mock.patch.object(k.image_identity,'live',return_value={'phase':'revalidation-123'}),mock.patch.object(k,'runtime_metadata',return_value={}):
                with self.assertRaisesRegex(ValueError,'installed graph'):k.revalidate_image_worker('screen',out,profile)

    def test_confirmation_requires_receipt_bound_original_profile_and_final_raw_provider(self):
        import io,zipfile
        import keel_brace_experiment as k
        value,ident,evidence,run,jobs=Provider().fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);wrapper=b'{"synthetic":"wrapper"}'
            source_ident=dict(**ident,runAttempt=1,bootId='fixture-boot',workspace='/actual/work')
            captures={};snapshot_bytes={}
            for phase,window in zip(('before-tools','prepared'),Provider.windows().values()):
                captures[phase]=json.dumps(dict(utcStart=window[0],utcEnd=window[1])).encode()
                snapshot_bytes[phase]=json.dumps(dict(**source_ident,reportSHA256=i.sha(captures[phase]))).encode()
            before,prepared=snapshot_bytes['before-tools'],snapshot_bytes['prepared']
            actual_anchor=anchor();actual_anchor.update(beforeToolsSnapshotSHA256=i.sha(before),preparedSnapshotSHA256=i.sha(prepared),preparedWrapperSHA256=i.sha(wrapper))
            value.update({key:actual_anchor[key] for key in ('beforeToolsSnapshotSHA256','preparedSnapshotSHA256','preparedWrapperSHA256')})
            profile=dict(identity=dict(**ident,runAttempt=1,bootId='fixture-boot',workspace='/actual/work'),imageIdentity=actual_anchor)
            profile_raw=json.dumps(profile).encode();prefix='build/keel-brace/screen/'
            buffer=io.BytesIO()
            with zipfile.ZipFile(buffer,'w') as archive:
                archive.writestr(prefix+'provenance/profile.json',profile_raw)
                for name,data in [('before-tools/snapshot.json',before),('prepared/snapshot.json',prepared),('prepared-wrapper.json',wrapper)]:
                    archive.writestr(prefix+'provenance/image-identity/'+name,data)
                for phase,collector_phase in [('before-tools','before-tools'),('prepared','after-tools')]:
                    archive.writestr(prefix+'provenance/image-identity/'+phase+'/collector/'+collector_phase+'/report.json',captures[phase])
            raw=buffer.getvalue();(root/'part').write_bytes(raw)
            custody=dict(freezeMember=prefix+'provenance/freeze.json',originalZIP=dict(bytes=len(raw),parts=[dict(path='part',offset=0,bytes=len(raw))]))
            retained={}
            for path,data in evidence.items():(root/path).write_bytes(data);retained[path]=i.sha(data)
            gate=dict(screenRunId=9,retainedEvidenceFiles=retained)
            receipt=dict(approval=dict(profileSHA256=i.sha(profile_raw),bootId='fixture-boot',workspace='/actual/work'))
            report=dict(headSHA=ident['headSHA'],reviewerTask='/root/independent',agentToolProvenance=dict(imageIdentityReconciliation=value))
            with mock.patch.object(k,'ROOT',root):
                self.assertEqual(k.verify_final_screen_provenance(report,gate,custody,receipt,run,jobs['jobs'])['status'],'PASS')
                for mutation in ('reviewer','profile','missing-file'):
                    a=copy.deepcopy(report);r=copy.deepcopy(receipt);g=copy.deepcopy(gate)
                    if mutation=='reviewer':a['reviewerTask']='/root/other'
                    elif mutation=='profile':r['approval']['profileSHA256']='f'*64
                    else:g['retainedEvidenceFiles'].pop('log')
                    with self.subTest(mutation=mutation),self.assertRaises(ValueError):k.verify_final_screen_provenance(a,g,custody,r,run,jobs['jobs'])

    def test_runtime_image_review_cannot_be_missing_or_after_approval(self):
        import keel_brace_experiment as k
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);ident=dict(runId=9,headSHA='a'*40);k.save(out/'profile.json',dict(identity=ident,imageIdentity=anchor()))
            k.save(out/'artifact.json',dict(updated_at='2026-11-04T10:00:00Z'))
            payload=dict(agentToolProvenance=dict(imageIdentityReview=review(ident)))
            report=dict(body=k.REPORT_PREFIX+'\n'+json.dumps(payload),created_at='2026-11-04T10:01:00Z')
            approval=dict(created_at='2026-11-04T10:02:00Z')
            k.validate_image_review(report,out,approval)
            payload['agentToolProvenance'].clear();report['body']=k.REPORT_PREFIX+'\n'+json.dumps(payload)
            with self.assertRaises(ValueError):k.validate_image_review(report,out,approval)
            payload['agentToolProvenance']['imageIdentityReview']=review(ident,'2026-11-04T10:03:00Z');report['body']=k.REPORT_PREFIX+'\n'+json.dumps(payload)
            with self.assertRaises(ValueError):k.validate_image_review(report,out,approval)


if __name__=='__main__':unittest.main()
