"""Complete synthetic producer final-provider reader; no engine or collector calls."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import keel_wake_image_identity as i
import test_keel_wake_image_identity as fixtures


class CompletePhaseReader(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.base=Path(self.temp.name)
        self.sources={p:(i.ROOT/p).read_bytes() for p in (i.WORKFLOW_PATH,i.EMITTER_PATH,i.RAW_COLLECTOR_PATH,i.CONTROLLER_PATH)}
        self.assets={p:i.sha(raw) for p,raw in self.sources.items()}
        self.snapshots={}
        for phase in i.PHASE_MAPPING:
            snapshot=dict(stage='screen',phase=phase,runId=9,runAttempt=1,headSHA='a'*40,collectorSourceSHA256=self.assets[i.RAW_COLLECTOR_PATH])
            folder=self.base/phase;folder.mkdir();i.write(folder/'snapshot.json',snapshot);self.snapshots[phase]=snapshot
        self.plan=i.create_phase_plan(self.base,self.snapshots['before-tools'],self.snapshots['prepared'])
        self.plan_bytes=(self.base/'phase-order-plan.json').read_bytes()
        self.hashes={phase:i.sha((self.base/phase/'snapshot.json').read_bytes()) for phase in i.PHASE_MAPPING}
        self.context=dict(plan=self.plan,planBytes=self.plan_bytes,assets=self.assets,source=self.sources,snapshots=self.snapshots,snapshotHashes=self.hashes)
        self.steps=i.declared_steps(self.sources[i.WORKFLOW_PATH])
        self.job=dict(id=7,name='registered-stage',run_id=9,run_attempt=1,head_sha='a'*40,status='completed',conclusion='success',completed_at='2026-11-04T10:03:00Z',steps=[dict(number=1,name='Set up job',started_at='2026-11-04T10:00:02Z',completed_at='2026-11-04T10:00:04Z',status='completed',conclusion='success')])
        for entry in self.plan['phasePlans']:
            minute='01' if entry['phase']=='before-tools' else '02'
            for field,second,end in [('capture','00','01'),('next','02','04')]:
                self.job['steps'].append(dict(number=entry[field+'StepNumber'],name=entry[field+'StepName'],started_at='2026-11-04T10:'+minute+':'+second+'Z',completed_at='2026-11-04T10:'+minute+':'+end+'Z',status='completed',conclusion='success'))
        self.jobs={'jobs':[self.job]}
        self.log=self.make_log()
        value,ident,refs,run,side,pack=fixtures.ActualReviewer().fixture()
        self.refs=refs;self.ident=ident;self.run=dict(id=9,run_attempt=1,head_sha='a'*40,status='completed',conclusion='success',created_at=run['created_at'])
        self.anchor=fixtures.anchor();self.anchor.update(beforeToolsSnapshotSHA256=self.hashes['before-tools'],preparedSnapshotSHA256=self.hashes['prepared'],phaseOrderPlanSHA256=i.sha(self.plan_bytes))
        self.value=dict(schemaVersion=1,status='PASS',runId=9,runAttempt=1,jobId=7,headSHA='a'*40,reviewerTask=value['reviewerTask'],reviewerIdentityEvidence=value['reviewerIdentityEvidence'],completedAtUTC=self.job['completed_at'],nativeClaim=i.JOINT_CASES[0]['native'],**{k:self.anchor[k] for k in ('policyID','policySHA256','caseID','beforeToolsSnapshotSHA256','preparedSnapshotSHA256','preparedWrapperSHA256')})
        self.windows={entry['captureStepName']:('2026-11-04T10:'+('01' if entry['phase']=='before-tools' else '02')+':00.100000Z','2026-11-04T10:'+('01' if entry['phase']=='before-tools' else '02')+':01.500000Z') for entry in self.plan['phasePlans']}
        self.refresh()

    def make_log(self):
        lines=['2026-11-04T10:00:03.0000001Z '+line for line in fixtures.header(i.CASE_A).decode().splitlines()]
        for entry in self.plan['phasePlans']:
            step=self.steps[entry['captureStepNumber']-2];following=self.steps[entry['nextStepNumber']-2]
            run=step['run'].rstrip('\n').splitlines();stamp='2026-11-04T10:'+('01' if entry['phase']=='before-tools' else '02')+':01.8000001Z '
            lines+=[stamp+'##[group]Run '+run[0]]+[stamp+'\x1b[36;1m'+line+'\x1b[0m' for line in run]+[stamp+'shell: /usr/bin/bash -e {0}',stamp+'##[endgroup]',stamp+i.PHASE_PREFIX+i.compact(entry['markerPayload']).decode(),stamp+'##[group]Run '+following['uses'],stamp+'##[endgroup]']
        return ('\n'.join(lines)+'\n').encode()

    def bind(self,path,raw):
        self.refs[path]=raw;return dict(path=path,bytes=len(raw),sha256=i.sha(raw))

    def refresh(self):
        self.value['runAPI']=self.bind('provider/run.json',i.compact(self.run));self.value['jobsAPI']=self.bind('provider/jobs.json',i.compact(self.jobs));self.value['jobLog']=self.bind('provider/job.log',self.log)
        self.proof=dict(schemaVersion=1,kind=i.PHASE_KIND,phaseOrderPlanSHA256=i.sha(self.plan_bytes),jobId=7,runId=9,runAttempt=1,headSHA='a'*40,jobLog=self.value['jobLog'],jobsAPI=self.value['jobsAPI'],phaseRecords=i.phase_log_records(self.plan,self.steps,self.log))
        self.value['phaseOrderEvidence']=self.bind('provider/phase-order-proof.json',i.compact(self.proof))

    def read(self):
        return i.final_reconciliation(self.value,self.anchor,self.ident,self.refs.__getitem__,self.run,self.jobs,self.windows,self.context)

    def test_complete_real_shape_seven_digit_log_causal_overlap_and_direct_floor(self):
        self.assertTrue(self.read())
        for name,(start,end) in self.windows.items():self.windows[name]=(start,end.replace('01.500000Z','01.000000Z'))
        self.assertTrue(self.read())

    def test_logical_derived_proof_survives_physical_custody_relocation_but_not_role_or_bytes_change(self):
        original_proof=self.refs[self.value['phaseOrderEvidence']['path']]
        for role in ('jobLog','jobsAPI'):
            original=self.value[role];self.value[role]=dict(original,path='benchmarks/retained-screen/'+original['path'])
            self.refs[self.value[role]['path']]=self.refs[original['path']]
        self.assertTrue(self.read())
        self.assertEqual(self.refs[self.value['phaseOrderEvidence']['path']],original_proof)
        for change in ({'path':'provider/other-job.log'},{'bytes':1},{'sha256':'0'*64}):
            proof=copy.deepcopy(self.proof);proof['jobLog'].update(change)
            self.value['phaseOrderEvidence']=self.bind('provider/phase-order-proof.json',i.compact(proof))
            with self.subTest(change=change),self.assertRaises(ValueError):self.read()

    def test_ambiguous_end_without_proof_missing_final_marker_is_incomplete(self):
        self.value.pop('phaseOrderEvidence')
        with self.assertRaises(ValueError):self.read()
        self.setUp();self.log=self.log.replace(i.PHASE_PREFIX.encode(),b'NO-PHASE-PROOF ',1)
        self.value['jobLog']=self.bind('provider/job.log',self.log)
        with self.assertRaises(ValueError):self.read()

    def test_fractional_exact_and_one_hundred_ns_contradictions_never_rescued(self):
        phase=self.plan['phasePlans'][0];name=phase['captureStepName'];step=self.job['steps'][1]
        for complete,end in [('2026-11-04T10:01:01.0000000Z','2026-11-04T10:01:01.0000001Z'),('2026-11-04T10:01:01.0000001Z','2026-11-04T10:01:01.0000009Z'),('2026-11-04T10:01:01Z','2026-11-04T10:01:02.0000000Z')]:
            step['completed_at']=complete;self.windows[name]=(self.windows[name][0],end);self.refresh()
            with self.subTest(complete=complete,end=end),self.assertRaises(ValueError):self.read()

    def test_source_step_failed_missing_reordered_or_wrong_name_rejects(self):
        for mutation in ('failed','missing','renamed','reordered'):
            job=copy.deepcopy(self.job)
            if mutation=='failed':self.job['steps'][2]['conclusion']='failure'
            elif mutation=='missing':self.job['steps'].pop(2)
            elif mutation=='renamed':self.job['steps'][2]['name']='unrelated action'
            else:self.job['steps'][2]['number']=99
            self.refresh()
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):self.read()
            self.job.clear();self.job.update(job)

    def test_duplicate_quoted_wrong_phase_run_head_snapshot_and_out_of_scope_markers_reject(self):
        original=self.log
        for mutation in ('duplicate','quoted','phase','run','head','snapshot','after-next','echo-only'):
            lines=original.decode().splitlines();index=next(n for n,line in enumerate(lines) if i.PHASE_PREFIX in line)
            if mutation=='duplicate':lines.insert(index,lines[index])
            elif mutation=='quoted':lines[index]=lines[index].replace(i.PHASE_PREFIX,'echo "'+i.PHASE_PREFIX)+'"'
            elif mutation=='phase':lines[index]=lines[index].replace('"phase":"before-tools"','"phase":"prepared"')
            elif mutation=='run':lines[index]=lines[index].replace('"runId":9','"runId":10')
            elif mutation=='head':lines[index]=lines[index].replace('a'*40,'b'*40)
            elif mutation=='snapshot':lines[index]=lines[index].replace(self.hashes['before-tools'],'f'*64)
            elif mutation=='after-next':line=lines.pop(index);lines.insert(index+2,line)
            else:lines[index]=lines[index].split(' ',1)[0]+' \x1b[36;1m'+lines[index].split(' ',1)[1]+'\x1b[0m'
            self.log=('\n'.join(lines)+'\n').encode();self.value['jobLog']=self.bind('provider/job.log',self.log)
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):self.read()

    def test_declared_command_digest_and_source_hash_mutation_reject(self):
        self.plan['phasePlans'][0]['foregroundCommandSHA256']='0'*64
        self.context['planBytes']=i.compact(self.plan);self.anchor['phaseOrderPlanSHA256']=i.sha(self.context['planBytes']);self.proof['phaseOrderPlanSHA256']=self.anchor['phaseOrderPlanSHA256'];self.value['phaseOrderEvidence']=self.bind('provider/phase-order-proof.json',i.compact(self.proof))
        with self.assertRaises(ValueError):self.read()

    def test_frozen_workflow_emitter_raw_collector_and_controller_whole_bytes_are_required(self):
        for path in self.sources:
            original=self.sources[path];self.sources[path]=original+b' changed source'
            with self.subTest(path=path),self.assertRaises(ValueError):self.read()
            self.sources[path]=original

    def test_unknown_final_proof_keys_bool_ordinals_or_observation_substitution_reject(self):
        original=copy.deepcopy(self.proof)
        for mode in ('unknown','bool','observed','ordinal'):
            proof=copy.deepcopy(original)
            if mode=='unknown':proof['extra']=1
            elif mode=='bool':proof['phaseRecords'][0]['markerLineOrdinal']=True
            elif mode=='observed':proof['phaseRecords'][0]['observedProviderTimestamp']='2026-11-04T10:01:01.8000009Z'
            else:proof['phaseRecords'][0]['markerLineOrdinal']+=1
            self.value['phaseOrderEvidence']=self.bind('provider/phase-order-proof.json',i.compact(proof))
            with self.subTest(mode=mode),self.assertRaises(ValueError):self.read()

    def test_background_parallel_continued_error_or_detach_source_even_rehashed_reject(self):
        original=self.sources[i.WORKFLOW_PATH]
        for bad in (original.replace(b'    steps:',b'    background: true\n    steps:'),original.replace(b'    steps:',b'    parallel: true\n    steps:'),original.replace(b'    steps:',b'    continue-on-error: true\n    steps:'),original.replace(b'before-tools\n',b'before-tools &\n')):
            with self.assertRaises(ValueError):i.declared_steps(bad)


if __name__=='__main__':unittest.main()
