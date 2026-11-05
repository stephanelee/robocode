"""Pure qualified-case/provider/schema/budget tests, no collection or runtime setup."""
import copy
import datetime
import json
import tempfile
from pathlib import Path
import time
import unittest
from unittest import mock
import keel_wake_image_identity as i


def anchor():
    return dict(policyID=i.POLICY_ID,policySHA256=i.POLICY_SHA,caseID=i.CASE_A,
        effectiveFreezePath=i.EFFECTIVE_FREEZE_PATH,effectiveFreezeSHA256='5'*64,
        beforeToolsSnapshotSHA256='1'*64,preparedSnapshotSHA256='2'*64,preparedWrapperSHA256='3'*64,preparedNumericJobId=7)


def review(ident=None):
    ident=ident or dict(runId=9,headSHA='a'*40)
    return dict(schemaVersion=1,**{k:v for k,v in anchor().items() if k not in ('preparedNumericJobId','effectiveFreezePath','effectiveFreezeSHA256')},
        primaryPregameAuthority='native-only-qualified-observation',finalExternalReconciliationRequired=True,
        earlyHeaderEvidence=dict(status='unavailable-404-deferred',endpoint='https://api.github.com/repos/stephanelee/robocode/actions/jobs/7/logs',runId=ident['runId'],runAttempt=1,jobId=7,headSHA=ident['headSHA'],observedAtUTC='2026-11-04T10:00:30Z',httpStatus=404,responseBytes=10,responseSHA256='4'*64,responseFile='early-header-response.bin',scope='actual-job-log-request',runAccessVerified=True,jobsAccessVerified=True,artifactAccessVerified=True,jobInProgress=True,nativeClaim=None))


def header(case):
    n=i.case_by_id(case)['native']
    return ("Current runner version: 'fixture'\n##[group]Runner Image\nImage: "+n['image']+'\nVersion: '+n['version']+'\nIncluded Software: '+n['includedSoftwareURL']+'\nImage Release: '+n['imageReleaseURL']+'\n##[endgroup]\nComplete job name: registered-stage\n').encode()


class JointCases(unittest.TestCase):
    def test_exact_both_cases_without_mixing_or_normalizing(self):
        for case in i.JOINT_CASES:
            claims={k:copy.deepcopy(case['native']) for k in ('imagedataNative','setupInfoNative')}
            claims.update({k:copy.deepcopy(case['environment']) for k in ('processEnvironment','etcEnvironment')})
            self.assertEqual(i.joint_case(claims),case)
            other=next(c for c in i.JOINT_CASES if c!=case)
            for name in claims:
                bad=copy.deepcopy(claims);bad[name]=copy.deepcopy(other['native' if 'Native' in name else 'environment'])
                with self.subTest(case=case['caseID'],name=name),self.assertRaises(ValueError):i.joint_case(bad)
    def test_missing_extra_unknown_tuple_fail_closed(self):
        c=i.JOINT_CASES[0];claims={k:copy.deepcopy(c['native']) for k in ('imagedataNative','setupInfoNative')};claims.update({k:copy.deepcopy(c['environment']) for k in ('processEnvironment','etcEnvironment')})
        for patch in [dict(claims,unknown={}),{k:v for k,v in claims.items() if k!='etcEnvironment'}]:
            with self.assertRaises(ValueError):i.joint_case(patch)
        claims['imagedataNative']['version']='ubuntu-latest'
        with self.assertRaises(ValueError):i.joint_case(claims)


class Provider(unittest.TestCase):
    def test_complete_original_header_matches_exact_case(self):
        for case in i.JOINT_CASES:self.assertEqual(i.parse_provider_header(header(case['caseID']),case['caseID']),case['native'])
    def test_duplicate_group_missing_boundary_and_workflow_echo_rejected(self):
        raw=header(i.CASE_A)
        for bad in [raw.replace(b'##[endgroup]',b'##[endgroup]\n##[group]Runner Image\n##[endgroup]'),raw.split(b'Complete job name:')[0],b'##[group]Run echo copied-header\n'+raw]:
            with self.assertRaises(ValueError):i.parse_provider_header(bad,i.CASE_A)
    def test_other_case_header_is_not_a_fallback(self):
        with self.assertRaises(ValueError):i.parse_provider_header(header(i.CASE_B),i.CASE_A)
    def test_submicrosecond_boundaries_are_rejected_not_truncated(self):
        for suffix in ('0000001', '0000009', '0000000'):
            with self.subTest(suffix=suffix), self.assertRaises(ValueError):
                i.within_api_step('2026-11-04T10:00:01Z','2026-11-04T10:00:02.'+suffix+'Z',
                    dict(started_at='2026-11-04T10:00:00Z',completed_at='2026-11-04T10:00:02.0000000Z'))
    def test_whole_second_completion_is_not_one_second_grace(self):
        step=dict(started_at='2026-11-04T10:00:00Z',completed_at='2026-11-04T10:00:02Z')
        self.assertFalse(i.within_api_step('2026-11-04T10:00:01Z','2026-11-04T10:00:02.1Z',step))
        self.assertTrue(i.within_api_step('2026-11-04T10:00:01Z','2026-11-04T10:00:02Z',step))


class StrictEarlyReview(unittest.TestCase):
    def test_only_exact_bound_404_defer_or_complete_matched_200(self):
        ident=dict(runId=9,headSHA='a'*40);value=review(ident)
        self.assertTrue(i.review(value,anchor(),ident))
        good=copy.deepcopy(value);good['earlyHeaderEvidence'].update(status='available-matched',httpStatus=200,scope='complete-startup-header',nativeClaim=i.JOINT_CASES[0]['native'])
        self.assertTrue(i.review(good,anchor(),ident))
        for key,bad in [('httpStatus',403),('jobId',8),('jobInProgress',False),('runAccessVerified',False)]:
            value=review(ident);value['earlyHeaderEvidence'][key]=bad
            with self.subTest(key=key),self.assertRaises(ValueError):i.review(value,anchor(),ident)
    def test_unknown_missing_bool_case_and_noncanonical_claim_fail(self):
        ident=dict(runId=9,headSHA='a'*40)
        for key,bad in [('schemaVersion',True),('caseID',i.CASE_B),('primaryPregameAuthority','canonical-VM')]:
            value=review(ident);value[key]=bad
            with self.assertRaises(ValueError):i.review(value,anchor(),ident)
        value=review(ident);value.pop('caseID')
        with self.assertRaises(ValueError):i.review(value,anchor(),ident)
        value=review(ident);value['unknown']=1
        with self.assertRaises(ValueError):i.review(value,anchor(),ident)


class ActualReviewer(unittest.TestCase):
    def fixture(self):
        def pack(value):return json.dumps(value,sort_keys=True,separators=(',',':')).encode()
        task='/root/independent';impl=['/root','/root/producer']
        tool=dict(reviewerTask=task,activeBeforeDispatch=True,toolEvidence={'read':'original assignment'})
        assignment=dict(schemaVersion=1,repository='stephanelee/robocode',issue=118,stage='screen',headSHA='a'*40,
            reviewerTask=task,reviewerAccount='stephanelee',implementationTasks=impl,assignmentToolRecord=tool,
            assignmentToolRecordSHA256=i.sha(pack(tool)))
        body='KEEL_WAKE_DISPATCH_ASSIGNMENT_V1\n'+pack(assignment).decode()
        comment=dict(id=4,user=dict(login='stephanelee'),created_at='2026-11-04T10:00:00Z',updated_at='2026-11-04T10:00:00Z',body=body)
        reviewtool=dict(reviewerTask=task,toolEvidence={'read':'actual retained original API'})
        side={k:v for k,v in assignment.items() if k!='reviewerAccount'}
        side.update(runId=9,runAttempt=1,assignmentCommentId=4,assignmentBodySHA256=i.sha(body.encode()),
            reviewToolRecord=reviewtool,reviewToolRecordSHA256=i.sha(pack(reviewtool)))
        raw=pack(side);refs={'provider/reviewer-identity.json':raw,'provider/comments/4.json':pack(comment)}
        value=dict(reviewerTask=task,reviewerIdentityEvidence=dict(path='provider/reviewer-identity.json',bytes=len(raw),sha256=i.sha(raw)))
        return value,dict(runId=9,headSHA='a'*40),refs,dict(created_at='2026-11-04T10:00:02Z'),side,pack
    def test_original_assigned_actual_task_and_tool_bytes_bind_final_identity(self):
        value,ident,refs,run,side,pack=self.fixture()
        self.assertEqual(i.verify_actual_reviewer(value,ident,refs.__getitem__,run)['reviewerTask'],'/root/independent')
    def test_self_review_edited_assignment_missing_tool_and_mutated_sidecar_fail(self):
        for mode in ('self','edited','no-tool','mutated','traversal'):
            value,ident,refs,run,side,pack=self.fixture()
            if mode=='self':side['implementationTasks'].append(side['reviewerTask'])
            if mode=='no-tool':side['reviewToolRecord']['toolEvidence']={}
            if mode=='edited':
                comment=json.loads(refs['provider/comments/4.json']);comment['updated_at']='2026-11-04T10:00:01Z';refs['provider/comments/4.json']=pack(comment)
            if mode in ('self','no-tool'):
                refs['provider/reviewer-identity.json']=pack(side);value['reviewerIdentityEvidence'].update(bytes=len(pack(side)),sha256=i.sha(pack(side)))
            if mode=='mutated':refs['provider/reviewer-identity.json']+=b' '
            if mode=='traversal':value['reviewerIdentityEvidence']['path']='../provider/reviewer-identity.json'
            with self.subTest(mode=mode),self.assertRaises(ValueError):i.verify_actual_reviewer(value,ident,refs.__getitem__,run)


class Budget(unittest.TestCase):
    def tearDown(self):i.set_live_budget(None,None)
    def test_image_subprocess_caps_use_remaining_entire_check_and_job(self):
        i.set_live_budget(110,1000)
        with mock.patch.object(i.time,'monotonic',return_value=100),mock.patch.object(i.time,'time',return_value=996):self.assertEqual(i.bounded_timeout(120),4)
        with mock.patch.object(i.time,'monotonic',return_value=111),self.assertRaises(TimeoutError):i.bounded_timeout(120)
    def test_expired_budget_prevents_any_raw_capture_subprocess(self):
        i.set_live_budget(time.monotonic()-1,time.time()+4000)
        with tempfile.TemporaryDirectory() as folder,mock.patch.object(i,'policy'),mock.patch.object(i,'source_context',return_value={}),mock.patch.object(i.subprocess,'run') as process:
            with self.assertRaises(TimeoutError):i.snapshot(Path(folder),'screen','prepared')
            process.assert_not_called()
    def test_metadata_file_cap_is_exact_not_unbounded_read(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'raw';path.write_bytes(b'1234');self.assertEqual(i.read(path,4),b'1234')
            with self.assertRaises(ValueError):i.read(path,3)


if __name__=='__main__':unittest.main()
