"""Qualified future-Brace observations; never canonical VM/build attestation."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import keel_brace_image_probe as probe

ROOT = Path(__file__).resolve().parent.parent
POLICY_PATH = 'benchmarks/keel-brace-v1/runtime-identity-amendment.json'
DOCUMENT_PATH = 'docs/improvements/keel-brace-runtime-identity-amendment.md'
POLICY_SHA = '7429753f1d72c888718bd8965a674c7d7f18003f7f9edf3dee309bbb30976edf'
DOCUMENT_SHA = '85229f18285ce4dabbd076d5f8e56d0a6ec93174c5ab98bfac56ff6553dabb84'
POLICY_ID = 'keel-brace-qualified-native-observations-v1'
NATIVE = dict(image='ubuntu-26.04', version='20261025.179.1',
    includedSoftwareURL='https://github.com/actions/runner-images/blob/ubuntu26/20261025.179/images/ubuntu/Ubuntu2604-Readme.md',
    imageReleaseURL='https://github.com/actions/runner-images/releases/tag/ubuntu26%2F20261025.179')
ENVIRONMENT = dict(ImageOS='ubuntu26', ImageVersion='20261026.180.1')
STABLE_RAW = ('sourcePath','resolvedPath','size','wholeSHA256','mode','uid','gid')
SNAPSHOT_KEYS = {'schemaVersion','policyID','policySHA256','stage','phase','runId','runAttempt','headSHA',
    'workspace','bootId','architecture','kernel','kernelVersion','logicalJobName','collectorSourceSHA256',
    'snapshotManifestSHA256','reportSHA256','stable'}
REVIEW_KEYS = {'schemaVersion','policyID','policySHA256','beforeToolsSnapshotSHA256','preparedSnapshotSHA256',
    'preparedWrapperSHA256','primaryPregameAuthority','earlyHeaderEvidence','finalExternalReconciliationRequired'}
EARLY_KEYS = {'status','endpoint','runId','runAttempt','jobId','headSHA','observedAtUTC','httpStatus',
    'responseBytes','responseSHA256','responseFile','scope','runAccessVerified','jobsAccessVerified',
    'artifactAccessVerified','jobInProgress','nativeClaim'}


def sha(data): return hashlib.sha256(data).hexdigest()


def pairs(values):
    result={}
    for key,value in values:
        if key in result: raise ValueError('Duplicate JSON key')
        result[key]=value
    return result


def decode(data):
    return json.loads(data.decode('utf-8'),object_pairs_hook=pairs,
        parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def read(path, cap=probe.FILE_CAP):
    with Path(path).open('rb') as f: data=f.read(cap+1)
    if len(data)>cap: raise ValueError('Image identity file cap')
    return data


def load(path): return decode(read(path))


def write(path, value):
    data=json.dumps(value,sort_keys=True,indent=2,allow_nan=False).encode()+b'\n'
    if len(data)>probe.FILE_CAP: raise ValueError('Image identity report cap')
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f:f.write(data)
    return sha(data)


def policy():
    if sha(read(ROOT/POLICY_PATH))!=POLICY_SHA or sha(read(ROOT/DOCUMENT_PATH))!=DOCUMENT_SHA:
        raise ValueError('Published runtime amendment bytes changed')
    p=load(ROOT/POLICY_PATH)
    if p['policyID']!=POLICY_ID or p['requiredNativeClaim']!=NATIVE or p['requiredSeparateEnvironmentClaim']!=ENVIRONMENT:
        raise ValueError('Published runtime policy tuple changed')
    return p


def integer(value, minimum=0):
    if type(value) is not int or value<minimum: raise ValueError('Expected strict integer')
    return value


def safe(root, name):
    if not isinstance(name,str) or not name or Path(name).is_absolute() or '..' in Path(name).parts:
        raise ValueError('Unsafe retained image path')
    path=Path(root)/name
    if path.is_symlink() or not path.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError('Retained image path escaped custody')
    return path


def source_context(stage):
    if stage not in ('screen','confirmation'): raise ValueError('Unknown stage')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True,timeout=10).strip()
    if not re.fullmatch('[a-f0-9]{40}',head):raise ValueError('Actual checkout SHA missing')
    if os.environ.get('GITHUB_REPOSITORY')!='stephanelee/robocode' or os.environ.get('GITHUB_RUN_ATTEMPT')!='1':
        raise ValueError('Wrong repository/attempt')
    if head!=os.environ.get('GITHUB_SHA') and head!=os.environ.get('KEEL_BRACE_IMAGE_PROBE_HEAD_SHA'):
        raise ValueError('Actual checkout not bound to reviewed stage head')
    run=int(os.environ['GITHUB_RUN_ID']);integer(run,1)
    if os.environ.get('GITHUB_JOB')!='registered-stage':raise ValueError('Wrong logical job')
    return dict(stage=stage,runId=run,runAttempt=1,headSHA=head,workspace=str(ROOT.resolve()),logicalJobName='registered-stage')


def verify_capture(directory):
    """Reparse all original raw bytes; no repaired or normalized metadata is accepted."""
    directory=Path(directory);status=load(directory/'capture-status.json');report=load(directory/'report.json')
    if status['report']['sha256']!=sha(read(directory/'report.json')) or status['report']['complete'] is not True:
        raise ValueError('Incomplete/mutated collector report')
    integer(report.get('durationNs'))
    if report.get('collectorPython',{}).get('executable')!='/usr/bin/python3':
        raise ValueError('Collector must execute using stock /usr/bin/python3')
    if report.get('errors') or report.get('durationNs',10**30)>120*10**9:
        raise ValueError('Collector error/deadline')
    if status['files'] != report['rawArtifactIndex'] | {'report.json': status['report']}:
        raise ValueError('Collector report/status inventory differs')
    for name,row in report['rawArtifactIndex'].items():
        data=read(safe(directory,name))
        if row['complete'] is not True or type(row['bytes']) is not int or row['bytes']!=len(data) or row['suppliedBytes']!=len(data) or row['sha256']!=sha(data):
            raise ValueError('Incomplete/mutated collector payload')
    if report['rawArtifactIndex'].get('collector.py') != report['collectorSource']:
        raise ValueError('Collector source missing from manifest')
    if report['collectorSource']['sha256']!=sha(read(ROOT/'tools/keel_brace_image_probe.py')):
        raise ValueError('Executed collector bytes differ')
    discovery=report['runnerDiscovery']
    integer(discovery['scannedEntries'],1)
    for match in discovery['matches']:integer(match['pid'],1)
    if (discovery.get('errors') or discovery.get('truncated') is not False or len(discovery['roots'])!=1
            or not 0<discovery['scannedEntries']<=512 or not 0<len(discovery['matches'])<16
            or any(m['errors'] for m in discovery['matches'])):
        raise ValueError('Ambiguous/incomplete runner-root discovery')
    root=discovery['roots'][0]
    for match in discovery['matches']:
        if (match['runnerRoot']!=root or match['comm'] not in probe.RUNNER_NAMES
                or match['exePath']!=str(Path(root)/'bin'/match['comm'])
                or match['setupInfoPath']!=str(Path(root)/'.setup_info')):
            raise ValueError('Observed runner executable does not bind root')
    required=set(probe.FIXED_FILES)|{'setup_info_0'}
    if set(report['captures'])!=required: raise ValueError('Wrong raw source inventory')
    raw={};claims={};optional_errors=0
    for name,row in report['captures'].items():
        expected=probe.FIXED_FILES.get(name,str(Path(root)/'.setup_info'))
        if row['requestedPath']!=expected:raise ValueError('Wrong metadata source path')
        if name=='etc_lsb_release' and row.get('exists') is False:
            errors=row.get('errors',[])
            if (len(errors)!=1 or set(errors[0])!={'operation','path','type','message','messageWasTruncated'}
                    or errors[0].get('type')!='FileNotFoundError' or errors[0].get('operation')!='lstat'
                    or errors[0].get('path')!=expected or errors[0].get('messageWasTruncated') is not False):
                raise ValueError('Optional LSB absence not clean ENOENT')
            optional_errors+=1
            raw[name]={'sourcePath':expected,'absent':True};continue
        if row.get('errors') or row.get('parseErrors') or row.get('bytesComplete') is not True or row.get('rawSHA256Scope')!='whole-observed-file':
            raise ValueError('Incomplete/error raw image source '+name)
        indexed = report['rawArtifactIndex'].get(row['raw']['artifactPath'])
        if indexed is None or indexed != row['raw']:
            raise ValueError('Referenced raw source absent/different in indexed manifest')
        data=read(safe(directory,row['raw']['artifactPath']))
        if row['raw']['sourceBytesComplete'] is not True or row['raw']['sha256']!=sha(data):raise ValueError('Raw source binding')
        if row['resolvedPath'] not in set(probe.FIXED_FILES.values())|{str(Path(root)/'.setup_info')}:
            raise ValueError('Resolved metadata path outside allowlist')
        if row['raw']['bytes'] != len(data) or row['raw']['suppliedBytes'] != len(data):raise ValueError('Complete raw read size mismatch')
        if 'parsedArtifact' in row:
            if report['rawArtifactIndex'].get(row['parsedArtifact']['artifactPath']) != row['parsedArtifact']:
                raise ValueError('Parsed source absent from manifest')
        before,opened,after=row['statBefore'],row['openedStat'],row['statAfter']
        if opened['size'] != len(data) and not (name=='boot_id' and opened['size']==0):
            raise ValueError('Complete regular source stat/read byte count differs')
        if any(opened[k]!=after[k] for k in ('size','mtimeNs','inode','device','mode','uid','gid')):
            raise ValueError('Metadata changed during read')
        if any(before[k]!=opened[k] for k in ('inode','device','size','mode','uid','gid')):raise ValueError('Metadata changed before open')
        for key in ('mode','uid','gid','size'):integer(opened[key])
        raw[name]=dict(sourcePath=expected,resolvedPath=row['resolvedPath'],size=opened['size'],wholeSHA256=sha(data),
            mode=opened['mode'],uid=opened['uid'],gid=opened['gid'])
        if name in ('imagedata','setup_info_0'):
            parsed=probe.parse_image_json(data)
            if parsed['errors'] or parsed['runnerImageRecords'][0]['unrecognizedLines'] or parsed['claim']!=NATIVE:raise ValueError('Unknown native claim '+name)
            claims['imagedataNative' if name=='imagedata' else 'setupInfoNative']=parsed['claim']
        elif name=='etc_environment':
            parsed=probe.parse_environment_file(data)
            value={k:v['value'] for k,v in parsed['claims'].items() if v['present']}
            if parsed['errors'] or value!=ENVIRONMENT:raise ValueError('Wrong separate etc environment')
            claims['etcEnvironment']=value
        if name in ('imagedata','setup_info_0','etc_environment'):
            actual_parsed=load(safe(directory,row['parsedArtifact']['artifactPath']))
            expected_claim=parsed.get('claim') if 'claim' in parsed else parsed.get('claims')
            if actual_parsed!=parsed or row.get('parsedClaims')!=expected_claim or row.get('parseErrors')!=parsed['errors']:
                raise ValueError('Retained parser report differs from original raw reparse')
        if name=='boot_id':
            boot=data.decode('ascii').strip()
            if not re.fullmatch('[0-9a-f-]{36}',boot):raise ValueError('Missing actual boot ID')
        elif name in ('etc_os_release','usr_lib_os_release'):
            if not re.search(rb'^VERSION_ID="?26\.04"?$',data,re.M):raise ValueError('Not Ubuntu26.04 metadata')
    if type(status.get('diagnosticErrorCount')) is not int or status['diagnosticErrorCount']!=optional_errors:
        raise ValueError('Collector global error count inconsistent with sole optional LSB exception')
    process=report['processImageEnvironment']
    if set(process)!=set(ENVIRONMENT) or any(process[k].get('present') is not True or process[k].get('value')!=v for k,v in ENVIRONMENT.items()):
        raise ValueError('Wrong separate process environment')
    claims['processEnvironment']=dict(ENVIRONMENT)
    if report['platform']['architecture']!='x86_64' or report['platform']['system']!='Linux':raise ValueError('Wrong worker platform')
    return dict(rawSources=raw,claims=claims,runnerRoot=root,setupInfoPath=str(Path(root)/'.setup_info'),
        bootId=boot,architecture=report['platform']['architecture'],kernel=report['platform']['kernel'],
        kernelVersion=report['platform']['kernelVersion'],collectorSourceSHA256=report['collectorSource']['sha256'])


def snapshot(base,stage,phase):
    policy();context=source_context(stage)
    if phase not in ('before-tools','prepared') and not phase.startswith('revalidation-'):raise ValueError('Unknown capture phase')
    folder=Path(base)/phase;collector_phase='before-tools' if phase=='before-tools' else 'after-tools'
    captured=folder/'collector'/collector_phase
    start=time.monotonic()
    subprocess.run(['/usr/bin/python3','-I','-B','-S',str(ROOT/'tools/keel_brace_image_probe.py'),
        '--output-dir',str(captured),'--phase',collector_phase],check=False,timeout=120)
    # A nonzero collector exit can represent only optional LSB absence; exact
    # raw/status/error validation below decides, never a return-code waiver.
    stable=verify_capture(captured)
    if time.monotonic()-start>120:raise ValueError('Total capture/validation deadline')
    stable['sourceIdentity']=context
    value=dict(schemaVersion=1,policyID=POLICY_ID,policySHA256=POLICY_SHA,phase=phase,**context,
        bootId=stable['bootId'],architecture=stable['architecture'],kernel=stable['kernel'],kernelVersion=stable['kernelVersion'],
        collectorSourceSHA256=stable['collectorSourceSHA256'],snapshotManifestSHA256=sha(read(captured/'capture-status.json')),
        reportSHA256=sha(read(captured/'report.json')),stable=stable)
    write(folder/'snapshot.json',value)
    return value


def original(base,phase):
    base=Path(base);value=load(base/phase/'snapshot.json')
    if set(value)!=SNAPSHOT_KEYS or value['policyID']!=POLICY_ID or value['policySHA256']!=POLICY_SHA:
        raise ValueError('Wrong snapshot schema/policy')
    for key in ('schemaVersion','runId','runAttempt'):integer(value[key],1)
    if value['schemaVersion']!=1 or value['runAttempt']!=1:raise ValueError('Wrong snapshot attempt/schema')
    for key in ('policySHA256','collectorSourceSHA256','snapshotManifestSHA256','reportSHA256'):
        if not re.fullmatch('[0-9a-f]{64}',value[key]):raise ValueError('Malformed snapshot SHA')
    captured=base/phase/'collector'/('before-tools' if phase=='before-tools' else 'after-tools')
    stable=verify_capture(captured)
    if value['phase'] != phase or value['stage'] not in ('screen','confirmation'):
        raise ValueError('Snapshot phase/stage mismatch')
    for key in ('bootId','architecture','kernel','kernelVersion','collectorSourceSHA256'):
        if value[key] != stable[key]:raise ValueError('Snapshot top-level/derived stable mismatch')
    if (not re.fullmatch('[a-f0-9]{40}',value['headSHA']) or not isinstance(value['workspace'],str)
            or not Path(value['workspace']).is_absolute() or value['logicalJobName']!='registered-stage'):
        raise ValueError('Malformed snapshot source identity')
    report=load(captured/'report.json');context=report['context']
    expected_phase='before-tools' if phase=='before-tools' else 'after-tools'
    if (report['phase']!=expected_phase or context['GITHUB_REPOSITORY']!='stephanelee/robocode'
            or context['GITHUB_RUN_ID']!=str(value['runId']) or context['GITHUB_RUN_ATTEMPT']!='1'
            or context['GITHUB_JOB']!=value['logicalJobName'] or context['KEEL_BRACE_IMAGE_PROBE_HEAD_SHA']!=value['headSHA']):
        raise ValueError('Collector raw context/phase not bound to snapshot')
    stable['sourceIdentity']={k:value[k] for k in ('stage','runId','runAttempt','headSHA','workspace','logicalJobName')}
    if stable!=value['stable'] or sha(read(captured/'capture-status.json'))!=value['snapshotManifestSHA256'] or sha(read(captured/'report.json'))!=value['reportSHA256']:
        raise ValueError('Approved raw image snapshot changed')
    return value


def compare(early,prepared):
    if early['stable']!=prepared['stable']:raise ValueError('Same-worker stable metadata/source changed; no repair')


def bind_prepared(base,jobs):
    base=Path(base);a=original(base,'before-tools');b=original(base,'prepared');compare(a,b)
    active=[j for j in jobs['jobs'] if j.get('name')=='registered-stage' and j.get('status')=='in_progress']
    if len(active)!=1:raise ValueError('Cannot bind unique actual numeric stage job')
    job=active[0];integer(job['id'],1)
    if job.get('run_id')!=a['runId'] or job.get('run_attempt')!=1 or job.get('head_sha')!=a['headSHA']:
        raise ValueError('Prepared numeric job belongs to other run/head/attempt')
    write(base/'prepared-jobs-api.json',jobs)
    wrapper=dict(schemaVersion=1,policyID=POLICY_ID,policySHA256=POLICY_SHA,runId=a['runId'],runAttempt=1,
        headSHA=a['headSHA'],logicalJobName=a['logicalJobName'],preparedNumericJobId=job['id'],
        beforeToolsSnapshotFile='before-tools/snapshot.json',beforeToolsSnapshotSHA256=sha(read(base/'before-tools/snapshot.json')),
        preparedSnapshotFile='prepared/snapshot.json',preparedSnapshotSHA256=sha(read(base/'prepared/snapshot.json')),
        jobsAPIFile='prepared-jobs-api.json',jobsAPISHA256=sha(read(base/'prepared-jobs-api.json')))
    write(base/'prepared-wrapper.json',wrapper)
    check_total(base)
    return wrapper


def check_total(base):
    size=sum(p.stat().st_size for p in Path(base).rglob('*') if p.is_file())+probe.ZIP_OVERHEAD_RESERVE
    if size>probe.ARTIFACT_CAP:raise ValueError('Two-snapshot payload plus archive reserve exceeds10MiB')


WRAPPER_KEYS = {'schemaVersion','policyID','policySHA256','runId','runAttempt','headSHA','logicalJobName',
    'preparedNumericJobId','beforeToolsSnapshotFile','beforeToolsSnapshotSHA256','preparedSnapshotFile',
    'preparedSnapshotSHA256','jobsAPIFile','jobsAPISHA256'}


def profile(base):
    policy();base=Path(base);a=original(base,'before-tools');b=original(base,'prepared');compare(a,b)
    wrapper=load(base/'prepared-wrapper.json')
    if set(wrapper)!=WRAPPER_KEYS:raise ValueError('Exact prepared wrapper keys required')
    for key in ('schemaVersion','runId','runAttempt','preparedNumericJobId'):integer(wrapper[key],1)
    if (wrapper['schemaVersion']!=1 or wrapper['policyID']!=POLICY_ID or wrapper['policySHA256']!=POLICY_SHA
            or any(wrapper[k]!=a[k] for k in ('runId','runAttempt','headSHA','logicalJobName'))
            or wrapper['beforeToolsSnapshotFile']!='before-tools/snapshot.json'
            or wrapper['preparedSnapshotFile']!='prepared/snapshot.json' or wrapper['jobsAPIFile']!='prepared-jobs-api.json'
            or wrapper['beforeToolsSnapshotSHA256']!=sha(read(base/'before-tools/snapshot.json'))
            or wrapper['preparedSnapshotSHA256']!=sha(read(base/'prepared/snapshot.json'))
            or wrapper['jobsAPISHA256']!=sha(read(base/'prepared-jobs-api.json'))):raise ValueError('Prepared wrapper binding changed')
    jobs=load(base/'prepared-jobs-api.json')['jobs']
    active=[j for j in jobs if j.get('name')=='registered-stage' and j.get('status')=='in_progress']
    if (len(active)!=1 or active[0].get('id')!=wrapper['preparedNumericJobId']
            or active[0].get('run_id')!=a['runId'] or active[0].get('run_attempt')!=1 or active[0].get('head_sha')!=a['headSHA']):
        raise ValueError('Prepared wrapper does not match actual numeric jobs API')
    return dict(policyID=POLICY_ID,policySHA256=POLICY_SHA,beforeToolsSnapshotSHA256=wrapper['beforeToolsSnapshotSHA256'],
        preparedSnapshotSHA256=wrapper['preparedSnapshotSHA256'],preparedWrapperSHA256=sha(read(base/'prepared-wrapper.json')),
        preparedNumericJobId=wrapper['preparedNumericJobId'],stable=b['stable'])


def live(base,stage,output):
    expected=original(base,'prepared');current=snapshot(output,stage,'revalidation-'+str(time.monotonic_ns()))
    compare(expected,current)
    return current


def parse_provider_header(raw):
    """Only original provider startup, before the first named workflow step."""
    text=raw.decode('utf-8-sig');lines=[]
    for line in text.splitlines():
        line=re.sub(r'^\d{4}-\d\d-\d\dT[0-9:.]+Z ', '',line)
        lines.append(line)
    starts=[i for i,line in enumerate(lines) if line=='##[group]Runner Image']
    # Startup has no workflow step Run/action markers before its own group.
    stops=[i for i,line in enumerate(lines) if line.startswith(('##[group]Run ','##[group]Post ','##[group]Complete job','Complete job name: '))]
    if not stops or not lines or not lines[0].startswith('Current runner version: '):
        raise ValueError('Missing original provider startup framing/terminating boundary')
    limit=min(stops)
    if sum(line.startswith('Current runner version: ') for line in lines[:limit])!=1:
        raise ValueError('Ambiguous provider startup framing')
    starts=[i for i in starts if i<limit]
    if len(starts)!=1:raise ValueError('Missing/ambiguous provider startup Runner Image')
    start=starts[0]
    end=next((i for i in range(start+1,limit) if lines[i]=='##[endgroup]'),None)
    if end is None:raise ValueError('Truncated provider Runner Image group')
    parsed=probe.parse_image_detail('\n'.join(lines[start+1:end]))
    if parsed['errors'] or parsed['unrecognizedLines'] or parsed['claim']!=NATIVE:
        raise ValueError('Provider startup conflicts/malformed qualified tuple')
    return parsed['claim']


def review(value,anchor,ident):
    if not isinstance(value,dict) or set(value)!=REVIEW_KEYS:raise ValueError('Exact image identity review required')
    for key in ('policyID','policySHA256','beforeToolsSnapshotSHA256','preparedSnapshotSHA256','preparedWrapperSHA256'):
        if value[key]!=anchor[key]:raise ValueError('Image review does not bind actual immutable snapshots')
    if type(value['schemaVersion']) is not int or value['schemaVersion']!=1 or value['primaryPregameAuthority']!='native-only-qualified-observation' or value['finalExternalReconciliationRequired'] is not True:
        raise ValueError('Qualified native boundary/final reconciliation not acknowledged')
    early=value['earlyHeaderEvidence']
    if not isinstance(early,dict) or set(early)!=EARLY_KEYS:raise ValueError('Exact early provider evidence required')
    for key in ('runId','runAttempt','jobId','httpStatus'):integer(early[key],1)
    integer(early['responseBytes'])
    if type(early['jobInProgress']) is not bool:raise ValueError('Strict live job status required')
    if (early['runId']!=ident['runId'] or early['runAttempt']!=1 or early['jobId']!=anchor['preparedNumericJobId']
            or early['headSHA']!=ident['headSHA'] or early['endpoint']!=f"https://api.github.com/repos/stephanelee/robocode/actions/jobs/{early['jobId']}/logs"
            or early['responseFile']!='early-header-response.bin' or not re.fullmatch('[0-9a-f]{64}',early['responseSHA256'])
            or any(early[k] is not True for k in ('runAccessVerified','jobsAccessVerified','artifactAccessVerified'))):
        raise ValueError('Early evidence wrong actual job/access/response binding')
    observed=dt.datetime.fromisoformat(early['observedAtUTC'].replace('Z','+00:00'))
    if observed.utcoffset()!=dt.timedelta(0):raise ValueError('Early observation requires explicit UTC')
    if early['status']=='available-matched':
        if early['httpStatus']!=200 or early['scope']!='complete-startup-header' or early['nativeClaim']!=NATIVE:
            raise ValueError('Available provider evidence must match native tuple')
    elif early['status']=='unavailable-404-deferred':
        if early['httpStatus']!=404 or early['scope']!='actual-job-log-request' or early['jobInProgress'] is not True or early['nativeClaim'] is not None:
            raise ValueError('Only actual in-progress HTTP404 may defer')
    else:raise ValueError('Unrecognized pregame header availability')
    return True


FINAL_KEYS = {'schemaVersion','status','policyID','policySHA256','runId','runAttempt','jobId','headSHA',
    'beforeToolsSnapshotSHA256','preparedSnapshotSHA256','preparedWrapperSHA256','nativeClaim','runAPI','jobsAPI',
    'jobLog','completedAtUTC','reviewerTask'}


def timestamp(value):
    result=dt.datetime.fromisoformat(value.replace('Z','+00:00'))
    if result.utcoffset()!=dt.timedelta(0):raise ValueError('Explicit UTC required')
    return result


def within_api_step(start,end,step):
    low,high=timestamp(step['started_at']),timestamp(step['completed_at'])
    first,last=timestamp(start),timestamp(end)
    # Provider API second-resolution timestamps describe that recorded second,
    # not an invented microsecond instant. No arbitrary clock-skew allowance.
    seconds_only=bool(re.fullmatch(r'\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:Z|\+00:00)',step['completed_at']))
    upper=last<high+dt.timedelta(seconds=1) if seconds_only else last<=high
    return low<=first<=last and upper


def final_reconciliation(value,anchor,ident,read_ref,current_run,current_jobs,capture_windows):
    """Verify externally retained full provider evidence, never mutate raw arithmetic."""
    if not isinstance(value,dict) or set(value)!=FINAL_KEYS:raise ValueError('Separate final provider reconciliation required')
    if anchor['policyID']!=POLICY_ID or anchor['policySHA256']!=POLICY_SHA:
        raise ValueError('Final screen uses wrong qualified-runtime policy')
    for key in ('schemaVersion','runId','runAttempt','jobId'):integer(value[key],1)
    if value['schemaVersion']!=1 or value['status']!='PASS' or value['nativeClaim']!=NATIVE:
        raise ValueError('Final provider provenance incomplete')
    for key in ('policyID','policySHA256','beforeToolsSnapshotSHA256','preparedSnapshotSHA256','preparedWrapperSHA256'):
        if value[key]!=anchor[key]:raise ValueError('Final review does not bind original image snapshots')
    if (not isinstance(value['reviewerTask'],str) or not value['reviewerTask'].strip()
            or value['runId']!=ident['runId'] or value['headSHA']!=ident['headSHA'] or value['runAttempt']!=1
            or value['jobId']!=anchor['preparedNumericJobId'] or not value['reviewerTask']):
        raise ValueError('Final provider review wrong actual worker identity')
    evidence={}
    for key in ('runAPI','jobsAPI','jobLog'):
        ref=value[key]
        if not isinstance(ref,dict) or set(ref)!={'path','bytes','sha256'}:raise ValueError('Exact original provider evidence ref required')
        integer(ref['bytes'],1)
        data=read_ref(ref['path'])
        if len(data)!=ref['bytes'] or sha(data)!=ref['sha256']:raise ValueError('Final original provider evidence changed')
        evidence[key]=data
    run=decode(evidence['runAPI']);jobs=decode(evidence['jobsAPI'])['jobs']
    for actual in (run,current_run):
        integer(actual.get('id'),1);integer(actual.get('run_attempt'),1)
        if (actual.get('id')!=ident['runId'] or actual.get('head_sha')!=ident['headSHA'] or actual.get('run_attempt')!=1
                or actual.get('status')!='completed' or actual.get('conclusion')!='success'):
            raise ValueError('Final provider run not exact completed successful attempt')
    selected=[]
    for population in (jobs,current_jobs['jobs']):
        stage=[j for j in population if j.get('name')=='registered-stage']
        if len(stage)!=1:raise ValueError('Ambiguous final registered-stage job')
        job=stage[0]
        for key in ('id','run_id','run_attempt'):integer(job.get(key),1)
        if (job.get('id')!=value['jobId'] or job.get('run_id')!=ident['runId'] or job.get('run_attempt')!=1
                or job.get('head_sha')!=ident['headSHA'] or job.get('status')!='completed' or job.get('conclusion')!='success'):
            raise ValueError('Final provider numeric job mismatch')
        selected.append(job)
    job=selected[0]
    if value['completedAtUTC']!=job['completed_at'] or selected[1]['completed_at']!=job['completed_at']:
        raise ValueError('Final provider completion binding changed')
    setup=[s for s in job['steps'] if s.get('name')=='Set up job']
    if len(setup)!=1 or setup[0].get('conclusion')!='success':raise ValueError('Original startup step missing')
    log=evidence['jobLog'];parse_provider_header(log)
    raw_lines=log.decode('utf-8-sig').splitlines()
    starts=[line.split(' ',1)[0] for line in raw_lines if line.endswith(' ##[group]Runner Image')]
    if not starts or not within_api_step(starts[0],starts[0],setup[0]):
        raise ValueError('Runner Image is outside bound Set up job UTC scope')
    # Original stage must have actual before-tools/prepared/upload steps in source order.
    required=['Capture original qualified image observations before tools',
        'Prepare exact sources/runtime/commands on this worker (zero games)',
        'Finalize immutable zero-game provenance before independent review']
    prior=timestamp(setup[0]['completed_at'])
    for name in required:
        steps=[s for s in job['steps'] if s.get('name')==name]
        if len(steps)!=1 or steps[0].get('conclusion')!='success' or timestamp(steps[0]['started_at'])<prior:
            raise ValueError('Original image capture/preparation/upload step order invalid')
        if name in capture_windows:
            start,end=capture_windows[name]
            if not within_api_step(start,end,steps[0]):
                raise ValueError('Original image capture UTC outside actual provider step')
        prior=timestamp(steps[0]['completed_at'])
    if set(capture_windows)!=set(required[:2]):raise ValueError('Both original image capture UTC windows required')
    return True


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('screen','confirmation'));p.add_argument('phase',choices=('before-tools',));args=p.parse_args()
    snapshot(ROOT/'.benchmark-cache/keel-brace-image-identity'/args.stage,args.stage,args.phase)


if __name__=='__main__':main()
