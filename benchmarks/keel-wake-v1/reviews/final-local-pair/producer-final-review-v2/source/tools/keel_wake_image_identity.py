"""Qualified future-Wake observations; never canonical VM/build attestation."""
import argparse
import datetime as dt
import hashlib
from fractions import Fraction
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import keel_wake_image_probe as probe

ROOT = Path(__file__).resolve().parent.parent
POLICY_PATH = 'benchmarks/keel-wake-v1/runtime-identity-policy.json'
DOCUMENT_PATH = 'docs/improvements/keel-wake-v1.md'
POLICY_SHA = '9b79df831b601459964ce175acd268f1c6c2fc5ef4a2685277e959e24d3b09cf'
DOCUMENT_SHA = 'dac8a5227923639ee44679b3d3b88ea9962dc896dd8832b34c34e57b93f9157f'
POLICY_ID = 'keel-wake-qualified-native-observations-v1'
EFFECTIVE_FREEZE_PATH = 'benchmarks/keel-wake-v1/freeze.json'
JOINT_CASES = [{'caseID': 'native-20261025.179.1_env-20261026.180.1', 'native': {'image': 'ubuntu-26.04', 'version': '20261025.179.1', 'includedSoftwareURL': 'https://github.com/actions/runner-images/blob/ubuntu26/20261025.179/images/ubuntu/Ubuntu2604-Readme.md', 'imageReleaseURL': 'https://github.com/actions/runner-images/releases/tag/ubuntu26%2F20261025.179'}, 'environment': {'ImageOS': 'ubuntu26', 'ImageVersion': '20261026.180.1'}}, {'caseID': 'native-20261101.186.1_env-20261101.186.1', 'native': {'image': 'ubuntu-26.04', 'version': '20261101.186.1', 'includedSoftwareURL': 'https://github.com/actions/runner-images/blob/ubuntu26/20261101.186/images/ubuntu/Ubuntu2604-Readme.md', 'imageReleaseURL': 'https://github.com/actions/runner-images/releases/tag/ubuntu26%2F20261101.186'}, 'environment': {'ImageOS': 'ubuntu26', 'ImageVersion': '20261101.186.1'}}]
CASE_A, CASE_B = [c['caseID'] for c in JOINT_CASES]
# Explicit historical Case A constants retained for pure fixtures only; admission
# always matches all four observations and an explicit literal caseID.
NATIVE, ENVIRONMENT = JOINT_CASES[0]['native'], JOINT_CASES[0]['environment']
STABLE_RAW = ('sourcePath','resolvedPath','size','wholeSHA256','mode','uid','gid')
SNAPSHOT_KEYS = {'schemaVersion','policyID','policySHA256','stage','phase','runId','runAttempt','headSHA',
    'workspace','bootId','architecture','kernel','kernelVersion','logicalJobName','collectorSourceSHA256',
    'snapshotManifestSHA256','reportSHA256','stable','caseID'}
REVIEW_KEYS = {'schemaVersion','policyID','policySHA256','beforeToolsSnapshotSHA256','preparedSnapshotSHA256',
    'preparedWrapperSHA256','primaryPregameAuthority','earlyHeaderEvidence','finalExternalReconciliationRequired','caseID','phaseOrderPlanSHA256','phaseOrderReview'}
EARLY_KEYS = {'status','endpoint','runId','runAttempt','jobId','headSHA','observedAtUTC','httpStatus',
    'responseBytes','responseSHA256','responseFile','scope','runAccessVerified','jobsAccessVerified',
    'artifactAccessVerified','jobInProgress','nativeClaim'}


PHASE_PREFIX = 'KEEL_WAKE_IMAGE_PHASE_END_V1 '
PHASE_KIND = 'source-backed-causal-phase-order-v1'
WORKFLOW_PATH = '.github/workflows/keel-wake-v1.yml'
EMITTER_PATH = 'tools/keel_wake_image_identity.py'
RAW_COLLECTOR_PATH = 'tools/keel_wake_image_probe.py'
CONTROLLER_PATH = 'tools/keel_wake_experiment.py'
PHASE_MAPPING = {
    'before-tools': (3, 'Capture original qualified image observations before tools', 4, 'Set up pinned Java17'),
    'prepared': (9, 'Prepare exact sources/runtime/commands on this worker (zero games)', 10,
                 'Finalize immutable zero-game provenance before independent review')}
PLAN_KEYS = {'schemaVersion','kind','repository','issue','stage','runId','runAttempt','headSHA','workflowSource','collectorSource','phasePlans'}
PHASE_KEYS = {'phase','captureStepName','captureStepNumber','nextStepName','nextStepNumber','foregroundCommandSHA256','snapshotSHA256','markerPayload'}
PHASE_RECORD_KEYS = {'phase','captureStepName','captureStepNumber','nextStepName','nextStepNumber','snapshotSHA256','markerLineOrdinal','markerMessageSHA256','observedProviderTimestamp','causalOrderVerified'}


def compact(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def yaml_scalar(value):
    value=value.strip()
    if value.startswith('"'):
        parsed=json.loads(value)
        if type(parsed) is not str:raise ValueError('String scalar required')
        return parsed
    if value.startswith("'"):
        if not value.endswith("'"):raise ValueError('Unterminated YAML scalar')
        return value[1:-1].replace("''", "'")
    if not value or value[0] in '&*!{[>|' or ' #' in value:raise ValueError('Unsupported declared YAML scalar')
    return value


def declared_steps(raw):
    """A restricted literal block workflow, independently implemented by producer."""
    text=raw.decode('utf-8')
    if '\r' in text or '\x00' in text:raise ValueError('Workflow must have exact LF source')
    if re.search(r'^\s*(?:[\'"]?)(?:background|parallel|continue-on-error)(?:[\'"]?)\s*:',text,re.M):
        raise ValueError('Background/parallel/continued-error workflow cannot prove foreground order')
    if len(re.findall(r'^  [A-Za-z_][\w-]*:\s*$',text.split('jobs:\n',1)[-1],re.M))!=1:
        raise ValueError('One ordinary source job required')
    lines=text.splitlines(keepends=True)
    starts=[n for n,line in enumerate(lines) if line.startswith('      - ')]
    if not starts:raise ValueError('No ordinary foreground source steps')
    def environment(block, indent):
        result={}
        for n,line in enumerate(block):
            if line.rstrip()==' '*indent+'env:':
                for item in block[n+1:]:
                    if not item.strip():continue
                    if len(item)-len(item.lstrip())<=indent:break
                    match=re.fullmatch(' '*(indent+2)+r'([A-Za-z_][A-Za-z_0-9]*):\s*(.*?)\n?',item)
                    if not match or match[1] in result:raise ValueError('Malformed/duplicate declared env')
                    result[match[1]]=yaml_scalar(match[2])
        return result
    job_env=environment(lines[:starts[0]],4)
    rows=[]
    for pos,start in enumerate(starts):
        block=lines[start:starts[pos+1] if pos+1<len(starts) else len(lines)]
        block=[block[0].replace('      - ','        ',1)]+block[1:]
        row=dict(number=pos+2,name=None,uses=None,run=None,env=job_env|environment(block,8))
        for n,line in enumerate(block):
            match=re.fullmatch(r'        (name|uses|run|shell):\s*(.*?)\n?',line)
            if not match:continue
            key,value=match.groups()
            if key=='run':
                if value!='|':raise ValueError('Run must be exact clip-one-LF literal')
                content=[]
                for item in block[n+1:]:
                    if item.strip() and len(item)-len(item.lstrip())<=8:break
                    if item.strip() and not item.startswith('          '):raise ValueError('Run indentation')
                    content.append(item[10:] if item.startswith('          ') else '\n')
                row['run']=''.join(content).rstrip('\n')+'\n'
                if re.search(r'(?<![&0-9>])&(?![&0-9])|\b(?:nohup|setsid|disown|coproc|dtach|tmux|screen|parallel)\b|xargs\s+[^\n]*-P',row['run']):
                    raise ValueError('Detached/background/parallel shell cannot prove order')
            elif key=='shell':
                if yaml_scalar(value) not in ('bash','sh'):raise ValueError('Ordinary foreground shell required')
            else:
                if row[key] is not None:raise ValueError('Duplicate source step field')
                row[key]=yaml_scalar(value)
        if bool(row['uses'])==bool(row['run']):raise ValueError('Exactly one uses/run per source step')
        row['name']=row['name'] or 'Run '+(row['uses'] or row['run'].splitlines()[0])
        rows.append(row)
    return rows


def declared_command_sha(step):
    # DECLARED source literals, including unresolved expressions; NOT argv hash.
    return sha(compact(dict(run=step['run'],env=step['env'])))


def phase_payload(snapshot_value, snapshot_sha):
    return dict(stage=snapshot_value['stage'],phase=snapshot_value['phase'],runId=snapshot_value['runId'],
        runAttempt=snapshot_value['runAttempt'],headSHA=snapshot_value['headSHA'],snapshotSHA256=snapshot_sha)


def create_phase_plan(base, first, prepared):
    source=lambda path:dict(path=path,bytes=(ROOT/path).stat().st_size,sha256=sha(read(ROOT/path)))
    steps=declared_steps(read(ROOT/WORKFLOW_PATH));entries=[]
    for phase,snapshot_value in [('before-tools',first),('prepared',prepared)]:
        number,name,next_number,next_name=PHASE_MAPPING[phase]
        capture=[s for s in steps if s['number']==number];following=[s for s in steps if s['number']==next_number]
        if len(capture)!=1 or len(following)!=1 or capture[0]['name']!=name or following[0]['name']!=next_name:
            raise ValueError('Exact registered phase source mapping changed')
        digest=sha(read(Path(base)/phase/'snapshot.json'))
        entries.append(dict(phase=phase,captureStepName=name,captureStepNumber=number,nextStepName=next_name,
            nextStepNumber=next_number,foregroundCommandSHA256=declared_command_sha(capture[0]),
            snapshotSHA256=digest,markerPayload=phase_payload(snapshot_value,digest)))
    plan=dict(schemaVersion=1,kind=PHASE_KIND,repository='stephanelee/robocode',issue=118,
        **{k:first[k] for k in ('stage','runId','runAttempt','headSHA')},workflowSource=source(WORKFLOW_PATH),
        collectorSource=source(EMITTER_PATH),phasePlans=entries)
    write(Path(base)/'phase-order-plan.json',plan)
    return plan


def validate_phase_plan(plan, ident, source_reader, assets, snapshots, snapshot_hashes):
    if (type(plan) is not dict or set(plan)!=PLAN_KEYS or type(plan['schemaVersion']) is not int
            or plan['schemaVersion']!=1 or plan['kind']!=PHASE_KIND or plan['repository']!='stephanelee/robocode'
            or type(plan['issue']) is not int or plan['issue']!=118
            or any(type(plan[k]) is not type(ident[k]) or plan[k]!=ident[k] for k in ('stage','runId','runAttempt','headSHA'))):
        raise ValueError('Exact actual phase plan identity required')
    for field,path in [('workflowSource',WORKFLOW_PATH),('collectorSource',EMITTER_PATH)]:
        ref=plan[field];raw=source_reader(path)
        if (type(ref) is not dict or set(ref)!={'path','bytes','sha256'} or ref['path']!=path
                or type(ref['bytes']) is not int or ref['bytes']!=len(raw) or ref['sha256']!=sha(raw)
                or sha(raw)!=assets.get(path)):
            raise ValueError('Phase source ref/frozen source bytes differ')
    for path in (RAW_COLLECTOR_PATH,CONTROLLER_PATH):
        if sha(source_reader(path))!=assets.get(path):raise ValueError('Raw collector/controller closure differs')
    if any(s['collectorSourceSHA256']!=assets[RAW_COLLECTOR_PATH] for s in snapshots.values()):
        raise ValueError('Raw collector must bind separately from marker emitter')
    workflow=source_reader(WORKFLOW_PATH)
    if PHASE_PREFIX.encode() in workflow or PHASE_PREFIX.strip().encode() not in source_reader(EMITTER_PATH):
        raise ValueError('Actual emitter required; workflow echo is not an emitter')
    steps=declared_steps(workflow)
    if any(snapshot['stage']!=plan['stage'] or snapshot['runId']!=plan['runId'] or snapshot['runAttempt']!=1 or snapshot['headSHA']!=plan['headSHA'] for snapshot in snapshots.values()):
        raise ValueError('Actual original snapshots differ from phase plan identity')
    if type(plan['phasePlans']) is not list or [e.get('phase') for e in plan['phasePlans']]!=list(PHASE_MAPPING):
        raise ValueError('Exactly two original phase plans required')
    for entry in plan['phasePlans']:
        if type(entry) is not dict or set(entry)!=PHASE_KEYS:raise ValueError('Exact phase mapping required')
        phase=entry['phase'];number,name,next_number,next_name=PHASE_MAPPING[phase]
        capture=steps[number-2];following=steps[next_number-2]
        expected_payload=phase_payload(snapshots[phase],entry['snapshotSHA256'])
        if (type(entry['captureStepNumber']) is not int or type(entry['nextStepNumber']) is not int
                or (entry['captureStepNumber'],entry['captureStepName'],entry['nextStepNumber'],entry['nextStepName'])!=(number,name,next_number,next_name)
                or capture['name']!=name or following['name']!=next_name
                or following['uses']!=('actions/setup-java@v4' if phase=='before-tools' else 'actions/upload-artifact@v4')
                or entry['foregroundCommandSHA256']!=declared_command_sha(capture)
                or entry['snapshotSHA256']!=snapshot_hashes[phase] or entry['markerPayload']!=expected_payload or any(type(entry['markerPayload'].get(k)) is not type(v) for k,v in expected_payload.items())):
            raise ValueError('Frozen phase command/graph/marker mapping differs')
        required=('/usr/bin/python3 -I -B -S '+EMITTER_PATH) if phase=='before-tools' else ('python -I -B -S '+CONTROLLER_PATH+' prepare')
        if required not in capture['run']:raise ValueError('Isolated actual foreground source chain required')
    return steps


def verify_phase_graph(plan, jobs, job_id):
    selected=[j for j in jobs['jobs'] if j.get('id')==job_id]
    if (len(selected)!=1 or selected[0].get('run_id')!=plan['runId'] or selected[0].get('run_attempt')!=1
            or selected[0].get('head_sha')!=plan['headSHA']):raise ValueError('Actual phase job identity differs')
    job=selected[0]
    for entry in plan['phasePlans']:
        for number,name in [(entry['captureStepNumber'],entry['captureStepName']),(entry['nextStepNumber'],entry['nextStepName'])]:
            rows=[s for s in job['steps'] if s.get('number')==number]
            if (len(rows)!=1 or rows[0].get('name')!=name or rows[0].get('status')!='completed'
                    or rows[0].get('conclusion')!='success'):raise ValueError('Exact successful ordinary phase/next steps required')
    return job


def phase_log_records(plan, steps, raw):
    lines=raw.decode('utf-8-sig').splitlines();parsed=[]
    for ordinal,line in enumerate(lines,1):
        match=re.fullmatch(r'(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d{1,18})?Z) (.*)',line)
        if not match:
            if not line.strip():parsed.append((ordinal,None,''));continue
            raise ValueError('Complete original timestamped provider log required')
        # Validate calendar fields without rounding the raw fractional observation.
        dt.datetime.fromisoformat(match[1][:19]+'+00:00')
        # Preserve raw provider precision as an observation. No UTC lower bound.
        parsed.append((ordinal,match[1],match[2]))
    records=[]
    for entry in plan['phasePlans']:
        step=steps[entry['captureStepNumber']-2];following=steps[entry['nextStepNumber']-2]
        run=step['run']
        for key,value in {'github.event.pull_request.head.sha':plan['headSHA'],'github.run_id':str(plan['runId']),'env.STAGE':plan['stage']}.items():
            run=re.sub(r'\$\{\{\s*'+re.escape(key)+r'\s*\}\}',lambda m:str(value),run)
        if '${{' in run:raise ValueError('Unknown rendered foreground expression')
        run_lines=run.rstrip('\n').split('\n')
        openings=[n for n,row in enumerate(parsed) if row[2]=='##[group]Run '+run_lines[0]]
        if len(openings)!=1:raise ValueError('Unique actual capture Run group required')
        opening=openings[0]
        echo=[row[2] for row in parsed[opening+1:opening+1+len(run_lines)]]
        if echo!=['\x1b[36;1m'+line+'\x1b[0m' for line in run_lines]:raise ValueError('Complete exact frozen source echo required')
        echo_end=next((n for n in range(opening+1+len(run_lines),len(parsed)) if parsed[n][2]=='##[endgroup]'),None)
        if echo_end is None:raise ValueError('Missing Run source-echo end')
        next_group=next((n for n in range(echo_end+1,len(parsed)) if parsed[n][2].startswith('##[group]Run ')),None)
        if next_group is None or parsed[next_group][2]!='##[group]Run '+following['uses']:
            raise ValueError('Capture must precede exact next foreground source segment')
        wanted=PHASE_PREFIX+compact(entry['markerPayload']).decode()
        relevant=[(n,row) for n,row in enumerate(parsed) if PHASE_PREFIX in row[2] and ('"phase":"'+entry['phase']+'"') in row[2]]
        if len(relevant)!=1 or relevant[0][1][2]!=wanted or not echo_end<relevant[0][0]<next_group:
            raise ValueError('Missing/duplicate/quoted/echo/out-of-step phase marker')
        ordinal,observed,message=relevant[0][1]
        records.append({k:entry[k] for k in ('phase','captureStepName','captureStepNumber','nextStepName','nextStepNumber','snapshotSHA256')}|
            dict(markerLineOrdinal=ordinal,markerMessageSHA256=sha(message.encode()),observedProviderTimestamp=observed,causalOrderVerified=True))
    # Every occurrence is one of the two full actual messages, never a hidden replay.
    if sum(PHASE_PREFIX in row[2] for row in parsed)!=2:raise ValueError('Only two unique original phase messages allowed')
    return records


def phase_window_valid(start,end,step,causal_verified):
    if within_api_step(start,end,step):return True
    high=timestamp(step['completed_at']);last=timestamp(end)
    return (timestamp(step['started_at'])<=timestamp(start)<=last and whole_second(step['completed_at'])
        and high<last<high+1 and causal_verified is True)


LIVE_DEADLINE = None
LIVE_JOB_DEADLINE = None


def set_live_budget(monotonic_deadline, job_deadline):
    global LIVE_DEADLINE, LIVE_JOB_DEADLINE
    LIVE_DEADLINE, LIVE_JOB_DEADLINE = monotonic_deadline, job_deadline


def bounded_timeout(cap):
    remaining = cap
    if LIVE_DEADLINE is not None:
        remaining = min(remaining, LIVE_DEADLINE - time.monotonic())
    if LIVE_JOB_DEADLINE is not None:
        remaining = min(remaining, LIVE_JOB_DEADLINE - time.time())
    if remaining <= 0:
        raise TimeoutError('Complete live image revalidation deadline')
    return remaining


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
    bounded_timeout(120)
    with Path(path).open('rb') as f: data=f.read(cap+1)
    bounded_timeout(120)
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
    if sha(read(ROOT/POLICY_PATH)) != POLICY_SHA or sha(read(ROOT/DOCUMENT_PATH)) != DOCUMENT_SHA:
        raise ValueError('Published Wake runtime policy/protocol bytes changed')
    value = load(ROOT/POLICY_PATH)
    if value['policyID'] != POLICY_ID or value['allowedJointCases'] != JOINT_CASES or value['issue'] != 118:
        raise ValueError('Published exact Wake two-case policy changed')
    return value

def case_by_id(case_id):
    found=[c for c in JOINT_CASES if type(case_id) is str and c['caseID']==case_id]
    if len(found)!=1:raise ValueError('Unknown literal joint caseID')
    return found[0]


def joint_case(claims):
    if set(claims)!={'imagedataNative','setupInfoNative','processEnvironment','etcEnvironment'}:
        raise ValueError('All four independent observation objects required')
    found=[c for c in JOINT_CASES if all(claims[k]==c['native'] for k in ('imagedataNative','setupInfoNative'))
        and all(claims[k]==c['environment'] for k in ('processEnvironment','etcEnvironment'))]
    if len(found)!=1:raise ValueError('No single complete allowed joint case; source mixing/unknown tuple')
    return found[0]


def effective_origin():
    data=read(ROOT/EFFECTIVE_FREEZE_PATH)
    freeze=decode(data)
    if (type(freeze.get('schemaVersion')) is not int or freeze['schemaVersion']!=1
            or freeze.get('effectiveFreezePath')!=EFFECTIVE_FREEZE_PATH or freeze.get('qualifiedImagePolicyID')!=POLICY_ID
            or freeze.get('qualifiedImagePolicySHA256')!=POLICY_SHA):
        raise ValueError('Wrong effective v2 source freeze origin/policy')
    return dict(effectiveFreezePath=EFFECTIVE_FREEZE_PATH,effectiveFreezeSHA256=sha(data))


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
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True,timeout=bounded_timeout(10)).strip()
    if not re.fullmatch('[a-f0-9]{40}',head):raise ValueError('Actual checkout SHA missing')
    if os.environ.get('GITHUB_REPOSITORY')!='stephanelee/robocode' or os.environ.get('GITHUB_RUN_ATTEMPT')!='1':
        raise ValueError('Wrong repository/attempt')
    if head!=os.environ.get('GITHUB_SHA') and head!=os.environ.get('KEEL_WAKE_IMAGE_PROBE_HEAD_SHA'):
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
    if report['collectorSource']['sha256']!=sha(read(ROOT/'tools/keel_wake_image_probe.py')):
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
            if parsed['errors'] or parsed['runnerImageRecords'][0]['unrecognizedLines']:raise ValueError('Unknown native claim '+name)
            claims['imagedataNative' if name=='imagedata' else 'setupInfoNative']=parsed['claim']
        elif name=='etc_environment':
            parsed=probe.parse_environment_file(data)
            value={k:v['value'] for k,v in parsed['claims'].items() if v['present']}
            if parsed['errors'] or set(value)!={'ImageOS','ImageVersion'} or any(not v for v in value.values()):raise ValueError('Wrong separate etc environment')
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
    if set(process)!={'ImageOS','ImageVersion'} or any(process[k].get('present') is not True or not isinstance(process[k].get('value'),str) or not process[k]['value'] for k in process):
        raise ValueError('Wrong separate process environment')
    claims['processEnvironment']={k:process[k]['value'] for k in process}
    case=joint_case(claims)
    if report['platform']['architecture']!='x86_64' or report['platform']['system']!='Linux':raise ValueError('Wrong worker platform')
    return dict(caseID=case['caseID'],rawSources=raw,claims=claims,runnerRoot=root,setupInfoPath=str(Path(root)/'.setup_info'),
        bootId=boot,architecture=report['platform']['architecture'],kernel=report['platform']['kernel'],
        kernelVersion=report['platform']['kernelVersion'],collectorSourceSHA256=report['collectorSource']['sha256'])


def snapshot(base,stage,phase):
    policy();context=source_context(stage)
    if phase not in ('before-tools','prepared') and not phase.startswith('revalidation-'):raise ValueError('Unknown capture phase')
    folder=Path(base)/phase;collector_phase='before-tools' if phase=='before-tools' else 'after-tools'
    captured=folder/'collector'/collector_phase
    start=time.monotonic()
    subprocess.run(['/usr/bin/python3','-I','-B','-S',str(ROOT/'tools/keel_wake_image_probe.py'),
        '--output-dir',str(captured),'--phase',collector_phase],check=False,timeout=bounded_timeout(120))
    # A nonzero collector exit can represent only optional LSB absence; exact
    # raw/status/error validation below decides, never a return-code waiver.
    stable=verify_capture(captured)
    if time.monotonic()-start>120:raise ValueError('Total capture/validation deadline')
    stable['sourceIdentity']=context
    value=dict(schemaVersion=1,policyID=POLICY_ID,policySHA256=POLICY_SHA,caseID=stable['caseID'],phase=phase,**context,
        bootId=stable['bootId'],architecture=stable['architecture'],kernel=stable['kernel'],kernelVersion=stable['kernelVersion'],
        collectorSourceSHA256=stable['collectorSourceSHA256'],snapshotManifestSHA256=sha(read(captured/'capture-status.json')),
        reportSHA256=sha(read(captured/'report.json')),stable=stable)
    write(folder/'snapshot.json',value)
    if phase in PHASE_MAPPING:
        print(PHASE_PREFIX+compact(phase_payload(value,sha(read(folder/'snapshot.json')))).decode(),flush=True)
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
    for key in ('caseID','bootId','architecture','kernel','kernelVersion','collectorSourceSHA256'):
        if value[key] != stable[key]:raise ValueError('Snapshot top-level/derived stable mismatch')
    if (not re.fullmatch('[a-f0-9]{40}',value['headSHA']) or not isinstance(value['workspace'],str)
            or not Path(value['workspace']).is_absolute() or value['logicalJobName']!='registered-stage'):
        raise ValueError('Malformed snapshot source identity')
    report=load(captured/'report.json');context=report['context']
    expected_phase='before-tools' if phase=='before-tools' else 'after-tools'
    if (report['phase']!=expected_phase or context['GITHUB_REPOSITORY']!='stephanelee/robocode'
            or context['GITHUB_RUN_ID']!=str(value['runId']) or context['GITHUB_RUN_ATTEMPT']!='1'
            or context['GITHUB_JOB']!=value['logicalJobName'] or context['KEEL_WAKE_IMAGE_PROBE_HEAD_SHA']!=value['headSHA']):
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
    create_phase_plan(base,a,b)
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
    plan=load(base/'phase-order-plan.json')
    assets={path:sha(read(ROOT/path)) for path in (WORKFLOW_PATH,EMITTER_PATH,RAW_COLLECTOR_PATH,CONTROLLER_PATH)}
    validate_phase_plan(plan,a,lambda path:read(ROOT/path),assets,{'before-tools':a,'prepared':b},
        {phase:sha(read(base/phase/'snapshot.json')) for phase in PHASE_MAPPING})
    return dict(phaseOrderPlanSHA256=sha(read(base/'phase-order-plan.json')),policyID=POLICY_ID,policySHA256=POLICY_SHA,caseID=b['caseID'],**effective_origin(),beforeToolsSnapshotSHA256=wrapper['beforeToolsSnapshotSHA256'],
        preparedSnapshotSHA256=wrapper['preparedSnapshotSHA256'],preparedWrapperSHA256=sha(read(base/'prepared-wrapper.json')),
        preparedNumericJobId=wrapper['preparedNumericJobId'],stable=b['stable'])


def live(base,stage,output):
    expected=original(base,'prepared');current=snapshot(output,stage,'revalidation-'+str(time.monotonic_ns()))
    compare(expected,current)
    return current


def parse_provider_header(raw,case_id):
    """Only original provider startup, before the first named workflow step."""
    native=case_by_id(case_id)['native']
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
    if parsed['errors'] or parsed['unrecognizedLines'] or parsed['claim']!=native:
        raise ValueError('Provider startup conflicts/malformed qualified tuple')
    return parsed['claim']


def review(value,anchor,ident):
    if not isinstance(value,dict) or set(value)!=REVIEW_KEYS:raise ValueError('Exact image identity review required')
    for key in ('policyID','policySHA256','beforeToolsSnapshotSHA256','preparedSnapshotSHA256','preparedWrapperSHA256','phaseOrderPlanSHA256'):
        if value[key]!=anchor[key]:raise ValueError('Image review does not bind actual immutable snapshots')
    if type(value['schemaVersion']) is not int or value['schemaVersion']!=1 or value['primaryPregameAuthority']!='native-only-qualified-observation' or value['finalExternalReconciliationRequired'] is not True:
        raise ValueError('Qualified native boundary/final reconciliation not acknowledged')
    if value['caseID']!=anchor['caseID']:raise ValueError('Runtime image review changed frozen joint case')
    native=case_by_id(anchor['caseID'])['native']
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
        if early['httpStatus']!=200 or early['scope']!='complete-startup-header' or early['nativeClaim']!=native:
            raise ValueError('Available provider evidence must match native tuple')
    elif early['status']=='unavailable-404-deferred':
        if early['httpStatus']!=404 or early['scope']!='actual-job-log-request' or early['jobInProgress'] is not True or early['nativeClaim'] is not None:
            raise ValueError('Only actual in-progress HTTP404 may defer')
    else:raise ValueError('Unrecognized pregame header availability')
    phase=value['phaseOrderReview']
    wanted=dict(status='source-and-jobs-graph-pending-final-marker' if early['httpStatus']==404 else 'source-jobs-and-live-provider-markers-verified',
        providerMarkersObserved=early['httpStatus']==200,finalExternalProofRequired=True)
    if type(phase) is not dict or set(phase)!=set(wanted) or phase!=wanted or any(type(phase[k]) is not type(v) for k,v in wanted.items()):
        raise ValueError('Exact truthful pregame phase-order qualification required')
    return True


FINAL_KEYS = {'schemaVersion','status','policyID','policySHA256','runId','runAttempt','jobId','headSHA',
    'beforeToolsSnapshotSHA256','preparedSnapshotSHA256','preparedWrapperSHA256','nativeClaim','runAPI','jobsAPI',
    'jobLog','completedAtUTC','reviewerTask','caseID','reviewerIdentityEvidence','phaseOrderEvidence'}


def timestamp(value):
    """Exact UTC seconds, including the real provider's seven-digit log prefix."""
    match = re.fullmatch(r'(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)(?:\.(\d{1,9}))?(?:Z|\+00:00)', value) if isinstance(value, str) else None
    if match is None:
        raise ValueError('Unsupported or non-UTC timestamp precision')
    whole = dt.datetime.fromisoformat(match[1] + '+00:00')
    epoch = dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc)
    delta = whole - epoch
    fraction = match[2] or ''
    return Fraction(delta.days * 86400 + delta.seconds) + (Fraction(int(fraction), 10 ** len(fraction)) if fraction else 0)


def whole_second(value):
    return isinstance(value, str) and bool(re.fullmatch(r'\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:Z|\+00:00)', value))


def within_api_step(start,end,step):
    low,high=timestamp(step['started_at']),timestamp(step['completed_at'])
    first,last=timestamp(start),timestamp(end)
    return low<=first<=last<=high


def strictly_before(first, second):
    left, right = timestamp(first), timestamp(second)
    return left + 1 <= right if whole_second(first) else left < right


def verify_actual_reviewer(value, ident, read_ref, current_run):
    reference = value['reviewerIdentityEvidence']
    if not isinstance(reference, dict) or set(reference) != {'path', 'bytes', 'sha256'}:
        raise ValueError('Explicit actual reviewer sidecar reference required')
    integer(reference['bytes'], 1)
    if (not isinstance(reference['path'], str) or not reference['path'].endswith('provider/reviewer-identity.json')
            or Path(reference['path']).is_absolute() or '..' in Path(reference['path']).parts
            or not isinstance(reference['sha256'], str) or not re.fullmatch('[0-9a-f]{64}', reference['sha256'])):
        raise ValueError('Wrong actual reviewer sidecar path')
    raw = read_ref(reference['path'])
    if len(raw) != reference['bytes'] or sha(raw) != reference['sha256'] or len(raw) > probe.FILE_CAP:
        raise ValueError('Actual reviewer sidecar bytes/hash/cap differ')
    sidecar = decode(raw)
    keys = {'schemaVersion','repository','issue','stage','runId','runAttempt','headSHA','reviewerTask',
        'implementationTasks','assignmentCommentId','assignmentBodySHA256','assignmentToolRecord',
        'assignmentToolRecordSHA256','reviewToolRecord','reviewToolRecordSHA256'}
    if not isinstance(sidecar, dict) or set(sidecar) != keys:
        raise ValueError('Exact actual reviewer identity schema required')
    for name in ('schemaVersion','issue','runId','runAttempt','assignmentCommentId'):
        integer(sidecar[name], 1)
    if (sidecar['schemaVersion'] != 1 or sidecar['repository'] != 'stephanelee/robocode'
            or sidecar['issue'] != 118 or sidecar['stage'] != 'screen'
            or sidecar['runAttempt'] != 1 or sidecar['runId'] != ident['runId']
            or sidecar['headSHA'] != ident['headSHA'] or sidecar['reviewerTask'] != value['reviewerTask']):
        raise ValueError('Actual final reviewer does not bind original screen/run/head')
    tasks = sidecar['implementationTasks']
    if (not isinstance(tasks, list) or not tasks or any(not isinstance(t, str) or not t for t in tasks)
            or not isinstance(sidecar['reviewerTask'], str) or not sidecar['reviewerTask'].strip()
            or sidecar['reviewerTask'] in tasks):
        raise ValueError('Implementation/fictional actor cannot supply independent final identity')
    prefix = reference['path'][:-len('provider/reviewer-identity.json')]
    assignment_raw = read_ref(prefix + 'provider/comments/' + str(sidecar['assignmentCommentId']) + '.json')
    comment = decode(assignment_raw)
    if (comment.get('id') != sidecar['assignmentCommentId'] or comment.get('user', {}).get('login') != 'stephanelee'
            or comment.get('created_at') != comment.get('updated_at')
            or not strictly_before(comment['created_at'], current_run['created_at'])
            or sha(comment['body'].encode()) != sidecar['assignmentBodySHA256']):
        raise ValueError('Original independent predispatch comment/body/time differs')
    marker = 'KEEL_WAKE_DISPATCH_ASSIGNMENT_V1\n'
    if not comment['body'].startswith(marker):
        raise ValueError('Wrong original assignment namespace')
    assignment = decode(comment['body'][len(marker):].encode())
    required = {'schemaVersion','repository','issue','stage','headSHA','reviewerTask','reviewerAccount',
        'implementationTasks','assignmentToolRecord','assignmentToolRecordSHA256'}
    if (not isinstance(assignment, dict) or set(assignment) != required
            or type(assignment['schemaVersion']) is not int or assignment['schemaVersion'] != 1
            or type(assignment['issue']) is not int or assignment['issue'] != 118
            or assignment['reviewerAccount'] != 'stephanelee'
            or any(assignment[name] != sidecar[name] for name in ('repository','stage','headSHA','reviewerTask','implementationTasks','assignmentToolRecord','assignmentToolRecordSHA256'))):
        raise ValueError('Actual final reviewer differs from originally assigned task')
    for name in ('assignmentToolRecord','reviewToolRecord'):
        record = sidecar[name]
        if (not isinstance(record, dict) or not record or record.get('reviewerTask') != sidecar['reviewerTask']
                or sha(json.dumps(record, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()) != sidecar[name+'SHA256']):
            raise ValueError('Actual task/tool evidence hash binding required')
    if sidecar['assignmentToolRecord'].get('activeBeforeDispatch') is not True or not sidecar['reviewToolRecord'].get('toolEvidence'):
        raise ValueError('Missing active prospective assignment or actual retained final tool evidence')
    return sidecar


def final_reconciliation(value,anchor,ident,read_ref,current_run,current_jobs,capture_windows,phase_context):
    """Verify externally retained full provider evidence, never mutate raw arithmetic."""
    if not isinstance(value,dict) or set(value)!=FINAL_KEYS:raise ValueError('Separate final provider reconciliation required')
    if anchor['policyID']!=POLICY_ID or anchor['policySHA256']!=POLICY_SHA:
        raise ValueError('Final screen uses wrong qualified-runtime policy')
    for key in ('schemaVersion','runId','runAttempt','jobId'):integer(value[key],1)
    if value['caseID']!=anchor['caseID']:raise ValueError('Final reconciliation changed frozen joint case')
    native=case_by_id(anchor['caseID'])['native']
    if value['schemaVersion']!=1 or value['status']!='PASS' or value['nativeClaim']!=native:
        raise ValueError('Final provider provenance incomplete')
    for key in ('policyID','policySHA256','beforeToolsSnapshotSHA256','preparedSnapshotSHA256','preparedWrapperSHA256'):
        if value[key]!=anchor[key]:raise ValueError('Final review does not bind original image snapshots')
    if (not isinstance(value['reviewerTask'],str) or not value['reviewerTask'].strip()
            or value['runId']!=ident['runId'] or value['headSHA']!=ident['headSHA'] or value['runAttempt']!=1
            or value['jobId']!=anchor['preparedNumericJobId'] or not value['reviewerTask']):
        raise ValueError('Final provider review wrong actual worker identity')
    verify_actual_reviewer(value, ident, read_ref, current_run)
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
    log=evidence['jobLog'];parse_provider_header(log,anchor['caseID'])
    ref=value['phaseOrderEvidence']
    if type(ref) is not dict or set(ref)!={'path','bytes','sha256'} or type(ref['bytes']) is not int:
        raise ValueError('Exact independent final phase proof reference required')
    raw=read_ref(ref['path'])
    if len(raw)!=ref['bytes'] or sha(raw)!=ref['sha256']:raise ValueError('Original phase proof bytes differ')
    proof=decode(raw)
    proof_keys={'schemaVersion','kind','phaseOrderPlanSHA256','jobId','runId','runAttempt','headSHA','jobLog','jobsAPI','phaseRecords'}
    if (type(proof) is not dict or set(proof)!=proof_keys or type(proof['schemaVersion']) is not int or proof['schemaVersion']!=1
            or proof['kind']!=PHASE_KIND or proof['phaseOrderPlanSHA256']!=anchor['phaseOrderPlanSHA256']
            or any(type(proof[k]) is not type(value[k]) or proof[k]!=value[k] for k in ('jobId','runId','runAttempt','headSHA'))):
        raise ValueError('Independent final phase proof identity differs')
    for role,logical in [('jobLog','provider/job.log'),('jobsAPI','provider/jobs.json')]:
        source_ref=proof[role]
        if (type(source_ref) is not dict or set(source_ref)!={'path','bytes','sha256'}
                or source_ref['path']!=logical or type(source_ref['bytes']) is not int
                or source_ref['bytes']!=len(evidence[role]) or source_ref['sha256']!=sha(evidence[role])
                or source_ref['bytes']!=value[role]['bytes'] or source_ref['sha256']!=value[role]['sha256']):
            raise ValueError('Original logical phase proof body/role differs after relocation')
    plan=phase_context['plan']
    if sha(phase_context['planBytes'])!=anchor['phaseOrderPlanSHA256'] or decode(phase_context['planBytes'])!=plan:
        raise ValueError('Original approval-bound phase plan differs')
    steps=validate_phase_plan(plan,dict(stage=plan['stage'],runId=ident['runId'],runAttempt=1,headSHA=ident['headSHA']),
        phase_context['source'].__getitem__,phase_context['assets'],phase_context['snapshots'],phase_context['snapshotHashes'])
    verify_phase_graph(plan,dict(jobs=jobs),value['jobId'])
    verify_phase_graph(plan,current_jobs,value['jobId'])
    records=phase_log_records(plan,steps,log)
    if (type(proof['phaseRecords']) is not list or len(proof['phaseRecords'])!=2
            or any(type(row) is not dict or set(row)!=PHASE_RECORD_KEYS for row in proof['phaseRecords'])
            or proof['phaseRecords']!=records
            or any(any(type(actual[k]) is not type(expected[k]) for k in PHASE_RECORD_KEYS) for actual,expected in zip(proof['phaseRecords'],records))):
        raise ValueError('Independent scoped marker/ordinal/message proof differs')
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
            if not phase_window_valid(start,end,steps[0],True):
                raise ValueError('Original image capture UTC outside actual provider step; no valid causal alternative')
        prior=timestamp(steps[0]['completed_at'])
    if set(capture_windows)!=set(required[:2]):raise ValueError('Both original image capture UTC windows required')
    return True


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('screen','confirmation'));p.add_argument('phase',choices=('before-tools',));args=p.parse_args()
    snapshot(ROOT/'.benchmark-cache/keel-wake-image-identity'/args.stage,args.stage,args.phase)


if __name__=='__main__':main()
