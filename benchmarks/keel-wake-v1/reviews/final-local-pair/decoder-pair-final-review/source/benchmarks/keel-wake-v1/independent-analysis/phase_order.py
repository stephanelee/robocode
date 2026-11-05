"""Independent stdlib source/event-order evidence verifier. No process/network code.

The command digest describes DECLARED frozen YAML run/job+step env literals;
it is not an executed argv hash. Marker timestamps are opaque observations.
"""
import datetime
from fractions import Fraction
import hashlib
import json
import re

PREFIX='KEEL_WAKE_IMAGE_PHASE_END_V1 '
KIND='source-backed-causal-phase-order-v1'
PLAN_KEYS={'schemaVersion','kind','repository','issue','stage','runId','runAttempt','headSHA','workflowSource','collectorSource','phasePlans'}
PHASE_KEYS={'phase','captureStepName','captureStepNumber','nextStepName','nextStepNumber','foregroundCommandSHA256','snapshotSHA256','markerPayload'}
PAYLOAD_KEYS={'stage','phase','runId','runAttempt','headSHA','snapshotSHA256'}
REVIEW_KEYS={'status','providerMarkersObserved','finalExternalProofRequired'}
PROOF_KEYS={'schemaVersion','kind','phaseOrderPlanSHA256','jobId','runId','runAttempt','headSHA','jobLog','jobsAPI','phaseRecords'}
RECORD_KEYS={'phase','captureStepName','captureStepNumber','nextStepName','nextStepNumber','snapshotSHA256','markerLineOrdinal','markerMessageSHA256','observedProviderTimestamp','causalOrderVerified'}
WORKFLOW='.github/workflows/keel-wake-v1.yml'
EMITTER='tools/keel_wake_image_identity.py'
RAW_COLLECTOR='tools/keel_wake_image_probe.py'
CONTROLLER='tools/keel_wake_experiment.py'


def need(ok,message):
    if not ok:raise ValueError(message)


def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def sha(b):return hashlib.sha256(b).hexdigest()


def instant(value):
    """Exact fractional UTC seconds; distinguish literal whole from .0000000."""
    need(type(value) is str,'UTC must be a string')
    m=re.fullmatch(r'(\d{4})-(\d\d)-(\d\d)[T ](\d\d):(\d\d):(\d\d)(?:\.(\d{1,18}))?(?:Z|\+00:00)',value)
    need(m is not None,'Unsupported exact UTC format or precision')
    stamp=datetime.datetime(*(int(m[i]) for i in range(1,7)),tzinfo=datetime.timezone.utc)
    delta=stamp-datetime.datetime(1970,1,1,tzinfo=datetime.timezone.utc)
    return Fraction(delta.days*86400+delta.seconds)+(Fraction(int(m[7]),10**len(m[7])) if m[7] else 0),m[7] is None


def strictly_before(a,b):
    x,whole=instant(a);y,_=instant(b)
    return x+1<=y if whole else x<y


def no_later_than_exact_deadline(a,deadline):
    x,whole=instant(a)
    return x+1<=deadline if whole else x<=deadline


def scalar(text):
    text=text.strip()
    if text.startswith('"'):
        value=json.loads(text);need(type(value) is str,'Only string YAML scalar subset supported');return value
    if text.startswith("'"):
        need(text.endswith("'"),'Unfinished quoted YAML string');return text[1:-1].replace("''", "'")
    need(text and ' #' not in text and text[0] not in '&*!{[>|','Unsupported/ambiguous declared YAML scalar')
    return text


def env_block(lines,index,indent):
    env={}
    for line in lines[index+1:]:
        if not line.strip() or line.lstrip().startswith('#'):continue
        leading=len(line)-len(line.lstrip())
        if leading<=indent:break
        need(leading==indent+2,'Unsupported nested/ambiguous declared environment')
        m=re.fullmatch(r'\s*([A-Za-z_][A-Za-z_0-9]*):\s*(.*?)\n?',line)
        need(m is not None and m[1] not in env,'Duplicate/malformed declared environment key')
        env[m[1]]=scalar(m[2])
    return env


def source_steps(raw):
    """Strict, intentionally small block-YAML workflow subset, fail closed."""
    need(b'\r' not in raw and b'\x00' not in raw,'Frozen workflow must be UTF8/LF block YAML')
    text=raw.decode('utf-8');lines=text.splitlines(keepends=True)
    # Background/parallel/continued-error cannot establish an ordinary foreground chain.
    no_comments='\n'.join(line for line in text.splitlines() if not line.lstrip().startswith('#'))
    need(re.search(r'''(?:^|[\s{,\-])['"]?(?:background|parallel|continue-on-error)['"]?\s*:''',no_comments,re.M) is None,'Workflow background/parallel/continue-on-error forbidden for causal proof')
    need(len(re.findall(r'^jobs:\s*$',text,re.M))==1,'One literal jobs mapping required')
    jobs_start=next(i for i,line in enumerate(lines) if re.fullmatch(r'jobs:\s*\n?',line))
    jobs_end=next((i for i in range(jobs_start+1,len(lines)) if lines[i].strip() and not lines[i].startswith((' ','#'))),len(lines))
    job_heads=[i for i in range(jobs_start+1,jobs_end) if re.fullmatch(r'  [A-Za-z_][A-Za-z_0-9-]*:\s*\n?',lines[i])]
    need(len(job_heads)==1,'Exactly one frozen worker job, no parallel worker branches')
    step_heads=[i for i in range(job_heads[0]+1,jobs_end) if re.fullmatch(r'    steps:\s*\n?',lines[i])]
    need(len(step_heads)==1,'One literal job steps block required')
    for line in lines:
        if re.match(r'^ {4}(?:[\"\']?env[\"\']?)\s*:',line):need(re.fullmatch(r'    env:\s*\n?',line) is not None,'Only literal block job env supported')
        if re.match(r'^ {8}(?:[\"\']?env[\"\']?)\s*:',line):need(re.fullmatch(r'        env:\s*\n?',line) is not None,'Only literal block step env supported')
    job_env={};job_env_seen=False
    for i,line in enumerate(lines[job_heads[0]+1:step_heads[0]],job_heads[0]+1):
        if re.fullmatch(r'    env:\s*\n?',line):
            need(not job_env_seen,'Duplicate job env');job_env_seen=True;job_env=env_block(lines,i,4)
    starts=[i for i in range(step_heads[0]+1,jobs_end) if re.match(r'^      - [A-Za-z_][A-Za-z_0-9-]*:',lines[i])]
    need(starts,'No literal foreground steps')
    steps=[]
    for n,start in enumerate(starts):
        end=starts[n+1] if n+1<len(starts) else jobs_end
        block=lines[start:end];first=block[0].replace('      - ','        ',1);block=[first]+block[1:]
        row={'number':n+2,'name':None,'uses':None,'run':None,'env':dict(job_env)}
        for i,line in enumerate(block):
            m=re.match(r'^        (name|uses|run|shell):\s*(.*?)\n?$',line)
            if m:
                key,value=m[1],m[2]
                if key=='run':
                    need(value=='|','Only exact run: | clipping convention supported')
                    content=[]
                    for part in block[i+1:]:
                        if part.strip() and len(part)-len(part.lstrip())<=8:break
                        need(not part.strip() or part.startswith('          '),'Ambiguous run literal indentation')
                        content.append(part[10:] if part.startswith('          ') else '\n')
                    row['run']=''.join(content).rstrip('\n')+'\n'
                elif key=='shell':need(scalar(value) in ('bash','sh'),'Only ordinary foreground bash/sh source supported')
                else:
                    need(row[key] is None,'Duplicate source step field');row[key]=scalar(value)
            elif re.fullmatch(r'        env:\s*\n?',line):row['env'].update(env_block(block,i,8))
        need(bool(row['uses'])!=bool(row['run']),'Exactly one uses/run field per source step')
        if row['run']:
            command=row['run']
            need(re.search(r'(?<![&0-9>])&(?![&0-9])|\b(?:nohup|setsid|disown|coproc|dtach|tmux|screen|parallel)\b|xargs\s+[^\n]*-P',command) is None,'Shell background/detach/parallel command forbidden')
        row['name']=row['name'] or ('Run '+(row['uses'] or row['run'].splitlines()[0]))
        steps.append(row)
    return steps


def declared_digest(step):return sha(canonical({'run':step['run'],'env':step['env']}))


def rendered_run(run,plan):
    known={'github.event.pull_request.head.sha':plan['headSHA'],'github.run_id':str(plan['runId']),'env.STAGE':plan['stage']}
    def replace(m):
        key=m[1].strip();need(key in known,'Unsupported inline workflow expression in foreground echo');return known[key]
    return re.sub(r'\$\{\{\s*(.*?)\s*\}\}',replace,run)


def raw_lines(raw):
    """No timestamp arithmetic. Keep seven-digit provider prefixes verbatim."""
    result=[]
    physical=raw.decode('utf-8-sig').split('\n')
    if physical and physical[-1]=='':physical.pop()
    for ordinal,line in enumerate(physical,1):
        if line.endswith('\r'):line=line[:-1]
        step=None
        if not re.match(r'^\d{4}-\d\d-\d\dT',line) and '\t' in line:
            fields=line.split('\t',2)
            need(len(fields)==3,'Ambiguous provider transport tabs');_,step,line=fields
        m=re.fullmatch(r'(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d{1,18})?Z) (.*)',line)
        # Blank unframed lines count in ordinal but carry no trusted message.
        if not m:
            need(not line.strip(),'Missing original full provider timestamp framing')
            result.append((ordinal,None,'',step));continue
        instant(m[1]) # exact format/date only; no process-completion inference
        result.append((ordinal,m[1],m[2],step))
    return result


def verify_plan(plan,source_root,assets,expected,snapshots,jobs):
    need(type(plan) is dict and set(plan)==PLAN_KEYS,'Exact noncyclic phase-order plan')
    need(type(plan['schemaVersion']) is int and plan['schemaVersion']==1 and plan['kind']==KIND and plan['repository']=='stephanelee/robocode' and type(plan['issue']) is int and plan['issue']==118,'Actual Wake phase policy identity')
    for key in ('stage','runId','runAttempt','headSHA'):need(type(plan[key]) is type(expected[key]) and plan[key]==expected[key],'Phase plan actual '+key)
    for key,path in [('workflowSource',WORKFLOW),('collectorSource',EMITTER)]:
        ref=plan[key];need(type(ref) is dict and set(ref)=={'path','bytes','sha256'} and ref['path']==path and type(ref['bytes']) is int,'Exact phase source reference')
        raw=(source_root/path).read_bytes();need(len(raw)==ref['bytes'] and sha(raw)==ref['sha256']==assets.get(path),'Frozen phase source closure '+path)
    for path in (RAW_COLLECTOR,CONTROLLER):need(path in assets and sha((source_root/path).read_bytes())==assets[path],'Actual raw collector/controller source closure '+path)
    need(assets[RAW_COLLECTOR]==snapshots['before-tools']['collectorSourceSHA256']==snapshots['prepared']['collectorSourceSHA256'],'Do not conflate raw collector and marker emitter hashes')
    workflow=(source_root/WORKFLOW).read_bytes()
    need(PREFIX.encode() not in workflow,'Workflow echo-only marker source forbidden')
    need(PREFIX.strip().encode() in (source_root/EMITTER).read_bytes(),'Actual frozen marker emitter source literal required')
    steps=source_steps(workflow)
    need(type(plan['phasePlans']) is list and all(type(v) is dict for v in plan['phasePlans']) and [v.get('phase') for v in plan['phasePlans']]==['before-tools','prepared'],'Exactly two original ordered capture phases')
    job_id=expected['jobId'];selected=[j for j in jobs['jobs'] if j['id']==job_id]
    need(len(selected)==1,'Actual numeric worker job for causal source graph');job=selected[0]
    need(type(job['id']) is int and type(job['run_id']) is int and type(job['run_attempt']) is int and job['run_id']==plan['runId'] and job['run_attempt']==1,'Actual phase worker run/attempt')
    bindings=[]
    for entry in plan['phasePlans']:
        need(type(entry) is dict and set(entry)==PHASE_KEYS,'Exact phase source-step mapping')
        phase=entry['phase'];number=entry['captureStepNumber'];next_number=entry['nextStepNumber']
        need(type(number) is int and type(next_number) is int and (number,next_number)==({'before-tools':(3,4),'prepared':(9,10)}[phase]),'Exact adjacent original before-tools3->4/prepared9->10 foreground steps required')
        source=[v for v in steps if v['number']==number];next_source=[v for v in steps if v['number']==next_number]
        need(len(source)==len(next_source)==1 and source[0]['name']==entry['captureStepName'] and next_source[0]['name']==entry['nextStepName'] and source[0]['run'] is not None,'Frozen exact capture/next source mapping')
        need(declared_digest(source[0])==entry['foregroundCommandSHA256'],'Exact DECLARED YAML run/job+step env digest, not self-reported argv')
        need(next_source[0]['uses']==('actions/setup-java@v4' if phase=='before-tools' else 'actions/upload-artifact@v4'),'Exact next ordinary setup/upload source action')
        need((('/usr/bin/python3 -I -B -S '+EMITTER) in source[0]['run']) if phase=='before-tools' else CONTROLLER in source[0]['run'],'Actual declared foreground phase helper/controller chain')
        payload=entry['markerPayload'];need(type(payload) is dict and set(payload)==PAYLOAD_KEYS,'Exact six-field phase-END payload')
        wanted=dict(stage=plan['stage'],phase=phase,runId=plan['runId'],runAttempt=1,headSHA=plan['headSHA'],snapshotSHA256=entry['snapshotSHA256'])
        need(all(type(payload[k]) is type(v) and payload[k]==v for k,v in wanted.items()),'Actual phase-END identity')
        need(entry['snapshotSHA256']==expected['snapshotSHA256'][phase],'Original whole snapshot bytes SHA, never normalized JSON substitution')
        actual=[v for v in job['steps'] if v.get('number')==number];later=[v for v in job['steps'] if v.get('number')==next_number]
        need(len(actual)==len(later)==1 and actual[0]['name']==entry['captureStepName'] and later[0]['name']==entry['nextStepName'],'Actual provider exact source phase/next step')
        need(all(v['status']=='completed' and v['conclusion']=='success' for v in (actual[0],later[0])),'Capture and sequential next step completed-success required')
        bindings.append((entry,source[0],next_source[0],actual[0],later[0]))
    return bindings


def scoped_markers(raw,plan,bindings):
    lines=raw_lines(raw);prefixes=[row for row in lines if row[2].startswith(PREFIX)]
    need(len(prefixes)==2,'Exactly two actual unescaped phase-END provider messages')
    records=[]
    for entry,source,next_source,actual,later in bindings:
        message=PREFIX+canonical(entry['markerPayload']).decode()
        matches=[row for row in prefixes if row[2]==message]
        need(len(matches)==1,'Unique actual canonical phase-END message, not quoted/echo-only/duplicate')
        marker=matches[0];ordinal=marker[0]
        # Annotated original CLI logs bind actual provider step label directly.
        if marker[3] is not None:
            need(marker[3]==entry['captureStepName'],'Marker in exact actual capture-step scope')
            later_rows=[row for row in lines if row[0]>ordinal and row[3]==entry['nextStepName']]
            need(later_rows,'Actual subsequent next-step source segment retained')
        else:
            # Native GHA shell Run group echoes the exact frozen source script.
            rendered=rendered_run(source['run'],plan).splitlines()
            starts=[i for i,row in enumerate(lines) if row[2]=='##[group]Run '+rendered[0]]
            scoped=[]
            for index in starts:
                end=index+1
                while end<len(lines) and lines[end][2]!='##[endgroup]':end+=1
                if end>=len(lines):continue
                echo=[]
                for row in lines[index+1:end]:
                    m=re.fullmatch(r'\x1b\[36;1m(.*)\x1b\[0m',row[2])
                    if m:echo.append(m[1])
                if echo!=rendered:continue
                next_run=next((i for i in range(end+1,len(lines)) if lines[i][2].startswith('##[group]Run ')),None)
                if next_run is None:continue
                expected_next=next_source['uses'] or rendered_run(next_source['run'],plan).splitlines()[0]
                if lines[next_run][2]!='##[group]Run '+expected_next:continue
                if lines[end][0]<ordinal<lines[next_run][0]:scoped.append((index,end,next_run))
            need(len(scoped)==1,'Actual marker must follow exact foreground echo end and precede exact next source Run group')
        records.append(dict(phase=entry['phase'],captureStepName=entry['captureStepName'],captureStepNumber=entry['captureStepNumber'],nextStepName=entry['nextStepName'],nextStepNumber=entry['nextStepNumber'],snapshotSHA256=entry['snapshotSHA256'],markerLineOrdinal=ordinal,markerMessageSHA256=sha(message.encode()),observedProviderTimestamp=marker[1],causalOrderVerified=True))
    return records


def contains_capture(step,start,end,causal=False):
    a,_=instant(start);b,_=instant(end);lo,_=instant(step['started_at']);finish,whole=instant(step['completed_at'])
    need(lo<=a<=b,'Capture before source-step recorded start or reversed UTC')
    if b<=finish:return 'direct-nonoverlap'
    need(whole and finish<b<finish+1,'Known fractional or definitely-late completion cannot use causal alternative')
    need(causal,'Whole-second overlap needs exact independently retained causal phase proof')
    return KIND
