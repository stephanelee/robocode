"""Synthetic no-engine source/API/log fixture wiring for phase-order verification."""
import copy
import datetime
import hashlib
import json
from pathlib import Path
import phase_order as phase


def add(e,p,head,run,job,stage,early_status,causal_overlap=False):
    import test_audit as t
    a=t.audit;dump=t.dump;pro=e/'provenance';image=pro/'image-identity';source=pro/'source'
    at=lambda minute,second=0:(datetime.datetime(2026,11,5,tzinfo=datetime.timezone.utc)+datetime.timedelta(minutes=minute,seconds=second)).isoformat()
    raw=b'PURE SYNTHETIC COLLECTOR BYTES; NOT EXECUTED'
    workflow='''name: PURE SYNTHETIC WORKFLOW; NEVER EXECUTED
on: push
jobs:
  registered-stage:
    runs-on: ubuntu-26.04
    env:
      STAGE: ${{ env.REQUESTED_STAGE }}
      KEEL_WAKE_IMAGE_PROBE_HEAD_SHA: ${{ github.event.pull_request.head.sha }}
      GH_TOKEN: ${{ github.token }}
    steps:
      - name: Checkout exact source
        uses: actions/checkout@v4
      - name: Capture original qualified image observations before tools
        run: |
          /usr/bin/python3 -I -B -S tools/keel_wake_image_identity.py before-tools --stage "$STAGE"
      - name: Set up pinned Java17
        uses: actions/setup-java@v4
      - name: Set up pinned Gradle
        uses: gradle/actions/setup-gradle@v4
      - name: Set up pinned Python
        uses: actions/setup-python@v5
      - name: Build exact Java17 sources and parity
        run: |
          echo PURE_FIXTURE_NO_BUILD_EXECUTED
      - name: Set up pinned Java25
        uses: actions/setup-java@v4
      - name: Prepare exact sources/runtime/commands on this worker (zero games)
        run: |
          python3 tools/keel_wake_experiment.py prepare --stage "$STAGE"
      - name: Finalize immutable zero-game provenance before independent review
        uses: actions/upload-artifact@v4
      - name: Run fixed population
        run: |
          echo PURE_FIXTURE_NO_GAME_EXECUTED
'''.encode()
    emitter=("# PURE SYNTHETIC INTERFACE SOURCE, NOT ACTUAL RUNTIME\nPREFIX='"+phase.PREFIX+"'\ndef emit(payload):\n    print(PREFIX+payload,flush=True)\n").encode()
    controller=b'# PURE SYNTHETIC CONTROLLER INTERFACE, NEVER EXECUTED\nimport keel_wake_image_identity\n'
    freeze=a.read_json(pro/'freeze.json')
    for rel,data in [(phase.WORKFLOW,workflow),(phase.EMITTER,emitter),(phase.RAW_COLLECTOR,raw),(phase.CONTROLLER,controller)]:
        target=source/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data);freeze['assets'][rel]=phase.sha(data)
    repo=Path(__file__).resolve().parents[3]
    for name in ['phase_order.py','launcher_compilation.py']:
        rel='benchmarks/keel-wake-v1/independent-analysis/'+name;data=(repo/rel).read_bytes();target=source/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data);freeze['assets'][rel]=phase.sha(data)
    dump(pro/'freeze.json',freeze);(source/a.EFFECTIVE_FREEZE_PATH).write_bytes((pro/'freeze.json').read_bytes());fs=a.digest(pro/'freeze.json')
    origin=a.read_json(pro/'effective-freeze-origin.json');origin['effectiveFreezeSHA256']=fs;dump(pro/'effective-freeze-origin.json',origin)
    steps=phase.source_steps(workflow);entries=[]
    snapshots={name:a.read_json(image/name/'snapshot.json') for name in ['before-tools','prepared']}
    for name,n in [('before-tools',3),('prepared',9)]:
        src=next(v for v in steps if v['number']==n);later=next(v for v in steps if v['number']==n+1);snapshot_sha=a.digest(image/name/'snapshot.json');payload=dict(stage=stage,phase=name,runId=run,runAttempt=1,headSHA=head,snapshotSHA256=snapshot_sha)
        entries.append(dict(phase=name,captureStepName=src['name'],captureStepNumber=n,nextStepName=later['name'],nextStepNumber=n+1,foregroundCommandSHA256=phase.declared_digest(src),snapshotSHA256=snapshot_sha,markerPayload=payload))
    plan=dict(schemaVersion=1,kind=phase.KIND,repository='stephanelee/robocode',issue=118,stage=stage,runId=run,runAttempt=1,headSHA=head,workflowSource=dict(path=phase.WORKFLOW,bytes=len(workflow),sha256=phase.sha(workflow)),collectorSource=dict(path=phase.EMITTER,bytes=len(emitter),sha256=phase.sha(emitter)),phasePlans=entries);dump(image/'phase-order-plan.json',plan)
    profile=a.read_json(pro/'profile.json');profile['imageIdentity'].update(effectiveFreezeSHA256=fs,phaseOrderPlanSHA256=a.digest(image/'phase-order-plan.json'));dump(pro/'profile.json',profile)
    # Full native log: preserve original header content, wrap each physical line
    # with real-shape seven-fraction provider transport. No UTC completion claim.
    old=(p/'job.log').read_text().splitlines();lines=[]
    stamp=lambda minute,second=0:at(minute,second).replace('+00:00','.1234567Z')
    def emit(message,minute,second=0):lines.append(stamp(minute,second)+' '+message)
    for line in old:emit(line,1)
    minutes={2:1,3:2,4:3,5:3,6:3,7:3,8:3,9:4,10:5,11:6}
    seconds={2:0,3:2,4:0,5:10,6:20,7:30,8:40,9:2,10:0,11:1}
    for row in steps:
        minute=minutes[row['number']];second=seconds[row['number']]
        if row['uses']:
            emit('##[group]Run '+row['uses'],minute,second);emit('with:',minute,second);emit('##[endgroup]',minute,second)
        else:
            script=phase.rendered_run(row['run'],plan).splitlines();emit('##[group]Run '+script[0],minute,second)
            for line in script:emit('\x1b[36;1m'+line+'\x1b[0m',minute,second)
            emit('shell: /usr/bin/bash -e {0}',minute,second);emit('##[endgroup]',minute,second)
            match=[v for v in entries if v['captureStepNumber']==row['number']]
            if match:emit(phase.PREFIX+phase.canonical(match[0]['markerPayload']).decode(),minute,32 if row['number']==9 else second)
    (p/'job.log').write_text('\n'.join(lines)+'\n')
    final_jobs=a.read_json(p/'jobs.json');worker=final_jobs['jobs'][0]
    api_steps=[dict(number=1,name='Set up job',status='completed',conclusion='success',started_at=at(1),completed_at=at(1))]
    for row in steps:
        if row['number']==11:continue
        start=(2,0) if row['number']==3 else (4,0) if row['number']==9 else (minutes[row['number']],seconds[row['number']])
        end=(3,0) if row['number']==3 else (5,0) if row['number']==9 else (6,0) if row['number']==10 else (minutes[row['number']],seconds[row['number']]+1)
        if causal_overlap and row['number'] in (3,9):end=(2,0) if row['number']==3 else (4,30)
        api_steps.append(dict(number=row['number'],name=row['name'],status='completed',conclusion='success',started_at=at(*start),completed_at=at(*end)))
    worker['steps']=api_steps;dump(p/'jobs.json',final_jobs)
    early_jobs=a.read_json(p/'early/jobs.json');early_jobs['jobs'][0]['steps']=copy.deepcopy(api_steps);dump(p/'early/jobs.json',early_jobs)
    comment=a.read_json(e/'review-report-comment.json');body=json.loads(comment['body'].split('\n',1)[1]);review=body['agentToolProvenance']['imageIdentityReview'];review.update(phaseOrderPlanSHA256=a.digest(image/'phase-order-plan.json'),phaseOrderReview=dict(status='source-and-jobs-graph-pending-final-marker' if early_status==404 else 'source-jobs-and-live-provider-markers-verified',providerMarkersObserved=early_status==200,finalExternalProofRequired=True))
    if early_status==200:
        response=(p/'job.log').read_bytes();(p/'early/early-header-response.bin').write_bytes(response);review['earlyHeaderEvidence'].update(responseBytes=len(response),responseSHA256=phase.sha(response),scope='complete-current-job-log')
    comment['body']='KEEL_WAKE_RUNTIME_REVIEW_V1\n'+json.dumps(body);dump(e/'review-report-comment.json',comment)
    meta=a.read_json(p/'job-log-api.json');meta['logSHA256']=a.digest(p/'job.log');dump(p/'job-log-api.json',meta)
    for path in (e/'approval-revalidations').glob('*.json'):
        data=a.read_json(path);data['imageIdentityRevalidation']['runtimeProfileSHA256']=a.digest(pro/'profile.json');dump(path,data)
    # Independently computed facts, no reviewer identity or claim of admission.
    expected=dict(stage=stage,runId=run,runAttempt=1,headSHA=head,jobId=job,snapshotSHA256={name:a.digest(image/name/'snapshot.json') for name in snapshots})
    bindings=phase.verify_plan(plan,source,freeze['assets'],expected,snapshots,final_jobs);records=phase.scoped_markers((p/'job.log').read_bytes(),plan,bindings)
    proof=dict(schemaVersion=1,kind=phase.KIND,phaseOrderPlanSHA256=a.digest(image/'phase-order-plan.json'),jobId=job,runId=run,runAttempt=1,headSHA=head,jobLog=dict(path='provider/job.log',bytes=(p/'job.log').stat().st_size,sha256=a.digest(p/'job.log')),jobsAPI=dict(path='provider/jobs.json',bytes=(p/'jobs.json').stat().st_size,sha256=a.digest(p/'jobs.json')),phaseRecords=records);dump(p/'phase-order-proof.json',proof)
