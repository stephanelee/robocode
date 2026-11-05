"""Pure synthetic full-byte evidence fixture. Never starts an engine, bot or network.

Explicitly not actual runtime/admission evidence. Real frozen Keel/AIS/source bytes
are copied read-only solely to exercise the unchanged source binding verifier.
"""
import copy
import datetime
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import test_audit as t

audit=t.audit
dump=t.dump
sha=lambda b:hashlib.sha256(b).hexdigest()


def full_fixture(root,stage='screen',run_id=7,job_id=83,capture_end_second=1,causal_overlap=False,early_status=404):
    repo=Path(__file__).resolve().parents[3]
    e,p,head,run,snapshot=t.qualified_image_fixture(root,early_status=early_status,stage=stage,return_snapshot=True,run_id=run_id,job_id=job_id,capture_end_second=capture_end_second,causal_overlap=causal_overlap)
    pro=e/'provenance';source=pro/'source';image=pro/'image-identity';job=job_id
    at=lambda minute,second=0:(datetime.datetime(2026,11,5,tzinfo=datetime.timezone.utc)+datetime.timedelta(minutes=minute,seconds=second)).isoformat()
    population=audit.read_json(repo/'benchmarks/keel-wake-v1/games.json')[stage]
    # Replace source-only fixture's tiny population with the exact registered population.
    files=['docs/improvements/keel-wake-v1.md','benchmarks/keel-wake-v1/games.json','benchmarks/keel-wake-v1/runtime-pins.json','benchmarks/keel-wake-v1/baseline.json','benchmarks/keel-wake-v1/runtime-identity-policy.json',audit.AUDIT_RELATIVE_PATH,'tools/KeelWakeBattle.java']
    baseline=audit.read_json(repo/'benchmarks/keel-wake-v1/baseline.json')
    files += [v['path'] for v in baseline['files']+baseline['productionAISurvivor'] if v['path'].startswith(('experiments/Keel/','AISurvivor/'))]
    freeze=audit.read_json(pro/'freeze.json')
    for rel in sorted(set(files)):
        raw=(repo/rel).read_bytes();target=source/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw);freeze['assets'][rel]=sha(raw)
    # This is opaque synthetic fixture data, NOT a built candidate or claimed candidate hash.
    rel='experiments/KeelWake/KeelWake.jar';raw=b'PURE FIXTURE OPAQUE CANDIDATE BYTES; NEVER EXECUTED'
    target=source/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw);freeze['assets'][rel]=sha(raw)
    collector=b'PURE SYNTHETIC COLLECTOR BYTES; NOT EXECUTED';target=source/'tools/keel_wake_image_probe.py';target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(collector)
    dump(pro/'freeze.json',freeze);target=source/audit.EFFECTIVE_FREEZE_PATH;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((pro/'freeze.json').read_bytes());freeze_sha=audit.digest(target)
    origin=audit.read_json(pro/'effective-freeze-origin.json');origin['effectiveFreezeSHA256']=freeze_sha;dump(pro/'effective-freeze-origin.json',origin)
    profile=audit.read_json(pro/'profile.json');profile['imageIdentity']['effectiveFreezeSHA256']=freeze_sha
    profile['identity'].update(repository='stephanelee/robocode',headSHA=head,pullNumber=119,runId=run,runAttempt=1,architecture='x86_64',os='ID=ubuntu\nVERSION_ID="26.04"',pythonVersion='3.13.17 PURE SYNTHETIC FIXTURE')
    profile['sourceAssets']=freeze['assets'];profile['runner']='02f6d1d8e9346a4aeae1b91e7abb278c98ac1f1d30c605f867758566f5f327cf';profile['java17']={'version':'Temurin17.0.21+9 PURE FIXTURE'};profile['java25']={'version':'Temurin25.0.5+8 PURE FIXTURE','home':'/synthetic/jdk25','files':{'bin/javac':{'sha256':'1'*64}}}
    dump(pro/'python-runtime.json',{'scope':'PURE SYNTHETIC FINGERPRINT, not actual runtime'});profile['pythonRuntime']=audit.read_json(pro/'python-runtime.json');profile['pythonRuntimeSHA256']=audit.digest(pro/'python-runtime.json');(pro/'build.log').write_text('PURE SYNTHETIC BUILD LOG; no build executed');profile['buildLogSHA256']=audit.digest(pro/'build.log')
    # Opaque compiler-output fixture, never a javac/JAR/main execution.
    cr=pro/'launcher-compilation';(cr/'classes').mkdir(parents=True,exist_ok=True);(cr/'stdout.bin').write_bytes(b'');(cr/'stderr.bin').write_bytes(b'');(cr/'classes/KeelWakeBattle.class').write_bytes(b'PURE SYNTHETIC CLASS OUTPUT; NEVER EXECUTED')
    cwd=workspace=profile['identity']['workspace'];stage_dir=workspace+'/.benchmark-cache/keel-wake-v1/'+stage;compiler='/synthetic/jdk25/bin/javac';classpath=workspace+'/.benchmark-cache/runner.jar';src=workspace+'/tools/KeelWakeBattle.java'
    proof=dict(schemaVersion=1,status='valid',command=[compiler,'-proc:none','-cp',classpath,'-d',stage_dir+'/launcher-compilation/classes',src],cwd=workspace,source=dict(path=src,sha256=freeze['assets']['tools/KeelWakeBattle.java']),classpath=dict(path=classpath,sha256=profile['runner']),compiler=dict(path=compiler,sha256='1'*64),compilerVersionCommand=[compiler,'-version'],compilerVersion='javac 25.0.5\n',stdout=dict(path=stage_dir+'/launcher-compilation/stdout.bin',bytes=0,sha256=audit.digest(cr/'stdout.bin')),stderr=dict(path=stage_dir+'/launcher-compilation/stderr.bin',bytes=0,sha256=audit.digest(cr/'stderr.bin')),exitCode=0,classes={'KeelWakeBattle.class':dict(sha256=audit.digest(cr/'classes/KeelWakeBattle.class'),mode=(cr/'classes/KeelWakeBattle.class').stat().st_mode&0o7777)},startedAtUTC=at(4,10),completedAtUTC=at(4,11),elapsedMonotonicSeconds=1)
    dump(cr/'proof.json',proof);profile['launcherCompilation']=dict(proof=proof,proofSHA256=audit.digest(cr/'proof.json'))
    dump(pro/'profile.json',profile)
    participants={arm:dict(name=name,version='0.1.0',jarSHA256=freeze['assets']['experiments/'+name+'/'+name+'.jar']) for arm,name in [('candidate','KeelWake'),('baseline','Keel')]};dump(pro/'participant-manifest.json',participants);dump(pro/'opponent-manifest.json',{'scope':'PURE FIXTURE immutable opponent binding, no private input'})
    commands=[];rows=[]
    for index,g in enumerate(population):
        folder=e/'games'/g['id'];folder.mkdir(parents=True,exist_ok=True)
        prepared=dict(game=g,command=['SYNTHETIC-NOT-EXECUTED',str(g['port']),str(g['rounds'])],environment={},cwd='/synthetic/'+g['id'],modelFilesBefore=[],persistentModelBefore='N/A in-memory',inputFiles=[])
        dump(folder/'prepared-game.json',prepared);commands.append(dict(id=g['id'],preparedGame=prepared,preparedGameSHA256=audit.digest(folder/'prepared-game.json')))
        name=g['participantName'];people=[dict(id=1,name=name,version='0.1.0'),dict(id=2,name=g['opponentName'],version=g['opponentVersion'])]
        share=(60 if g['bot']=='candidate' else 55) if g['opponent']=='diamond' else 95
        events=[dict(type='GameStartedEventForObserver',participants=people,gameSetup=dict(gameType='1v1',arenaWidth=800,arenaHeight=600,numberOfRounds=g['rounds'],turnTimeout=100000))]
        marker=[]
        for n in range(1,g['rounds']+1):
            bullet=lambda owner,power,bid:dict(ownerId=owner,power=power,bulletId=bid)
            events.append(dict(type='TickEventForObserver',roundNumber=n,turnNumber=1,botStates=people,events=[dict(type='BulletFiredEvent',bullet=bullet(1,.1,n*10)),dict(type='BulletFiredEvent',bullet=bullet(2,2,n*10+1))]))
            events.append(dict(type='TickEventForObserver',roundNumber=n,turnNumber=2,botStates=people,events=[dict(type='BulletHitBotEvent',bullet=bullet(2,2,n*10+1),victimId=1,damage=10)]))
            result=[dict(**person,totalScore=(share if person['id']==1 else 100-share)*n,firstPlaces=n if person['id']==1 else 0) for person in people]
            events.append(dict(type='RoundEndedEventForObserver',roundNumber=n,turnNumber=2,results=result))
            extra=' matchedDangerLabels=1 refreshEligiblePlans=1 refreshedWaves=1 densityChangedWaves=1' if g['bot']=='candidate' else ''
            marker.extend([f'{n} start',f'{n} stats maxMs=1'+extra,f'{n} end'])
        events.append(dict(type='GameEndedEventForObserver',numberOfRounds=g['rounds'],results=result))
        with gzip.open(folder/'synthetic.battle.gz','wt') as stream:
            for event in events:stream.write(json.dumps(event)+'\n')
        (folder/'markers.txt').write_text('\n'.join(marker)+'\n')
        start=at(11+index,10);end=at(11+index,20)
        dump(folder/'attempt.json',dict(prepared,headSHA=head,startUTC=start,freshProcess=True,freshWritableDirectory=True,importedModels=[]))
        dump(folder/'teardown.json',dict(ownedProcessGroup=1234,termSent=False,killSent=False,hardDeadlineRespected=True,hardLimitSeconds=1800,runStopReserveSeconds=16,endUTC=end,elapsedMonotonicSeconds=10))
        if g['opponent']=='ais':
            (folder/'ais.jsonl').write_text(''.join(json.dumps(dict(round=n,mode='train',seed=g['seed'],checkpointLoaded=False))+'\n' for n in range(1,g['rounds']+1)))
            raw=bytearray(24);raw[:4]=b'AISV';raw[4:8]=(1).to_bytes(4,'big');raw[20:24]=g['rounds'].to_bytes(4,'big');(folder/'ais-model.dat').write_bytes(raw)
        row=audit.reduce_game(folder/'synthetic.battle.gz',folder/'markers.txt',name,g['opponentName'],g['opponentVersion'],g['rounds'],g['bot']=='candidate');assert row['valid'],row;row['game']=g;rows.append(row)
    dump(pro/'commands.json',commands);dump(e/'commands.json',commands)
    raw=audit.stage_gates(stage,rows);assert raw['allPass'],raw
    canonical_games=[]
    for row in rows:
        g=row['game'];v=dict(g,status='valid',**{k:copy.deepcopy(v) for k,v in row.items() if k not in ('game','valid','timingPass')});v.update(observedPhysicalRoundIDs=list(range(1,g['rounds']+1)),completedPhysicalRoundIDs=list(range(1,g['rounds']+1)),observedPhysicalRounds=g['rounds'],completedPhysicalRounds=g['rounds'],partialRecordingError=None);canonical_games.append(v)
    decision=dict(stage=stage,headSHA=head,games=canonical_games,unplayedIDs=[],decision='progress; confirmation open' if stage=='screen' else 'confirmed',observedPhysicalRounds=sum(g['rounds'] for g in population),completedPhysicalRounds=sum(g['rounds'] for g in population),checks=audit.canonical_gate_checks(stage,rows,raw),diamondMean=raw['groups']['candidate/diamond']['mean']['decimal'],diamondDelta=raw['diamondDifference']['decimal'],retconMean=raw['groups']['candidate/retcon']['mean']['decimal'],retconDelta=raw['retconDifference']['decimal'],candidateDiamondDensityChangedWaves=raw['candidateDiamondDensityChangedWaves'],betweenGameShareSD={k:v['sampleSDPercentagePoints']/100 if v['sampleSDPercentagePoints'] is not None else None for k,v in raw['groups'].items() if k.rsplit('/',1)[1] in ('diamond','retcon')});dump(e/'decision.json',decision)
    reviewer='/synthetic-actually-assigned-separate-reviewer';pre=dict(host='synthetic-host',volume='/synthetic-volume',observedAtUTC=at(0),freeBytes=8*1024**3,toolEvidence={'scope':'PURE FIXTURE not actual disk'})
    record=dict(reviewerTask=reviewer,activeBeforeDispatch=True,toolEvidence={'scope':'PURE FIXTURE independent assignment'},storagePreflight=pre)
    dispatch=dict(schemaVersion=1,repository='stephanelee/robocode',issue=118,stage=stage,headSHA=head,reviewerTask=reviewer,reviewerAccount='stephanelee',implementationTasks=['/synthetic-implementation'],assignmentToolRecord=record,assignmentToolRecordSHA256=sha(audit.canonical(record)))
    def comment(cid,prefix,value,minute):return dict(id=cid,node_id='IC_'+str(cid),url=f'https://api.github.com/repos/stephanelee/robocode/issues/comments/{cid}',issue_url='https://api.github.com/repos/stephanelee/robocode/issues/118',html_url=f'https://github.com/stephanelee/robocode/issues/118#issuecomment-{cid}',body=prefix+'\n'+json.dumps(value),created_at=at(minute),updated_at=at(minute),user=dict(login='stephanelee',id=30698997,type='User'))
    dc=comment(1,'KEEL_WAKE_DISPATCH_ASSIGNMENT_V1',dispatch,0)
    artifact=audit.read_json(e/'artifact.json')
    approval=dict(schemaVersion=1,verdict='CLEAR',repository='stephanelee/robocode',issue=118,stage=stage,headSHA=head,runId=run,runAttempt=1,bootId=profile['identity']['bootId'],workspace=profile['identity']['workspace'],artifactId=artifact['id'],artifactDigestSHA256=artifact['digest'][7:],profileSHA256=audit.digest(pro/'profile.json'),freezeSHA256=freeze_sha,gamesSHA256=audit.GAMES_SHA256,participantManifestSHA256=audit.digest(pro/'participant-manifest.json'),opponentManifestSHA256=audit.digest(pro/'opponent-manifest.json'),commandsSHA256=audit.digest(pro/'commands.json'),reviewerTask=reviewer,dispatchAssignmentSHA256=sha(dc['body'].encode()))
    image_review=json.loads(audit.read_json(e/'review-report-comment.json')['body'].split('\n',1)[1])['agentToolProvenance']['imageIdentityReview']
    report=dict(approvalPayloadSHA256=sha(audit.canonical(approval)),reviewerTask=reviewer,reviewReport='PURE SYNTHETIC whole-reader fixture, not actual authority',agentToolProvenance=dict(reviewerTask=reviewer,reportGeneratedBeforeGames=True,imageIdentityReview=image_review))
    rc=comment(2,'KEEL_WAKE_RUNTIME_REVIEW_V1',report,8);ac=comment(3,'KEEL_WAKE_RUNTIME_APPROVAL_V1',approval,9)
    comments={'approval-comment.json':ac,'review-report-comment.json':rc,'dispatch-assignment-comment.json':dc}
    for name,c in comments.items():dump(e/name,c);dump(p/'comments'/f"{c['id']}.json",c)
    authority=dict(schemaVersion=1,approvalPayload=approval,approvalPayloadSHA256=sha(audit.canonical(approval)),comments={k:audit.comment_identity(v) for k,v in comments.items()},originalAcceptedCopiesSHA256={k:audit.digest(e/k) for k in comments});dump(e/'accepted-authority.json',authority);dump(e/'approval.json',approval);dump(e/'dispatch-assignment.json',dispatch)
    payload=dict(approval=approval,receiptRef='refs/tags/experiment-receipts/keel-wake-v1-'+stage,jobRemainingSecondsAtAdmission=10000,dispatchAssignment=dc,acceptedComment=ac,reviewReport=rc,acceptedAuthority=authority,acceptedAuthoritySHA256=sha(audit.canonical(authority)))
    tag=dict(sha='b'*40,object=dict(sha=head,type='commit'),message=json.dumps(payload));ref=dict(object=dict(sha=tag['sha'],type='tag'));dump(e/'receipt.json',dict(payload=payload,annotatedTag=tag,createdRef=ref));dump(p/'receipt-ref.json',ref);dump(p/'receipt-tag.json',tag)
    # Replace tiny original live records with every exact full-population phase.
    shutil.rmtree(e/'approval-revalidations');shutil.rmtree(e/'image-identity-revalidations')
    for index,phase in enumerate(['before-receipt']+['before-'+g['id'] for g in population]):
        minute=10+index;rel=f'image-identity-revalidations/revalidation-{minute}/snapshot.json';snapshot((e/rel).parent,f'revalidation-{minute}',minute,'after-tools')
        v=dict(schemaVersion=1,stage=stage,phase=phase,status='valid',acceptedAuthoritySHA256=payload['acceptedAuthoritySHA256'],currentComments=comments,currentArtifact=artifact,currentPullRequest=dict(head=dict(sha=head),body='KEEL-WAKE-SOURCE-CLEARED '+head),scannedComments=list(comments.values()),currentReceiptRef=ref,currentReceiptTag=tag,imageIdentityRevalidation=dict(caseID=profile['imageIdentity']['caseID'],snapshotFile=rel,snapshotSHA256=audit.digest(e/rel),preparedSnapshotSHA256=profile['imageIdentity']['preparedSnapshotSHA256'],runtimeProfileSHA256=audit.digest(pro/'profile.json')))
        t.add_live_fields(e,phase,v,profile,at(minute,2),at(341));dump(e/'approval-revalidations'/(phase+'.json'),v)
    for name,minute,minimum in [('before-preparation',2,20*1024**3),('before-receipt',10,8*1024**3)]+[('before-'+g['id'],11+i,8*1024**3) for i,g in enumerate(population)]:dump(e/'storage-admissions'/(name+'.json'),dict(schemaVersion=1,phase=name,observedAtUTC=at(minute),volume='/synthetic-worker',freeBytes=minimum,minimumFreeBytes=minimum,stageOutputBytes=100,stageOutputLimitBytes=4*1024**3))
    dump(e/'storage-monitor.json',dict(schemaVersion=1,firstObservedAtUTC=at(2),lastObservedAtUTC=at(25),observations=20,maximumStageOutputBytes=100,limitBytes=4*1024**3,limitExceeded=False))
    run_api=audit.read_json(p/'run.json');run_api.update(created_at=at(1),updated_at=at(55));dump(p/'run.json',run_api);dump(p/'workflow-runs.json',dict(workflow_runs=[run_api]))
    jobs=audit.read_json(p/'jobs.json');j=jobs['jobs'][0];j.update(started_at=at(1),completed_at=at(50));j['steps'].append(dict(number=11,name='Run fixed population',status='completed',conclusion='success',started_at=at(10),completed_at=at(49)));dump(p/'jobs.json',jobs)
    log_meta=audit.read_json(p/'job-log-api.json');log_meta['retrievedAtUTC']=at(51);dump(p/'job-log-api.json',log_meta)
    review_record=dict(reviewerTask=reviewer,toolEvidence={'scope':'PURE synthetic prior read/assignment evidence, not own final output'})
    identity=dict(schemaVersion=1,repository='stephanelee/robocode',issue=118,stage=stage,runId=run,runAttempt=1,headSHA=head,reviewerTask=reviewer,implementationTasks=dispatch['implementationTasks'],assignmentCommentId=1,assignmentBodySHA256=sha(dc['body'].encode()),assignmentToolRecord=record,assignmentToolRecordSHA256=sha(audit.canonical(record)),reviewToolRecord=review_record,reviewToolRecordSHA256=sha(audit.canonical(review_record)));dump(p/'reviewer-identity.json',identity)
    facts=audit.audit_runner_image(e,head,run,p,source,require_phase_proof=False)
    assert facts['pass'],facts
    dump(p/'phase-order-proof.json',facts['phaseOrderProof'])
    # Original ZIP generated once; external API and separate companion have no self-cycle.
    z=root/'original.zip'
    with zipfile.ZipFile(z,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for f in sorted(e.rglob('*')):
            if f.is_file():archive.write(f,str(f.relative_to(e)))
    final=dict(id=100,size_in_bytes=z.stat().st_size,digest='sha256:'+audit.digest(z),workflow_run=dict(id=run,head_sha=head),updated_at=at(52));dump(p/'evidence-artifact.json',final)
    c=root/'custody';dump(c/'evidence-artifact.json',final);dump(c/'export-validation.json',dict(schemaVersion=1,stage=stage,status='valid',observedAtUTC=at(53),originalZIPMaximumBytes=2*1024**3,rawDecisionSHA256=audit.digest(e/'decision.json'),artifact=final))
    cz=root/'companion.zip'
    with zipfile.ZipFile(cz,'w') as archive:
        for f in c.iterdir():archive.write(f,f.name)
    companion=dict(id=101,name=f'keel-wake-v1-{stage}-custody-validation-{run}',size_in_bytes=cz.stat().st_size,digest='sha256:'+audit.digest(cz),workflow_run=dict(id=run,head_sha=head));dump(p/'custody-artifact.json',companion)
    return dict(evidence=e,source=source,provider=p,head=head,run=run,zip=z,custodyZip=cz,identity=p/'reviewer-identity.json',reviewer=reviewer,stage=stage)


def repack(x):
    """Rebind only synthetic transport custody, so semantic mutations reach reader."""
    root=x['evidence'].parent;e=x['evidence'];p=x['provider'];z=x['zip'];cz=x['custodyZip'];c=root/'custody'
    with zipfile.ZipFile(z,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for f in sorted(e.rglob('*')):
            if f.is_file():archive.write(f,str(f.relative_to(e)))
    final=audit.read_json(p/'evidence-artifact.json');final.update(size_in_bytes=z.stat().st_size,digest='sha256:'+audit.digest(z));dump(p/'evidence-artifact.json',final);dump(c/'evidence-artifact.json',final)
    export=audit.read_json(c/'export-validation.json');export.update(artifact=final,rawDecisionSHA256=audit.digest(e/'decision.json'));dump(c/'export-validation.json',export)
    with zipfile.ZipFile(cz,'w') as archive:
        for f in c.iterdir():archive.write(f,f.name)
    companion=audit.read_json(p/'custody-artifact.json');companion.update(size_in_bytes=cz.stat().st_size,digest='sha256:'+audit.digest(cz));dump(p/'custody-artifact.json',companion)


def attach_upstream_screen(screen,x):
    """Retain actual synthetic original ZIP ranges/all metadata and review sidecar."""
    source=x['source'];base=source/'benchmarks/keel-wake-v1/fixture-screen-custody';base.mkdir(parents=True,exist_ok=True);retained={};root=x['evidence'].parent
    def retain(rel,data):
        target=source/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data);retained[rel]=sha(data);return rel
    parts=[];blob=lambda b:hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest();raw=screen['zip'].read_bytes()
    for i,offset in enumerate(range(0,len(raw),2*1024**2)):
        b=raw[offset:offset+2*1024**2];rel=retain(f'benchmarks/keel-wake-v1/fixture-screen-custody/part-{i:04d}.bin',b);parts.append(dict(path=rel,offset=offset,bytes=len(b),sha256=sha(b),gitBlobSHA1=blob(b)))
    members={}
    with zipfile.ZipFile(screen['zip']) as archive:
        for info in archive.infolist():
            b=archive.read(info);members[info.filename]=dict(bytes=info.file_size,sha256=sha(b),gitBlobSHA1=blob(b),crc32=info.CRC,compression=info.compress_type,compressedBytes=info.compress_size,flags=info.flag_bits,externalAttr=info.external_attr,timestamp=list(info.date_time),directory=info.is_dir())
    freeze=audit.read_json(screen['evidence']/'provenance/freeze.json');freeze_sha=audit.digest(screen['evidence']/'provenance/freeze.json');srcbindings={rel:dict(member='provenance/source/'+rel,commit=screen['head'],path=rel,gitBlobSHA1=members['provenance/source/'+rel]['gitBlobSHA1'],sha256=wanted) for rel,wanted in freeze['assets'].items()}
    population=audit.read_json(screen['source']/'benchmarks/keel-wake-v1/games.json')['screen']
    manifest=dict(schemaVersion=1,headSHA=screen['head'],sourceFreezeSHA256=freeze_sha,originalZIP=dict(artifactId=100,bytes=len(raw),sha256=sha(raw),parts=parts),members=members,decisionMember='decision.json',freezeMember='provenance/freeze.json',gameRecordingMembers={g['id']:'games/'+g['id']+'/synthetic.battle.gz' for g in population},sourceBindings=srcbindings)
    mrel='benchmarks/keel-wake-v1/fixture-screen-custody/custody-manifest.json';retain(mrel,(json.dumps(manifest,indent=2)+'\n').encode());drel=retain('benchmarks/keel-wake-v1/fixture-screen-custody/decision.json',(screen['evidence']/'decision.json').read_bytes())
    # Completed independent fixture-screen proof; still entirely synthetic.
    transport=dict(originalZIPBytes=screen['zip'].stat().st_size,originalZIPSHA256=audit.digest(screen['zip']),custodyZIPBytes=screen['custodyZip'].stat().st_size,custodyZIPSHA256=audit.digest(screen['custodyZip']),custodyRoot=screen['evidence'].parent/'custody')
    review=audit.audit_stage(screen['evidence'],screen['source'],'screen',screen['head'],screen['run'],screen['provider'],transport,screen['reviewer'],screen['identity']);assert review['decision']=='PASS',review
    recon=copy.deepcopy(review['runnerImage']['finalReconciliation'])
    for key,filename in [('runAPI','run.json'),('jobsAPI','jobs.json'),('jobLog','job.log'),('reviewerIdentityEvidence','reviewer-identity.json'),('phaseOrderEvidence','phase-order-proof.json')]:
        data=(screen['provider']/filename).read_bytes();rel=retain('benchmarks/keel-wake-v1/fixture-screen-custody/provider/'+filename,data);recon[key]=dict(path=rel,bytes=len(data),sha256=sha(data))
    cp=x['provider']/'screen';shutil.copytree(screen['provider'],cp)
    outcome=dict(verdict='SCREEN-PASS',stage='screen',sourceFreezeSHA256=freeze_sha,runId=screen['run'],headSHA=screen['head'],decisionSHA256=audit.digest(screen['evidence']/'decision.json'),reviewerTask=screen['reviewer'],agentToolProvenance=dict(reviewerTask=screen['reviewer'],scope='PURE FIXTURE nonphysical review',imageIdentityReconciliation=recon))
    comment=dict(id=500,user=dict(login='stephanelee',id=30698997),body='KEEL_WAKE_OUTCOME_REVIEW_V1\n'+json.dumps(outcome),created_at='2026-11-05T01:00:00Z',updated_at='2026-11-05T01:00:00Z');dump(cp/'outcome-comment.json',comment)
    gate=dict(schemaVersion=1,sourceFreezeSHA256=freeze_sha,screenRunId=screen['run'],receiptTagSHA='b'*40,decisionFile=drel,decisionSHA256=audit.digest(screen['evidence']/'decision.json'),retainedEvidenceFiles=retained,custodyManifestFile=mrel,independentOutcomeReviewCommentId=500);dump(source/'benchmarks/keel-wake-v1/screen-evidence.json',gate)
    # These archived inputs are read only by the offline upstream audit; a real
    # confirmation's source/command freeze must independently bind its gate.
    return gate


def bind_fixture_upstream_inputs(x,gate):
    e=x['evidence'];source=x['source'];pro=e/'provenance';p=x['provider']
    required={**gate['retainedEvidenceFiles'],'benchmarks/keel-wake-v1/screen-evidence.json':audit.digest(source/'benchmarks/keel-wake-v1/screen-evidence.json'),gate['custodyManifestFile']:audit.digest(source/gate['custodyManifestFile'])}
    manifest=audit.read_json(source/gate['custodyManifestFile']);required.update({part['path']:part['sha256'] for part in manifest['originalZIP']['parts']})
    upstream={rel:dict(sha256=wanted,mode=(source/rel).stat().st_mode&0o7777) for rel,wanted in required.items()}
    export={}
    for rel,item in upstream.items():
        member='provenance/upstream-inputs/'+rel;target=e/member;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((source/rel).read_bytes());export[rel]=dict(member=member,**item)
    dump(pro/'upstream-input-manifest.json',export)
    commands=audit.read_json(pro/'commands.json')
    for binding in commands:
        prepared=binding['preparedGame'];prepared.update(inputFiles=copy.deepcopy(upstream),upstreamInputBindings=copy.deepcopy(upstream));folder=e/'games'/binding['id'];dump(folder/'prepared-game.json',prepared);binding['preparedGameSHA256']=audit.digest(folder/'prepared-game.json')
        attempt=audit.read_json(folder/'attempt.json');attempt.update(inputFiles=copy.deepcopy(upstream),upstreamInputBindings=copy.deepcopy(upstream));dump(folder/'attempt.json',attempt)
    dump(pro/'commands.json',commands);dump(e/'commands.json',commands)
    profile=audit.read_json(pro/'profile.json')
    approval=audit.read_json(e/'approval.json');approval['commandsSHA256']=audit.digest(pro/'commands.json');dump(e/'approval.json',approval)
    comments={name:audit.read_json(e/name) for name in ['approval-comment.json','review-report-comment.json','dispatch-assignment-comment.json']}
    comments['approval-comment.json']['body']='KEEL_WAKE_RUNTIME_APPROVAL_V1\n'+json.dumps(approval)
    report=json.loads(comments['review-report-comment.json']['body'].split('\n',1)[1]);report['approvalPayloadSHA256']=sha(audit.canonical(approval));comments['review-report-comment.json']['body']='KEEL_WAKE_RUNTIME_REVIEW_V1\n'+json.dumps(report)
    for name,c in comments.items():dump(e/name,c);dump(p/'comments'/f"{c['id']}.json",c)
    authority=dict(schemaVersion=1,approvalPayload=approval,approvalPayloadSHA256=sha(audit.canonical(approval)),comments={name:audit.comment_identity(c) for name,c in comments.items()},originalAcceptedCopiesSHA256={name:audit.digest(e/name) for name in comments});dump(e/'accepted-authority.json',authority)
    receipt=audit.read_json(e/'receipt.json');receipt['payload'].update(approval=approval,acceptedAuthority=authority,acceptedAuthoritySHA256=sha(audit.canonical(authority)),acceptedComment=comments['approval-comment.json'],reviewReport=comments['review-report-comment.json']);receipt['annotatedTag']['message']=json.dumps(receipt['payload']);dump(e/'receipt.json',receipt);dump(p/'receipt-tag.json',receipt['annotatedTag'])
    for path in (e/'approval-revalidations').glob('*.json'):
        v=audit.read_json(path);v.update(acceptedAuthoritySHA256=receipt['payload']['acceptedAuthoritySHA256'],currentComments=comments,scannedComments=list(comments.values()))
        if v['receiptState']=='present':v['currentReceiptTag']=receipt['annotatedTag']
        t.add_live_fields(e,v['phase'],v,profile,v['checkedAtUTC'],v['jobDeadlineUTC']);dump(path,v)
    repack(x)
