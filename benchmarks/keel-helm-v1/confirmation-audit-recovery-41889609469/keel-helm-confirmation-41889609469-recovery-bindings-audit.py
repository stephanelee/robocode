from pathlib import Path
import json,hashlib,datetime,math,zipfile
B=Path('/private/tmp/keel-helm-confirmation-41889609469');S=B/'robocode/robocode/build/keel-helm/confirmation';R=S.parents[2]
Z=zipfile.ZipFile(str(B)+'.zip')
load=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
canon=lambda d:json.dumps(d,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
time=lambda s:datetime.datetime.fromisoformat(s.replace('Z','+00:00')).timestamp()
checks=[]
def ck(ok,label):
 if not ok:raise ValueError(label)
 checks.append(label)
run=load(str(B)+'-run.json');jobs=load(str(B)+'-jobs-final.json')['jobs'];tag=load(str(B)+'-receipt-tag.json');ref=load(str(B)+'-receipt-ref.json');payload=json.loads(tag['message']);a=load(S/'approval.json');profile=load(S/'profile.json');commands=load(S/'commands.json');plan=load('/private/tmp/robocode-keel-improvement/benchmarks/keel-helm-v1/games.json')['confirmation']
ck(run['status']=='completed' and run['conclusion']=='success' and run['run_attempt']==1 and run['head_sha']=='f2a25d3309062a362a13902329de405bf63094a7','Provider completed/success attempt1 exact head')
ck(len(jobs)==1 and jobs[0]['name']=='registered-stage' and jobs[0]['conclusion']=='success','Exactly one completed registered-stage job')
steps={s['name']:s for s in jobs[0]['steps']}
for name in ['Build source, semantic tests and all committed JAR parity on Java17 first','Prepare exact sources/runtime/commands on this worker (zero games)','Finalize immutable zero-game provenance before independent review','Wait at most30minutes for actively assigned independent reviewer','Rehash same worker, atomically consume sole receipt and run fixed population','Preserve every failure, partial budget, unplayed ID and raw result']:
 ck(steps[name]['conclusion']=='success','Completed step '+name)
ck((time(jobs[0]['completed_at'])-time(jobs[0]['started_at']))<340*60,'Overall job below340minutes')
ck(ref['object']['type']=='tag' and ref['object']['sha']==tag['sha']=='42b194e58f8b9b4dc3a477827658157a8582ad72','Fixed remote annotated receipt object')
ck(tag['object']['sha']==run['head_sha'] and payload['stage']=='confirmation' and payload['approval']==a,'Remote receipt exact head/stage/approval')
ck(payload==load(S/'receipt.json')['payload'],'Archived receipt payload byte-semantic identity with remote tag')
artifact=load(str(B)+'-provenance-artifact-api.json')
ck(artifact['id']==a['artifactId']==13352733399 and artifact['digest']=='sha256:'+a['artifactDigestSHA256'],'Unchanged pre-game finalized provenance API identity/digest')
ck(artifact['workflow_run']['id']==run['id'] and artifact['workflow_run']['head_sha']==run['head_sha'],'Pre-game artifact belongs to exact screen run/head')
files={'approval-comment.json':'approval-current.json','review-report-comment.json':'review-current.json','dispatch-assignment-comment.json':'dispatch-current.json'}
original={}
for name,suffix in files.items():
 old=load(S/name);now=load(str(B)+'-'+suffix);original[name]=old
 for k in ['id','node_id','issue_url','html_url','created_at','updated_at','body']:
  ck(old[k]==now[k],name+' current exact '+k)
 ck(old['user']['login']==now['user']['login']=='stephanelee' and old['user']['id']==now['user']['id']==30698997,name+' designated author')
dispatch=json.loads(original['dispatch-assignment-comment.json']['body'].split('\n',1)[1]);review=json.loads(original['review-report-comment.json']['body'].split('\n',1)[1]);approval=json.loads(original['approval-comment.json']['body'].split('\n',1)[1])
ck(approval==a and len(a)==20 and a['schemaVersion']==1 and a['verdict']=='CLEAR' and a['issue']==107,'Strict original20field approval binding')
ck(dispatch['reviewerTask']==a['reviewerTask']=='/root/stop24_acquisition_audit' and a['reviewerTask'] not in dispatch['implementationTasks'],'Assigned non-implementation runtime reviewer')
ck(dispatch['assignmentToolRecord']['activeBeforeDispatch'] is True and dispatch['assignmentToolRecord']['reviewerTask']==a['reviewerTask'],'Actual recorded active reviewer assignment')
ck(hashlib.sha256(canon(dispatch['assignmentToolRecord'])).hexdigest()==dispatch['assignmentToolRecordSHA256'],'Assignment tool provenance digest')
ck(hashlib.sha256(original['dispatch-assignment-comment.json']['body'].encode()).hexdigest()==a['dispatchAssignmentSHA256'],'Exact dispatch API body bound by approval')
ck(review['approvalPayloadSHA256']==hashlib.sha256(canon(a)).hexdigest() and review['reviewerTask']==a['reviewerTask'] and review['reviewReport'].strip() and review['agentToolProvenance']['reviewerTask']==a['reviewerTask'] and review['agentToolProvenance']['reportGeneratedBeforeGames'] is True,'Separate reasoned runtime report binds approval and real reviewer provenance')
ck(time(original['dispatch-assignment-comment.json']['created_at'])<time(run['created_at']),'Assignment predates actual dispatch')
ck(time(artifact['updated_at'])<time(original['review-report-comment.json']['created_at'])<=time(original['approval-comment.json']['created_at'])<=time(artifact['updated_at'])+1800,'Independent runtime review after finalization and within30minute deadline')
ck(original['review-report-comment.json']['id']<original['approval-comment.json']['id'],'Reasoned report precedes approval by provider ID')
revalidations=sorted((S/'approval-revalidations').glob('*.json'));ck(len(revalidations)==19,'Exactly before-receipt plus18before-game authority revalidations')
for p in revalidations:
 row=load(p);ck(row['status']=='valid' and row['stage']=='confirmation',p.name+' valid authority gate')
 for name,old in original.items():
  current=row['currentComments'][name]
  ck(all(current[k]==old[k] for k in ['id','body','created_at','updated_at']) and current['user']['id']==old['user']['id'],p.name+' exact authority '+name)
 ck(row['currentPullRequest']['head']['sha']==run['head_sha'] and 'KEEL-HELM-SOURCE-CLEARED '+run['head_sha'] in row['currentPullRequest']['body'],p.name+' current source clearance')
 ck(row['currentArtifact']['id']==artifact['id'] and row['currentArtifact']['digest']==artifact['digest'],p.name+' immutable provenance artifact')
 if p.name!='before-receipt.json':
  ck(row['currentReceiptRef']['object']['sha']==tag['sha'] and row['currentReceiptTag']['sha']==tag['sha'],p.name+' permanent receipt remains exact')
starts=[];all_inputs=0;admission_times=[]
for item,g in zip(commands,plan):
 d=S/'games'/g['id'];prepared=load(d/'prepared-game.json');attempt=load(d/'attempt.json');teardown=load(d/'teardown.json');before=load(S/'approval-revalidations'/('before-'+g['id']+'.json'))
 ck(prepared==item['preparedGame'] and sha(d/'prepared-game.json')==item['preparedGameSHA256'] and prepared['game']==g,g['id']+' exact fixed prepared population')
 ck(attempt['game']==g and attempt['headSHA']==run['head_sha'],g['id']+' exact physical attempt population/head')
 ck(attempt['command']==prepared['command'] and attempt['cwd']==prepared['cwd'] and attempt['environment']==prepared['environment'],g['id']+' actual command/cwd/environment unchanged')
 ck(attempt['freshProcess'] is True and attempt['freshWritableDirectory'] is True and attempt['importedModels']==[] and attempt['persistentKeelModelHashes'] is None,g['id']+' actual cold-state/no imported Keel models')
 ck(not teardown['termSent'] and not teardown['killSent'],g['id']+' normal owned process teardown')
 start=time(attempt['startUTC']);starts.append(start)
 ck(time(before['checkedAtUTC'])<=start and start>time(original['approval-comment.json']['created_at']) and start>time(tag['tagger']['date']),g['id']+' authority/approval/receipt before physical child')
 remain=time(jobs[0]['started_at'])+340*60-start;ck(remain>=2100,g['id']+' actual start35minute admission margin');admission_times.append(remain)
 for rel,info in prepared['inputFiles'].items():
  member=R.relative_to(B).as_posix()+'/'+rel;raw=Z.read(member);ck(hashlib.sha256(raw).hexdigest()==info['sha256'],g['id']+' originalZIP staged input '+rel);all_inputs+=1
 # Runner log and original recording persist even when private metrics aren't used.
 ck((d/'runner.log').is_file() and len(list(d.glob('*.battle.gz')))==1,g['id']+' full original runner log/recording')
ck(starts==sorted(starts),'Registered cold games attempted in fixed order')
ck(not list(S.rglob('failure.json')),'No hidden failure records in completed screen evidence')
workflow=load(str(B)+'-workflow-runs.json');ck(workflow['total_count']==2 and {v['id'] for v in workflow['workflow_runs']}=={41874753112,41889609469} and all(v['run_attempt']==1 and v['status']=='completed' for v in workflow['workflow_runs']),'Provider history has exactly one screen and one confirmation, both attempt1')
out={'scope':'Independent raw-provider/API/authority/receipt/command/budget checks; no game run or source modification','pass':True,'checks':checks,'checkCount':len(checks),'retainedStagedInputsChecked':all_inputs,'minimumRemainingSecondsAtActualChildStart':min(admission_times),'actualRun':run['id'],'receiptTagSHA':tag['sha'],'sourceFreezeSHA256':a['freezeSHA256'],'profileSHA256':a['profileSHA256'],'runtimeImage':profile['identity']['image'],'bootId':profile['identity']['bootId'],'preGameArtifactId':artifact['id'],'preGameArtifactSHA256':a['artifactDigestSHA256'],'approvalCommentId':original['approval-comment.json']['id'],'runtimeReviewCommentId':original['review-report-comment.json']['id'],'dispatchCommentId':original['dispatch-assignment-comment.json']['id'],'remainingManualLimit':'Unexported live native/JDK/Python filesystem bytes were rehashed by reviewed same-worker gate; this offline audit independently checks exported bytes/manifests/API bindings, not unexported live RAM/files.'}
Path(str(B)+'-recovery-bindings-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:out[k] for k in ['pass','checkCount','retainedStagedInputsChecked','minimumRemainingSecondsAtActualChildStart','receiptTagSHA']}))
