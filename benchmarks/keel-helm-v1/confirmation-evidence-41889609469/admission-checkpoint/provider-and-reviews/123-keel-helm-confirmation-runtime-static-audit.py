from pathlib import Path, PurePosixPath
import json,hashlib,zipfile,tarfile,datetime,re
R=Path('/private/tmp/keel-helm-confirmation-provenance-41889609469')
LOCAL=Path('/private/tmp/robocode-keel-improvement')
S=R/'source'
sha=lambda b:hashlib.sha256(b).hexdigest()
load=lambda p:json.loads(p.read_bytes())
canonical=lambda x:json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
a=load(Path('/private/tmp/keel-helm-confirmation-provenance-artifacts-41889609469.json'))['artifacts'][0]
run=load(Path('/private/tmp/keel-helm-confirmation-run41889609469.json'));jobs=load(Path('/private/tmp/keel-helm-confirmation-jobs41889609469-beforeapproval.json'))['jobs']
f=load(R/'freeze.json');p=load(R/'profile.json');part=load(R/'participant-manifest.json');opp=load(R/'opponent-manifest.json');commands=load(R/'commands.json');loc=load(R/'locations.json');manifest=load(R/'manifest.json');proof=load(R/'python-runtime.json');build=load(R/'build.json')
HEAD='f2a25d3309062a362a13902329de405bf63094a7';F='ef876a82069d5c73baf7171aa95cf4058e3bc4a196f46d51bdf5ff1cf0b81a9b';WORK='/home/runner/work/robocode/robocode'
assert sha((R/'freeze.json').read_bytes())==F and (R/'freeze.json').read_bytes()==(LOCAL/'benchmarks/keel-helm-v1/freeze.json').read_bytes()
actual={x.relative_to(R).as_posix() for x in R.rglob('*') if x.is_file()};assert actual==set(manifest)|{'manifest.json'}
for name,row in manifest.items():
 assert sha((R/name).read_bytes())==row['sha256'],name
 assert not row.get('symlink'),name
assert len(f['assets'])==90 and p['sourceAssets']==f['assets']
for name,h in f['assets'].items():
 assert sha((S/name).read_bytes())==h and (S/name).read_bytes()==(LOCAL/name).read_bytes(),name
 assert p['sourceModes'][name] in (420,493),name
ident=p['identity'];assert ident['headSHA']==HEAD and ident['repository']=='stephanelee/robocode' and ident['runId']==41889609469 and ident['runAttempt']==1 and ident['pullNumber']==108 and ident['workspace']==WORK
assert ident['image'] in f['allowedImages'] and ident['architecture']=='x86_64' and 'VERSION_ID="26.04"' in ident['os'] and ident['pythonVersion'].startswith('3.13.17 ')
assert re.fullmatch('[0-9a-f-]{36}',ident['bootId'])
assert ident['bootId']!='7bd7cace-1b34-4348-ad11-a8cfb7c1f6fb'
assert run['head_sha']==HEAD and run['run_attempt']==1 and run['event']=='pull_request' and run['head_branch']=='codex/keel-improvement-baseline' and run['status']=='in_progress'
steps=jobs[0]['steps'];assert jobs[0]['name']=='registered-stage' and jobs[0]['status']=='in_progress'
assert next(s for s in steps if s['name'].startswith('Wait at most30minutes'))['status']=='in_progress'
assert next(s for s in steps if s['name'].startswith('Rehash same worker'))['status']=='pending'
for prefix in ['Build source, semantic tests','Prepare exact sources/runtime/commands','Finalize immutable zero-game']:
 assert next(s for s in steps if s['name'].startswith(prefix))['conclusion']=='success'
assert sha((R/'build.log').read_bytes())==p['buildLogSHA256']==build['buildLogSHA256']
assert build['java17']==p['java17'] and build['casesOrGames']==0 and build['commandCWD']==WORK
assert '17.0.21+9' in p['java17']['version'] and '25.0.5+8-LTS' in p['java25']['version']
assert p['java17']['home']=='/usr/lib/jvm/temurin-17-jdk-amd64' and p['java25']['home']==loc['java25']=='/usr/lib/jvm/temurin-25-jdk-amd64'
for java in ('java17','java25'):
 for n in ('bin/java','bin/javac','lib/modules','lib/server/libjvm.so','release'):assert re.fullmatch('[0-9a-f]{64}',p[java]['files'][n]['sha256'])
assert p['runner']=='02f6d1d8e9346a4aeae1b91e7abb278c98ac1f1d30c605f867758566f5f327cf'
assert proof==p['pythonRuntime'] and sha((R/'python-runtime.json').read_bytes())==p['pythonRuntimeSHA256']
assert proof['schemaVersion']==1 and 'all existing pyc' in proof['policy'] and proof['files'][proof['executableKey']]['sha256']==ident['pythonSHA256']
for key,v in proof['files'].items():
 assert key.startswith(('base/','system/')) and '..' not in PurePosixPath(key).parts
 if v['sha256'] is not None:assert re.fullmatch('[0-9a-f]{64}',v['sha256'])
assert proof['builtinFrozenBinding']['executableSHA256']==ident['pythonSHA256']
assert proof['builtinFrozenBinding']['libpythonSHA256']==proof['files'][proof['libpythonKey']]['sha256']
assert proof['builtinFrozenBinding']['nativeRuntime']=={n:proof['files'][n] for n in proof['nativeDependencyKeys']}
assert proof['loaderConfiguration']=={n:proof['files'][n] for n in proof['loaderConfiguration']}
assert all(v['kind'] in ('directory','absent') for v in proof['startupPaths'])
strong=load(S/'tools/opponents/strong-melee.lock.json');ret=load(S/'tools/opponents/retcon.lock.json')
assert p['native']['source']=={v['path']:v['sha256'] for v in strong['files']}
assert {n:v['sha256'] for n,v in p['native']['wheels'].items()}=={v['filename']:v['sha256'] for v in strong['runtime']['wheels']}
assert opp['diamond']['commit']==strong['commit'] and opp['diamond']['sourceSHA256']==strong['sourceSHA256'] and opp['diamond']['lockSHA256']==sha((S/'tools/opponents/strong-melee.lock.json').read_bytes())
normalize=lambda n:n.lower().replace('_','-').replace('.','-')
assert {normalize(n):v for n,v in opp['diamond']['runtime']['packages']}=={normalize(w['name']):w['version'] for w in strong['runtime']['wheels']}
assert opp['diamond']['runtime']['version']==ident['pythonVersion'] and opp['diamond']['pipCheck']=='No broken requirements found.'
assert p['native']['venv']['bin/python']['sha256']==ident['pythonSHA256']
ret_raw={}
with tarfile.open('/private/tmp/keel-helm-runtime-retcon28981-source.tar.gz','r:gz') as t:
 members=t.getmembers();prefix=members[0].name.split('/')[0]+'/bots/java/Retcon/'
 for entry in ret['files']:
  b=t.extractfile(prefix+entry['path']).read();assert len(b)==entry['size'] and hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==entry['gitBlobSHA'];ret_raw[entry['path']]=b
h=hashlib.sha256()
for name,b in sorted(ret_raw.items()):h.update(name.encode()+b'\0'+b+b'\0')
assert h.hexdigest()==ret['sourceSHA256'] and {n:v['sha256'] for n,v in p['retcon']['source'].items()}=={n:sha(b) for n,b in ret_raw.items()}
assert next(iter(p['retcon']['api'].values()))['sha256']==ret['botApi']['sha256']
assert all(opp['retcon'][k]==ret[k] for k in opp['retcon'])
with zipfile.ZipFile('/private/tmp/keel-helm-runtime-samples1.4.0.zip') as z:
 sample={n:sha(z.read(n)) for n in z.namelist() if not n.endswith('/')}
assert {n:v['sha256'] for n,v in p['samples'].items()}==sample and opp['sampleFiles']==p['samples'] and opp['samplesArchiveSHA256']=='1707c1011a37c6b854fc79d68549ab204007bac6d33d803fa254152ac0765fe2'
base=load(S/'benchmarks/keel-improvement/baseline.json')
assert opp['productionAISurvivor']==base['productionAISurvivor']
for arm,name in [('baseline','Keel'),('candidate','KeelHelm')]:
 assert part[arm]['name']==name and part[arm]['version']=='0.1.0' and part[arm]['modelHashes']=='N/A: in-memory only'
 assert part[arm]['jarSHA256']==f['assets'][f'experiments/{name}/{name}.jar']
 assert {n:v['sha256'] for n,v in part[arm]['source'].items()}=={n.split('/src/')[1]:v for n,v in f['assets'].items() if n.startswith(f'experiments/{name}/src/')}
plan=load(S/'benchmarks/keel-helm-v1/games.json')['confirmation'];assert len(commands)==len(plan)==18 and sum(g['rounds'] for g in plan)==300
for binding,g in zip(commands,plan):
 q=binding['preparedGame'];gid=g['id'];name='Keel' if g['bot']=='baseline' else 'KeelHelm';rel=f'build/keel-helm/confirmation/work/{gid}/';data=f'{WORK}/build/keel-helm/confirmation/games/{gid}';work=WORK+'/'+rel[:-1]
 assert binding['id']==gid and q['game']==g and q['name']==name and q['version']=='0.1.0' and binding['preparedGameFile']==f'build/keel-helm/confirmation/games/{gid}/prepared-game.json'
 assert sha((json.dumps(q,indent=2,allow_nan=False)+'\n').encode())==binding['preparedGameSHA256']
 own={f'{name}/{name}.jar':f['assets'][f'experiments/{name}/{name}.jar']}
 own.update({f'{name}/{n}':f['assets'][f'experiments/{name}/bot/{n}'] for n in [name+'.json',name+'.sh',name+'.cmd']})
 expected=dict(own)
 enemy=g['opponent']
 if enemy=='diamond':
  for v in strong['files']:
   for pre,dest in [('bots/ports/diamond-port/','opponents/diamond-port/'),('bots/bot_core/','opponents/bot_core/')]:
    if v['path'].startswith(pre):expected[dest+v['path'][len(pre):]]=v['sha256']
  expected['opponents/LICENSE.shared-Apache-2.0']=next(v['sha256'] for v in strong['files'] if v['path']=='LICENSE');other='opponents/diamond-port';en='Diamond Port';ev='1.8.28'
 elif enemy=='retcon':expected.update({'Retcon/'+n:sha(b) for n,b in ret_raw.items()});expected['retcon-api/robocode-tankroyale-bot-api-1.3.1.jar']=ret['botApi']['sha256'];other='Retcon';en='Retcon';ev='0.2.0'
 elif enemy=='walls':expected.update({n:h for n,h in sample.items() if n.startswith(('Walls/','lib/'))});other='Walls';en='Walls';ev='1.0'
 elif enemy=='ais':expected.update({n:v for n,v in f['assets'].items() if n.startswith('AISurvivor/')});other='AISurvivor';en='AISurvivor';ev=load(S/'AISurvivor/AISurvivor.json')['version']
 assert {n[len(rel):]:v['sha256'] for n,v in q['inputFiles'].items() if n.startswith(rel)}==expected,gid
 assert all(n.startswith(rel) for n in q['inputFiles'])
 assert q['opponentName']==en and q['opponentVersion']==ev
 assert q['command']==[loc['java25']+'/bin/java','-cp',loc['runner'],WORK+'/tools/KeelHelmBattle.java',str(g['port']),str(g['rounds']),data,work+'/'+name,work+'/'+other]
 assert q['cwd']==work and q['ownWritableHome']==work+'/empty-home' and q['modelFilesBefore']==[] and q['candidateExposureOnly']==(g['bot']=='candidate')
 env={'PATH':loc['java25']+'/bin:/usr/bin:/bin','LANG':'C.UTF-8','JAVA_HOME':loc['java25'],'PYTHONDONTWRITEBYTECODE':'1','PYTHONNOUSERSITE':'1','LC_ALL':'C.UTF-8','HOME':work+'/empty-home',('KEEL_METRICS' if name=='Keel' else 'KEEL_HELM_METRICS'):data+'/markers.txt',('KEEL_SEED' if name=='Keel' else 'KEEL_HELM_SEED'):str(g['seed'])}
 if enemy=='diamond':env.update(ROBOCODE_PYTHON_BIN=loc['python'],ROBOCODE_ENV_FILE=work+'/absent-developer.env',ROBOCODE_GUN_ENV_FILE=work+'/absent-developer-guns.env')
 if enemy=='retcon':env['RUMBLE_JAVA_CACHE']=work+'/retcon-api'
 if enemy=='ais':env.update(AISURVIVOR_MODEL_PATH=data+'/ais-model.dat',AISURVIVOR_METRICS_PATH=data+'/ais.jsonl',AISURVIVOR_SEED=str(g['seed']),AISURVIVOR_MODE='train')
 assert q['environment']==env,gid
assignment_comment=load(R/'dispatch-assignment-comment.json');assignment=load(R/'dispatch-assignment.json');prefix='KEEL_HELM_DISPATCH_ASSIGNMENT_V1\n';assert assignment_comment['body'].startswith(prefix) and json.loads(assignment_comment['body'][len(prefix):])==assignment
original_assignment=load(Path('/private/tmp/keel-helm-confirmation-dispatch-comment6638977373.json'));assert all(assignment_comment[k]==original_assignment[k] for k in ['id','node_id','url','issue_url','html_url','created_at','updated_at','body']);assert all(assignment_comment['user'][k]==original_assignment['user'][k] for k in ['id','login','type'])
assert assignment['reviewerTask']=='/root/stop24_acquisition_audit' and assignment['reviewerTask'] not in assignment['implementationTasks'] and assignment['reviewerAccount']=='stephanelee'
assert sha(canonical(assignment['assignmentToolRecord']))==assignment['assignmentToolRecordSHA256'] and assignment['assignmentToolRecord']['activeBeforeDispatch'] is True
assert assignment_comment['created_at'] < run['created_at']
assert not any(n.endswith('.battle.gz') or n.endswith('/activation.json') for n in actual)
expected=dict(schemaVersion=1,verdict='CLEAR',repository='stephanelee/robocode',issue=107,stage='confirmation',headSHA=HEAD,runId=41889609469,runAttempt=1,bootId=ident['bootId'],workspace=WORK,artifactId=a['id'],artifactDigestSHA256=a['digest'].removeprefix('sha256:'),profileSHA256=sha((R/'profile.json').read_bytes()),freezeSHA256=F,gamesSHA256=sha((S/'benchmarks/keel-helm-v1/games.json').read_bytes()),participantManifestSHA256=sha((R/'participant-manifest.json').read_bytes()),opponentManifestSHA256=sha((R/'opponent-manifest.json').read_bytes()),commandsSHA256=sha((R/'commands.json').read_bytes()),reviewerTask=assignment['reviewerTask'],dispatchAssignmentSHA256=sha(assignment_comment['body'].encode()))
assert sorted(expected)==f['approvalKeys'] and len(expected)==20
Path('/private/tmp/keel-helm-confirmation-runtime-expected-approval.json').write_text(json.dumps(expected,sort_keys=True,indent=2)+'\n')
summary=dict(status='PASS pregame static artifact audit',head=HEAD,runId=41889609469,artifactId=a['id'],artifactSHA256=a['digest'],artifactBytes=a['size_in_bytes'],artifactFiles=len(actual),frozenSourceAssets=90,commandGames=18,roundBudget=300,identity=ident,java17Files=len(p['java17']['files']),java25Files=len(p['java25']['files']),basePythonProofEntries=len(proof['files']),pythonPyc=sum('.pyc' in k for k in proof['files']),nativeDependencyKeys=len(proof['nativeDependencyKeys']),nativeSourceFiles=len(p['native']['source']),venvFiles=len(p['native']['venv']),wheels=len(p['native']['wheels']),retconFiles=len(ret_raw),officialSampleFiles=len(sample),approvalPayloadSHA256=sha(canonical(expected)),expectedApproval=expected)
Path('/private/tmp/keel-helm-confirmation-runtime-static-audit.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({k:v for k,v in summary.items() if k not in ('identity','expectedApproval')},indent=2))
