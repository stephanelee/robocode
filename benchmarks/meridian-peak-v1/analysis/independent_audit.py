"""Independently audit preserved Peak v1 SCREEN evidence; imports no repository reducers and starts no games."""
import argparse,collections,gzip,hashlib,json,math,pathlib,struct,zipfile
P=argparse.ArgumentParser(description=__doc__)
P.add_argument('--repository',type=pathlib.Path,default=pathlib.Path(__file__).resolve().parents[3])
P.add_argument('--raw',type=pathlib.Path,help='Complete original screen acquisition directory, including stages/')
P.add_argument('--artifact-zip',type=pathlib.Path)
P.add_argument('--artifact-sha256',help='Expected original full artifact digest; required with --artifact-zip')
P.add_argument('--output',type=pathlib.Path)
P.add_argument('--expected-head',required=True,help='Independently cleared acquisition commit SHA')
P.add_argument('--expected-freeze',required=True,help='Independently cleared exact freeze SHA256')
A=P.parse_args();ROOT=A.repository.resolve();REG=ROOT/'benchmarks/meridian-peak-v1';RAW=(A.raw or ROOT/'build/meridian-peak/screen/acquisition').resolve()
HEAD=A.expected_head;FREEZE=A.expected_freeze
assert len(HEAD)==40 and len(FREEZE)==64 and all(c in '0123456789abcdef' for c in HEAD+FREEZE)
PACKAGE='52b7bfcf10f8226711798f27784be64583ea4c8fc676452d27f6e16b906bf191';CONTROL='5875565a13b5759940ccb080c0c7f237caec0bae525a68e68643a35890a1a8a9'
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def finite(x):
 if isinstance(x,dict):
  for v in x.values():finite(v)
 elif isinstance(x,list):
  for v in x:finite(v)
 elif isinstance(x,(int,float)) and not isinstance(x,bool):assert math.isfinite(x),x

def near(a,b,label='',tol=1e-6):assert math.isclose(a,b,abs_tol=tol,rel_tol=1e-10),(label,a,b)
def compare(a,b,label=''):
 if isinstance(a,dict):
  for k,v in a.items():compare(v,b[k],label+'.'+k)
 elif isinstance(a,list):
  assert len(a)==len(b),(label,len(a),len(b))
  for i,(x,y) in enumerate(zip(a,b)):compare(x,y,label+str(i))
 elif isinstance(a,float):near(a,b,label)
 else:assert a==b,(label,a,b)
def countrow(ticks,uid):
 r={k:0 for k in ['shots','hits','enemyShots','hitsTaken','wallContacts','botContacts','skips','callbackDeliverableOutgoingHits','observerOnlyOutgoingHits']}
 r.update(turns=len(ticks),shotEnergy=0.,damageDealt=0.,damageTaken=0.,callbackDeliverableOutgoingDamage=0.,observerOnlyOutgoingDamage=0.)
 for t in ticks:
  present=any(x['id']==uid for x in t['botStates'])
  for e in t['events']:
   k=e['type'];assert 'Skipped' not in k;b=e.get('bullet',{})
   if k=='BulletFiredEvent':
    if b['ownerId']==uid:r['shots']+=1;r['shotEnergy']+=b['power']
    else:r['enemyShots']+=1
   if k=='BulletHitBotEvent':
    if b['ownerId']==uid:
     r['hits']+=1;r['damageDealt']+=e['damage'];key='callbackDeliverableOutgoing' if present else 'observerOnlyOutgoing';r[key+'Hits']+=1;r[key+'Damage']+=e['damage']
    elif e['victimId']==uid:r['hitsTaken']+=1;r['damageTaken']+=e['damage']
   if k=='BotHitWallEvent' and e['victimId']==uid:r['wallContacts']+=1
   if k=='BotHitBotEvent' and e['botId']==uid:r['botContacts']+=1
 r['hitRate']=r['hits']/r['shots'] if r['shots'] else None;r['efficiency']=r['hitRate'];r['enemyHitRate']=r['hitsTaken']/r['enemyShots'] if r['enemyShots'] else None
 r['damagePer100Shots']=100*r['damageDealt']/r['shots'] if r['shots'] else None;r['damagePer100EnemyShots']=100*r['damageTaken']/r['enemyShots'] if r['enemyShots'] else None
 r['shotsPer1000ObserverTurns']=1000*r['shots']/r['turns'] if r['turns'] else None;r['enemyShotsPer1000ObserverTurns']=1000*r['enemyShots']/r['turns'] if r['turns'] else None
 return r

def validate_ais(data,game):
 raw=(data/'ais-model.dat').read_bytes();header=struct.unpack('>7i',raw[:28]);assert header[:6]==(0x41495356,1,14,32,6,game['rounds']) and header[6]>=0
 assert len(raw)==28+8+2*(14*32+32+32*6+6)*8+5*9*47*8
 values=struct.unpack('>'+str((len(raw)-28)//8)+'d',raw[28:]);assert all(math.isfinite(x) for x in values) and .15<=values[0]<=1 and min(values[-5*9*47:])>=0
 rows=[json.loads(l) for l in (data/'ais.jsonl').read_text().splitlines()];assert len(rows)==game['rounds']
 for n,r in enumerate(rows,1):
  finite(r);assert r['round']==r['totalRounds']==n and r['schemaVersion']==1 and r['mode']=='train' and r['seed']==game['seed'] and r['checkpointLoaded'] is False and r['action'] is None
  assert r['skippedTurns']==0 and all(type(r[k]) is int and r[k]>=0 for k in ['turns','totalRounds','shotsFired','bulletHits','wallCollisions','botCollisions','skippedTurns'])
  assert r['bulletHits']<=r['shotsFired'];near(r['hitRate'],r['bulletHits']/r['shotsFired'] if r['shotsFired'] else 0)
  assert type(r['won']) is bool and type(r['died']) is bool and not(r['won'] and r['died'])
 return dict(sha256=sha(data/'ais-model.dat'),rounds=game['rounds'])

def launch_context(launch,byturn,uid,shots):
 t=launch['turn'];assert type(t) is int and t>=1 and .1<=launch['power']<=3 and launch['sign'] in [-1,1]
 assert set(launch)=={'turn','x','y','power','bearing','sign','range','context'};finite(launch)
 actual=[b for turn,b in shots if turn==t];assert len(actual)==1,(t,'launch without one confirmed shot')
 for k in ['x','y','power']:near(launch[k],actual[0][k],k)
 previous=byturn.get(t-1);assert previous is not None,'missing firing-decision tick'
 own=next(b for b in previous['botStates'] if b['id']==uid);scans=[e for e in previous['events'] if e['type']=='ScannedBotEvent' and e['scannedByBotId']==uid];assert scans,'launch lacks fresh legal firing-decision scan'
 enemy=scans[-1];dx,dy=enemy['x']-own['x'],enemy['y']-own['y'];bearing=math.degrees(math.atan2(dy,dx));distance=math.hypot(dx,dy)
 lateral=enemy['speed']*math.sin(math.radians(enemy['direction']-bearing));radial=enemy['speed']*math.cos(math.radians(enemy['direction']-bearing))
 travel=enemy['direction']+(180 if enemy['speed']<0 else 0);cos,sin=math.cos(math.radians(travel)),math.sin(math.radians(travel))
 def boundary(value,component,size):return math.inf if abs(component)<1e-9 else ((size-18 if component>0 else 18)-value)/component
 wall=min(boundary(enemy['x'],cos,800),boundary(enemy['y'],sin,600));context=dict(lateral=abs(lateral),radial=radial,wall=max(0,min(1,wall/180)),flight=distance/(20-3*launch['power']))
 near(launch['bearing'],bearing);near(launch['range'],distance);assert launch['sign']==(-1 if lateral<0 else 1);compare(context,launch['context'],'legal launch context')

def legal_pose(p):
 assert set(p)=={'x','y','h','v'};finite(p);assert abs(p['v'])<=8+1e-9

def private_peak(data,game,markers,official,tickrounds,uid):
 directory=data/'state';assert {p.name for p in directory.iterdir()}=={'cold.json','endpoint.json','labels.jsonl',*[f'round-{n:03d}.json' for n in range(1,game['rounds']+1)]}
 cold=read(directory/'cold.json');finite(cold);coldhash=sha(directory/'cold.json')
 assert cold['name']=='MeridianPeak' and cold['version']=='0.1.0' and cold['seed']==game['seed'] and cold['checkpointLoaded'] is False
 assert cold['round']==cold['completedRounds']==cold['revision']==cold['cumulativeCrossings']==cold['cumulativeAcceptedHits']==0 and cold['sealed'] is False and cold['coldStateSHA256'] is None
 assert not any(cold['stats'].values());assert cold['models']['gunSkill']==[.2,.3,.25] and not cold['models']['pendingTrials']
 assert cold['models']['hitAngles']==dict(weights=[0.]*21,effectiveTotal=0.)
 assert cold['models']['crossingGun']==dict(capacity=192,count=0,next=0,examples=[],inFlight=[],requested=None)
 journal=[json.loads(l) for l in (directory/'labels.jsonl').read_text().splitlines()];finite(journal);assert all(set(e)=={'round','type','label'} for e in journal)
 assert [e['round'] for e in journal]==sorted(e['round'] for e in journal) and all(1<=e['round']<=game['rounds'] for e in journal)
 ring={};nextindex=0;crossings=accepted=0;skill=[.2,.3,.25];summaries=[]
 for number,per in enumerate(official['perRound'],1):
  model=read(directory/f'round-{number:03d}.json');finite(model)
  assert set(model)==set(cold) and model['schemaVersion']==1 and model['name']=='MeridianPeak' and model['version']=='0.1.0' and model['seed']==game['seed'] and model['checkpointLoaded'] is False
  assert model['round']==model['completedRounds']==number and model['sealed'] is True and type(model['revision']) is int and model['revision']>=1 and model['coldStateSHA256']==coldhash
  stats=model['stats'];assert set(stats)==set(cold['stats']);assert all(type(v) in [int,float] and v>=0 for v in stats.values())
  floats={'shotEnergy','damageDealt','damageTaken','maximumWaveRisk'};assert all(type(v) is int for k,v in stats.items() if k not in floats)
  assert stats['skippedTurns']==0 and stats['maximumWaveRisk']<=80+1e-9 and stats['maximumRaysPerWave']<=23 and stats['maximumModeledRays']<=23*stats['maximumActiveWaves']
  assert stats['maximumDecisionNanos']<=stats['decisionTotalNanos'] and stats['plannerDecisions']<=per['turns']
  for a,b in [('confirmedShots','shots'),('hits','callbackDeliverableOutgoingHits'),('hitsTaken','hitsTaken'),('skippedTurns','skips')]:assert stats[a]==per[b],(number,a,stats[a],per[b])
  for a,b in [('shotEnergy','shotEnergy'),('damageDealt','callbackDeliverableOutgoingDamage'),('damageTaken','damageTaken')]:near(stats[a],per[b],a)
  mm=[m.split() for m in markers if int(m.split()[0])==number];fires=[m for m in mm if len(m)==4 and m[1]=='fire'];assert len(fires)==stats['confirmedShots'] and len({int(m[2]) for m in fires})==len(fires)
  assert all(1<=int(m[2])<=per['turns'] for m in fires);near(sum(float(m[3]) for m in fires),stats['shotEnergy'])
  hits=[m[2] for m in mm if len(m)==3 and m[1]=='hit'];assert all(m in ['ACCEPTED','UNMATCHED','AMBIGUOUS','UNORIENTED','OUTSIDE'] for m in hits)
  assert len(hits)==stats['hitsTaken'] and hits.count('ACCEPTED')==stats['acceptedHitLabels'];accepted+=hits.count('ACCEPTED')
  assert sum(int(m[2]) for m in mm if len(m)==3 and m[1]=='crossings')==stats['crossingLabels']
  byturn={t['turnNumber']:t for t in tickrounds[number]};shots=[(t['turnNumber'],e['bullet']) for t in tickrounds[number] for e in t['events'] if e['type']=='BulletFiredEvent' and e['bullet']['ownerId']==uid]
  reasons=collections.Counter();labelled=set()
  for entry in [e for e in journal if e['round']==number]:
   label=entry['label'];reason=label['reason'];reasons[entry['type'],reason]+=1
   if entry['type']=='crossing':
    assert set(label)=={'turn','launch','enemy','previousTurn','previous','reason','offset'};launch=label['launch'];launch_context(launch,byturn,uid,shots);assert launch['turn'] not in labelled;labelled.add(launch['turn'])
    t=label['turn'];assert t in byturn and t>=launch['turn'];scans=[e for e in byturn[t]['events'] if e['type']=='ScannedBotEvent' and e['scannedByBotId']==uid];assert scans
    enemy=label['enemy'];legal_pose(enemy);compare(dict(x=scans[-1]['x'],y=scans[-1]['y'],h=scans[-1]['direction'],v=scans[-1]['speed']),enemy,'legal label scan')
    age=t-launch['turn']+1;v=20-3*launch['power'];assert math.hypot(enemy['x']-launch['x'],enemy['y']-launch['y'])<=v*(age+1)+1e-9
    prev=label['previous'];pt=label['previousTurn']
    if prev is None or pt!=t-1:expected_reason='MISSING_BRACKET'
    else:
     legal_pose(prev);ps=[e for e in byturn[pt]['events'] if e['type']=='ScannedBotEvent' and e['scannedByBotId']==uid];assert ps;compare(dict(x=ps[-1]['x'],y=ps[-1]['y'],h=ps[-1]['direction'],v=ps[-1]['speed']),prev)
     if math.hypot(prev['x']-launch['x'],prev['y']-launch['y'])<=age*v:expected_reason='INSIDE_PREVIOUS_FRONT'
     else:
      angle=math.degrees(math.atan2(enemy['y']-launch['y'],enemy['x']-launch['x']));offset=math.remainder(angle-launch['bearing'],360)*launch['sign']/math.degrees(math.asin(8/v));near(offset,label['offset']);expected_reason='OUTSIDE_OFFSET' if abs(offset)>1.25 else 'RESOLVED'
    assert reason==expected_reason,(reason,expected_reason)
    if reason=='RESOLVED':ring[nextindex]=dict(index=nextindex,context=launch['context'],offset=label['offset']);nextindex=(nextindex+1)%192;crossings+=1
   elif entry['type']=='trial':
    assert set(label)=={'observedTurn','reason','trial','enemy'};t=label['observedTurn'];trial=label['trial'];enemy=label['enemy'];legal_pose(enemy);assert set(trial)=={'due','angles','x','y'} and len(trial['angles'])==3
    assert t in byturn;scans=[e for e in byturn[t]['events'] if e['type']=='ScannedBotEvent' and e['scannedByBotId']==uid];assert scans;compare(dict(x=scans[-1]['x'],y=scans[-1]['y'],h=scans[-1]['direction'],v=scans[-1]['speed']),enemy)
    assert (reason=='SCORED' and trial['due']==t) or (reason=='MISSING_SCAN' and trial['due']<t)
    if reason=='SCORED':
     angle=math.degrees(math.atan2(enemy['y']-trial['y'],enemy['x']-trial['x']));radius=math.hypot(enemy['x']-trial['x'],enemy['y']-trial['y'])
     for i in range(3):miss=2*radius*abs(math.sin(math.radians(math.remainder(angle-trial['angles'][i],360)/2)));skill[i]=.94*skill[i]+.06/(1+miss*miss/324)
   else:raise AssertionError('Unknown label type')
  for typ,reason,key in [('crossing','RESOLVED','crossingLabels'),('crossing','MISSING_BRACKET','crossingMaskedMissingBracket'),('crossing','INSIDE_PREVIOUS_FRONT','crossingMaskedInsideFront'),('crossing','OUTSIDE_OFFSET','crossingMaskedOutsideOffset'),('trial','SCORED','trialScored'),('trial','MISSING_SCAN','trialMissingScan')]:assert reasons[typ,reason]==stats[key],key
  assert model['cumulativeCrossings']==crossings and model['cumulativeAcceptedHits']==accepted
  models=model['models'];assert set(models)=={'hitAngles','crossingGun','gunSkill','pendingTrials'};compare(skill,models['gunSkill'],'replayed virtual-ray skill')
  angles=models['hitAngles'];assert set(angles)=={'weights','effectiveTotal'} and len(angles['weights'])==21 and min(angles['weights'])>=0;near(sum(angles['weights']),angles['effectiveTotal'],tol=1e-9);near((1-.95**accepted)/.05,angles['effectiveTotal'],tol=1e-9)
  gun=models['crossingGun'];assert set(gun)=={'capacity','count','next','examples','inFlight','requested'} and gun['capacity']==192 and gun['count']==min(crossings,192) and gun['next']==nextindex
  compare([ring[i] for i in range(min(crossings,192))],gun['examples'],'replayed legal crossing bank')
  assert len(gun['inFlight'])<=stats['confirmedShots'] and stats['crossingLabels']<=stats['confirmedShots']
  for pending in gun['inFlight']:launch_context(pending,byturn,uid,shots)
  if gun['requested'] is not None:finite(gun['requested']);assert gun['requested']['turn']>=1 and .1<=gun['requested']['power']<=3
  for pending in models['pendingTrials']:finite(pending);assert set(pending)=={'due','angles','x','y'} and len(pending['angles'])==3
  summaries.append(dict(round=number,revision=model['revision'],sha256=sha(directory/f'round-{number:03d}.json'),cumulativeCrossings=crossings,cumulativeAcceptedHits=accepted))
 assert (directory/'endpoint.json').read_bytes()==(directory/f"round-{game['rounds']:03d}.json").read_bytes()
 return dict(coldSHA256=coldhash,endpointSHA256=sha(directory/'endpoint.json'),labelsSHA256=sha(directory/'labels.jsonl'),labelRows=len(journal),cumulativeCrossings=crossings,cumulativeAcceptedHits=accepted,seals=summaries,limitations='Incoming 21-bin positions are not independently reconstructed; nonnegativity, decayed mass, accepted-count conservation and source provenance are checked. Outgoing contexts, crossing bank and virtual-ray skills are replayed from legal scans/journal.')

if A.artifact_zip:
 assert A.artifact_sha256 and sha(A.artifact_zip)==A.artifact_sha256,'Missing/wrong expected original ZIP digest'
 with zipfile.ZipFile(A.artifact_zip) as z:
  prefix='build/meridian-peak/screen/acquisition/'
  for name in z.namelist():
   if name.startswith(prefix) and not name.endswith('/'):assert z.read(name)==(RAW/name[len(prefix):]).read_bytes(),name
assert sha(RAW/'freeze.json')==FREEZE and (RAW/'freeze.json').read_bytes()==(REG/'freeze.json').read_bytes();frozen=read(RAW/'freeze.json')
for name,h in {**frozen['assets'],**frozen['tooling']}.items():assert sha(ROOT/name)==h,name
profile=read(RAW/'profile.json');assert profile['runtime'] in frozen['allowedRuntime'] and profile['assets']==frozen['assets'] and profile['walls']==frozen['walls'];assert sha(RAW/'package-contents.json')==PACKAGE
host=read(RAW/'host.json');activation=read(RAW/'activation.json');assert activation['experiment']=='meridian-peak-v1' and activation['stage']=='screen' and activation['payloadSHA']==HEAD and activation['freezeSHA256']==FREEZE and activation['runAttempt']==1
assert activation['receiptRef']=='refs/tags/experiment-receipts/meridian-peak-v1-screen' and len(activation['receiptObject'])==40
assert activation['hostIdentity']==host and activation['hostDocumentSHA256']==sha(RAW/'host.json')
assert host['stage']=='screen' and host['runID']==activation['runID'] and host['runAttempt']=='1' and host['bootID'] and host['workspace']=='/home/runner/work/robocode/robocode'
assert host['profileSHA256']==sha(RAW/'profile.json') and host['packageSHA256']==PACKAGE and host['freezeSHA256']==FREEZE
assert host['toolingSHA256']==hashlib.sha256(json.dumps(frozen['tooling'],sort_keys=True).encode()).hexdigest()
for name,h in read(RAW/'artifact-files.json').items():assert sha(RAW/name)==h,name
plan=[g.copy() for g in read(REG/'games.json') if g['stage']=='screen'];assert len(plan)==5 and sum(g['rounds'] for g in plan)==25
for g in plan:
 if g['enemy']=='Walls':g['version']=frozen['walls']['version']
decision=read(RAW/'decision.json');assert decision['stage']=='screen' and decision['freezeSHA256']==FREEZE and decision['status']=='complete' and len(decision['games'])==5
assert {p.name for p in (RAW/'raw').iterdir()}=={g['id'] for g in plan}
rows=[];setup=dict(gameType='1v1',arenaWidth=800,arenaHeight=600,minNumberOfParticipants=2,maxNumberOfParticipants=2,gunCoolingRate=.1,maxInactivityTurns=450,turnTimeout=100000,readyTimeout=10000000,defaultTurnsPerSecond=-1)
for game,declared in zip(plan,decision['games']):
 compare(game,declared,'schedule');assert declared['status']=='valid';data=RAW/'raw'/game['id'];stage=RAW/'stages'/game['id'];attempt=read(data/'attempt.json')
 assert attempt['game']==game and attempt['payloadSHA']==HEAD and attempt['freezeSHA256']==FREEZE and attempt['profileSHA256']==sha(RAW/'profile.json')
 assert attempt['command'][4:6]==[str(game['port']),str(game['rounds'])]
 env=attempt['environment'];assert env['HOME'].endswith('/'+game['id']+'/home')
 assert env['MERIDIAN_SEED' if game['arm']=='control' else 'MERIDIAN_PEAK_SEED']==str(game['seed'])
 if game['arm']=='candidate':assert env['MERIDIAN_PEAK_STATE_DIR'].endswith('/'+game['id']+'/state')
 for name,h in attempt['stagedSHA256'].items():assert sha(stage/name)==h,(game['id'],name)
 assert sha(stage/game['ownName']/(game['ownName']+'.jar'))==(CONTROL if game['arm']=='control' else frozen['assets']['experiments/MeridianPeak/MeridianPeak.jar'])
 if game['enemy']=='Diamond Port':
  lock=read(ROOT/'tools/opponents/strong-melee.lock.json');items=[x for x in lock['files'] if x['path']=='LICENSE' or x['path'].startswith(('bots/bot_core/','bots/ports/diamond-port/'))];assert len(items)==104
  for x in items:
   rel='opponents/LICENSE.shared-Apache-2.0' if x['path']=='LICENSE' else 'opponents/'+x['path'].removeprefix('bots/ports/').removeprefix('bots/');assert sha(stage/rel)==x['sha256']
 elif game['enemy']=='Retcon':
  for x in read(ROOT/'tools/opponents/retcon.lock.json')['files']:
   b=(stage/'Retcon'/x['path']).read_bytes();assert len(b)==x['size'] and hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==x['gitBlobSHA']
 elif game['enemy']=='Walls':
  for name,h in frozen['walls']['files'].items():assert sha(stage/'official-samples'/name)==h
 else:assert sha(stage/'AISurvivor/AISurvivor.jar')==frozen['assets']['AISurvivor/AISurvivor.jar']
 files=list(data.glob('*.battle.gz'));assert len(files)==1
 with gzip.open(files[0],'rt') as stream:events=[json.loads(line) for line in stream]
 finite(events);assert events[0]['type']=='GameStartedEventForObserver' and events[-1]['type']=='GameEndedEventForObserver'
 for key,value in dict(setup,numberOfRounds=game['rounds']).items():assert events[0]['gameSetup'][key]==value,key
 expected=sorted([(game['ownName'],game['ownVersion']),(game['enemy'],game['version'])]);results=events[-1]['results'];assert sorted((x['name'],x['version']) for x in results)==expected and sorted((x['name'],x['version']) for x in events[0]['participants'])==expected
 assert events[-1]['numberOfRounds']==game['rounds'] and not any('Skipped' in e['type'] for e in events);uid=next(r['id'] for r in results if r['name']==game['ownName'])
 markers=(data/'markers.txt').read_text().splitlines();numbers=list(range(1,game['rounds']+1))
 for tag in ['start','end']:assert [int(m.split()[0]) for m in markers if m.endswith(' '+tag)]==numbers
 assert not any(' skip ' in m for m in markers)
 endpoints=[e for e in events if e['type']=='RoundEndedEventForObserver'];assert [e['roundNumber'] for e in endpoints]==numbers
 perround=[];tickrounds={};previous={r['id']:(0,0) for r in results};allticks=[]
 for n,end in zip(numbers,endpoints):
  ticks=[e for e in events if e['type']=='TickEventForObserver' and e['roundNumber']==n];assert [e['turnNumber'] for e in ticks]==list(range(1,end['turnNumber']+1));tickrounds[n]=ticks;allticks.extend(ticks)
  r=countrow(ticks,uid);scores={x['id']:x for x in end['results']};assert set(scores)==set(previous);delta={i:x['totalScore']-previous[i][0] for i,x in scores.items()};assert min(delta.values())>=0;total=sum(delta.values())
  r.update(round=n,score=delta[uid],totalScore=total,share=delta[uid]/total if total else None,wins=scores[uid]['firstPlaces']-previous[uid][1]);previous={i:(x['totalScore'],x['firstPlaces']) for i,x in scores.items()};compare(r,declared['perRound'][n-1],game['id']+' round'+str(n));perround.append(r)
 for x in results:assert previous[x['id']]==(x['totalScore'],x['firstPlaces'])
 r=countrow(allticks,uid);assert r['shots']>0 and r['enemyShots']>0;me=next(x for x in results if x['id']==uid);total=sum(x['totalScore'] for x in results);assert total>0
 r.update(share=me['totalScore']/total,wins=me['firstPlaces'],winRate=me['firstPlaces']/game['rounds'],perRound=perround,results=results,turnsByRound=[x['turns'] for x in perround],recordingSHA256=sha(files[0]));compare(r,declared,game['id'])
 if game['enemy']=='AISurvivor':compare(validate_ais(data,game),declared['aisEndpoint'])
 private=private_peak(data,game,markers,r,tickrounds,uid) if game['arm']=='candidate' else dict(available=False,limitation='Original-v3 control has markers only; no Peak private models')
 cost=read(data/'process-cost.json');finite(cost);assert all(cost[k]>=0 for k in ['elapsedWallSeconds','waitedChildUserSecondsDelta','waitedChildSystemSecondsDelta']);compare(cost,declared['processCost'])
 rows.append(dict(**game,**r,privateIndependentAudit=private,processCost=cost))
control,diamond,*transfer=rows
passed=diamond['share']>control['share'] and diamond['damagePer100EnemyShots']<control['damagePer100EnemyShots'] and all(r['share']>.35 and r['wins']>=1 for r in [diamond,*transfer])
assert decision['decision'] is passed,(decision['decision'],passed)
output=dict(audit='PASS',stage='screen',independent=True,head=HEAD,freezeSHA256=FREEZE,runID=activation['runID'],receiptObject=activation['receiptObject'],physicalRounds=25,invalidRounds=0,screenPass=passed,games=rows,artifactSHA256=sha(A.artifact_zip) if A.artifact_zip else None,limitations='No repository reducer or candidate evaluator imported; no new game, matrix or Java process. Full staged archive required. Incoming histogram bin positions are not reconstructed; finite legal mass/accepted-label conservation and source provenance checked. Outgoing bank and gun-skill updates independently replayed from actual legal scan labels. Offline audit cannot prove absence of external unrecorded games; live durable-ref/run identity must be independently queried.')
if A.output:A.output.parent.mkdir(parents=True,exist_ok=True);A.output.write_text(json.dumps(output,indent=2,sort_keys=True)+'\n')
for r in rows:print(r['id'],f"share={100*r['share']:.6f}% wins={r['wins']}/{r['rounds']} shots/hits={r['shots']}/{r['hits']} damage={r['damageDealt']:.6f}/{r['damageTaken']:.6f}")
print('Independent full SCREEN audit PASS; registered screen decision:',passed)
