import json,hashlib,math,re
from pathlib import Path
BASE=Path('/tmp/facet-adapter/attempt-1')
DEST=Path('/tmp/facet-review/revision4')
def loads(path):
 def reject(v):raise ValueError('nonfinite JSON '+v)
 return [json.loads(l,parse_constant=reject) for l in path.read_text().splitlines()]
def close(a,b):return abs(a-b)<=1e-6
report={'schema':1,'purpose':'independent live adapter integration validation, not strength','status':'running','games':[],'limitations':['Probe roundStart rows absent; callback-start observations not claimed. Probe startup logger calls state getters before first tick; required shot/hit/terminal receipts were captured.','Actual skipped-turn, rejected-request and multicontact families were not acquired in this diagnostic.','No competitive strength interpretation of stationary probe outcomes.','learning.disabled increments once per terminal drain by source control flow; no learned subsystem exception was observed.','Event damage includes overkill and can exceed official capped bullet-damage scores.']}
for game in (1,2):
 p=BASE/f'game-{game}';rows=loads(p/'FACET_metrics.jsonl');trace=loads(p/'FACET_trace.jsonl');probe=loads(p/'FACET_probe.jsonl');battle=json.loads((p/'battle.json').read_text())
 assert battle['numberOfRounds']==2 and {(r['name'],r['version']) for r in battle['results']}=={('Facet','0.1.0'),('ContractPartner','1.0')}
 assert len(rows)==2 and [r['round'] for r in rows]==[1,2] and len({r['gameId'] for r in rows})==1
 assert not [r for r in trace if r['kind']=='failure']; assert not [r for r in probe if r['type']=='skip']
 confirmed=[r for r in trace if r['kind']=='fire.confirmed'];fired=[r for r in trace if r['kind']=='BulletFiredEvent'];hit=[r for r in trace if r['kind']=='BulletHitBotEvent'];receipts=[r for r in probe if r['type']=='hitBy'];ownstates={(r['round'],r['turn']):r['value'] for r in trace if r['kind']=='own.state'}
 assert len({(r['round'],r['value'].split()[0]) for r in fired})==len(fired)==rows[-1]['shotsConfirmed']==22
 assert len(hit)==len(receipts)==rows[-1]['counters']['shot.hit']==20
 assert {(r['round'],int(r['value'].split()[0].split(':')[-1])) for r in hit}=={(r['round'],r['bullet']) for r in receipts}
 maxOrigin=maxHeading=0
 for r in confirmed:
  v=r['value'];request=int(re.search(r'request=(\d+)',v)[1]);assert 'pending=true' in v and r['turn']==request+1
  own=ownstates[(r['round'],request)]
  x=float(re.search(r'\bx=([^ ]+)',own)[1]);y=float(re.search(r'\by=([^ ]+)',own)[1]);gun=float(re.search(r'\bgun=([^ ]+)',own)[1]); bx,by=map(float,re.search(r'position=([^ ]+)',v)[1].split(','));bh=math.radians(float(re.search(r'heading=([^ ]+)',v)[1]))
  origin=math.hypot(x-bx,y-by);heading=abs((bh-gun+math.pi)%(2*math.pi)-math.pi);assert origin<=1e-6 and heading<=1e-7;maxOrigin=max(maxOrigin,origin);maxHeading=max(maxHeading,heading)
 previousDamage=0;previousHits=0;previousFired=0
 for row in rows:
  n=row['round'];receipt=[r for r in receipts if r['round']==n];got=sum(r['damage'] for r in receipt);observed=row['observationsCountSumMinMax']['damageGiven'][1]-previousDamage;assert close(observed,got)
  assert row['shotsConfirmed']-previousFired==len([r for r in fired if r['round']==n]);assert row['counters']['shot.hit']-previousHits==len(receipt)
  death=next(r for r in probe if r['round']==n and r['type']=='death');fatal=[r for r in receipt if r['eventTurn']==death['eventTurn']];assert fatal and any(r['round']==n and r['turn']==death['eventTurn'] for r in hit)
  hashes=[i for i,r in enumerate(trace) if r['round']==n and r['kind']=='model.roundHash'];assert len(hashes)==1
  assert not any(r['round']==n and r['kind'] in ('command.submitted','command.prepared') for r in trace[hashes[0]+1:])
  assert row['unmatchedConfirmations']==0 and row['shotsCancelled']==0 and row['neighborBytes']<=134217728
  assert not any(v for k,v in row['counters'].items() if k.endswith('.failure') or k in ('turn.skipped','turn.gap'))
  assert row['observationsCountSumMinMax']['predictorSteps'][3]<=2400
  previousDamage=row['observationsCountSumMinMax']['damageGiven'][1];previousHits=row['counters']['shot.hit'];previousFired=row['shotsConfirmed']
 model=(p/'FACET_model.dat').read_bytes();assert model.endswith(b'\x00\tEND-FACET') and b'FACET-OWN-GAME-EXPORT' in model[:35]
 item={'game':game,'gameId':rows[0]['gameId'],'completedRounds':2,'confirmedShots':len(fired),'outgoingHits':len(hit),'probeHitReceipts':len(receipts),'rawEventDamage':sum(r['damage'] for r in receipts),'maxOriginError':maxOrigin,'maxHeadingErrorRadians':maxHeading,'uniqueRoundRows':len(rows),'unmatchedConfirmations':rows[-1]['unmatchedConfirmations'],'postFinalizationCommands':0,'fatalTerminalHitsPreserved':2,'modelSHA256':hashlib.sha256(model).hexdigest(),'modelBytes':len(model),'hashes':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in [p/'FACET_trace.jsonl',p/'FACET_metrics.jsonl',p/'FACET_probe.jsonl',p/'battle.json',p/'Facet/Facet.jar']}}
 assert item['hashes']['Facet.jar']=='73be04d7386a60474743a78fcebf3fd0c143e37120b410b18aa668bd5ed50c16'
 report['games'].append(item)
assert report['games'][0]['gameId']!=report['games'][1]['gameId']
report['status']='passed-required-integration-checks-with-declared-probe-logging-limit';report['scheduledRounds']=report['completedRounds']=4;report['competitiveRounds']=0
(DEST/'adapter-independent-validation.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(report,indent=2,allow_nan=False))
