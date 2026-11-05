import json,re,math,hashlib
from pathlib import Path
BASE=Path('/tmp/facet-melee-adapter/attempt-1');DEST=Path('/tmp/facet-review/melee-final')
def load(p):
 def reject(v):raise ValueError(v)
 return [json.loads(l,parse_constant=reject) for l in p.read_text().splitlines()]
def first(r,key):return re.search(r'\b'+key+r'=([^ ]+)',r)[1]
def store(s):
 out={}
 for item in s.strip('[]').split(';'):
  if not item:continue
  out[int(first(item,'target'))]={'alive':first(item,'alive')=='true','scan':int(first(item,'scan')),'long':list(map(int,first(item,'long').split(':'))),'recent':[list(map(int,x.split(':'))) for x in first(item,'recent').strip(',').split(',')]}
 return out
report={'schema':1,'purpose':'independent actual own-three-bot semantic diagnostic, not strength','status':'running','games':[],'limitations':['The 78f460 artifact is the actually acquired binary. Its full-store auxiliary duel-comparison queries in melee are not charged to combined-aim telemetry; a final source correction remains required before benchmarking.','Actual queried-row total in these acquisitions is bounded below80 using monotone observed store sizes plus source query rules; omitted auxiliary queries were not directly instrumented. This is not universal full-store cap acceptance.','Analytic may win naturally in survivor mode; physically selected long/recent shots are not required by the prospective protocol.','Actual rejected requests, forced skips and multi-contact families remain unexercised/fixture-only; uncertain incoming associations are retained diagnostics, not ghost physical shots.','Individual unselected prediction angles are not logged; eligible common gun-rating cohorts and model/store provenance are logged.','Raw event damage includes overkill and differs from capped official bullet-damage scores.']}
for game in (1,2):
 p=BASE/f'game-{game}';rows=load(p/'FACET_metrics.jsonl');tr=load(p/'FACET_trace.jsonl');battle=json.loads((p/'battle.json').read_text());probes=[r for label in ('ContractAlpha','ContractBeta') for r in load(p/f'FACET_{label}.jsonl')]
 facetid=next(x['id'] for x in battle['results'] if x['name']=='Facet')
 assert battle['numberOfRounds']==2 and {(x['name'],x['version']) for x in battle['results']}=={('Facet','0.1.0'),('ContractAlpha','1.0'),('ContractBeta','1.0')}
 assert len(rows)==2 and [x['round'] for x in rows]==[1,2] and len({x['gameId'] for x in rows})==1
 assert not any(x['kind']=='failure' for x in tr);assert not any(x['type']=='skip' for x in probes)
 fired=[x for x in tr if x['kind']=='BulletFiredEvent'];confirmed=[x for x in tr if x['kind']=='fire.confirmed'];hits=[x for x in tr if x['kind']=='BulletHitBotEvent'];received=[x for x in probes if x['type']=='hitBy' and x['owner']==facetid]
 assert len(fired)==len(confirmed)==rows[-1]['shotsConfirmed']==55 and len(hits)==len(received)==rows[-1]['counters']['shot.hit']
 assert len({(x['round'],x['value'].split()[0]) for x in fired})==len(fired)
 assert {(x['round'],int(x['value'].split()[0].split(':')[-1])) for x in hits}=={(x['round'],x['bullet']) for x in received}
 prepared={(x['round'],first(x['value'],'plan')):x['value'] for x in tr if x['kind']=='fire.prepared'}
 submitted={(x['round'],first(x['value'],'plan')):x['value'].split('accepted=true ',1)[1] for x in tr if x['kind']=='fire.submitted' and 'accepted=true ' in x['value']}
 states={(x['round'],x['turn']):x['value'] for x in tr if x['kind']=='own.state'}
 transitions={}
 perround=[];maxStore=0
 for n in (1,2):
  pops=[x for x in tr if x['round']==n and x['kind']=='population.transition'];assert [int(first(x['value'],'to')) for x in pops]==[2,1,0]
  trans=pops[1];transitions[n]=trans['turn'];before,after=trans['value'].split(' before=',1)[1].split(' after=');a,b=store(before),store(after);survivors=[i for i in b if b[i]['alive']];assert len(survivors)==1;sid=survivors[0]
  assert a[sid]['long'][0]>0 and b[sid]['long'][0]>=a[sid]['long'][0] and b[sid]['long'][1]>=a[sid]['long'][1]
  for old,new in zip(a[sid]['recent'],b[sid]['recent']):assert new[0]>=old[0] and new[1]>=old[1]
  maxStore=max(maxStore,max(x['long'][0] for x in a.values()),max(x['long'][0] for x in b.values()))
  modes=[first(x['value'],'mode') for x in confirmed if x['round']==n];assert 'melee' in modes and 'survivor-duel' in modes
  assert any(x['round']==n and x['kind']=='scan.observed' and x['turn']>=trans['turn'] and int(first(x['value'],'id'))==sid for x in tr)
  for label in ('ContractAlpha','ContractBeta'):
   pr=load(p/f'FACET_{label}.jsonl');assert any(x['type']=='roundStart' and x['round']==n for x in pr)
   death=next(x for x in pr if x['type']=='death' and x['round']==n);assert any(x['round']==n and x['eventTurn']==death['eventTurn'] for x in pr if x['type']=='hitBy' and x['owner']==facetid)
  final=[i for i,x in enumerate(tr) if x['round']==n and x['kind']=='model.roundHash'];assert len(final)==1 and not any(x['round']==n and x['kind'] in ('command.prepared','command.submitted') for x in tr[final[0]+1:])
  perround.append({'round':n,'populationSequence':[2,1,0],'survivor':sid,'transitionTurn':trans['turn'],'survivorLongBeforeAfter':[a[sid]['long'],b[sid]['long']],'survivorRecentBeforeAfter':[a[sid]['recent'],b[sid]['recent']],'meleeConfirmations':modes.count('melee'),'survivorConfirmations':modes.count('survivor-duel')})
 originerr=headingerr=0
 for c in confirmed:
  n=c['round'];v=c['value'];req=int(first(v,'request'));assert c['turn']==req+1 and 'pending=true' in v
  evidence=v.split(' selected=',1)[1];key=(n,first(evidence,'plan'));assert evidence==submitted[key]==prepared[key]
  st=states[(n,req)];x,y=float(first(st,'x')),float(first(st,'y'));gun=float(first(st,'gun'));bx,by=map(float,first(v,'position').split(','));heading=math.radians(float(first(v,'heading')));origin=math.hypot(x-bx,y-by);angle=abs((heading-gun+math.pi)%(2*math.pi)-math.pi);assert origin<=1e-6 and angle<=1e-7;originerr=max(originerr,origin);headingerr=max(headingerr,angle)
  mode=first(evidence,'mode');pop=int(first(st,'population'));assert mode=='melee' and pop==2 or mode=='survivor-duel' and pop==1
  assert first(evidence,'beganMelee')=='true'
  if mode=='survivor-duel':assert int(first(evidence,'scan'))>=transitions[n]
 previousDamage=previousHits=previousShots=0
 for row in rows:
  n=row['round'];damage=sum(x['damage'] for x in received if x['round']==n);assert abs(row['observationsCountSumMinMax']['damageGiven'][1]-previousDamage-damage)<1e-6
  assert row['shotsConfirmed']-previousShots==len([x for x in fired if x['round']==n]);assert row['counters']['shot.hit']-previousHits==len([x for x in received if x['round']==n])
  assert row['unmatchedConfirmations']==row['shotsCancelled']==0
  assert not any(v for k,v in row['counters'].items() if k.endswith('.failure') or k in ('turn.skipped','turn.gap'))
  assert row['neighborBytes']<=134217728 and row['observationsCountSumMinMax']['predictorSteps'][3]<=2400 and row['observationsCountSumMinMax']['meleeDecisionSamples'][3]<=80
  assert row['observationsCountSumMinMax']['shadowSegmentWork'][3]<=4096 and row['observationsCountSumMinMax']['shadowIntervals'][3]<=2048 and row['observationsCountSumMinMax']['shadowAllocationUnits'][3]<=524288
  previousDamage=row['observationsCountSumMinMax']['damageGiven'][1];previousHits=row['counters']['shot.hit'];previousShots=row['shotsConfirmed']
 work=[x for x in tr if x['kind']=='melee.work'];combined=max(int(first(x['value'],'decisionReturned')) for x in work);assert all(int(first(x['value'],'decisionIterations'))<=960 for x in work)
 queryBound=combined+min(160,max(1,maxStore//12))+15;assert queryBound<=80
 # This is deliberately a conservative uninstrumented auxiliary-query bound, not an exact count.
 rates={}
 for x in tr:
  if x['kind']=='gun.rating':
   k=(x['round'],int(first(x['value'],'target')));rates.setdefault(k,{})[first(x['value'],'gun')]=int(first(x['value'],'cohort'))
 assert all(len(set(r.values()))==1 for r in rates.values())
 model=(p/'FACET_model.dat').read_bytes();assert model.endswith(b'\x00\tEND-FACET')
 item={'game':game,'gameId':rows[0]['gameId'],'completedRounds':2,'confirmedShots':55,'outgoingHits':len(hits),'probeHitReceipts':len(received),'rawEventDamage':sum(x['damage'] for x in received),'maxOriginError':originerr,'maxHeadingErrorRadians':headingerr,'postFinalizationCommands':0,'fatalTerminalFacetHitsPreserved':4,'rounds':perround,'selectedSurvivorGuns':sorted({first(c['value'],'gun') for c in confirmed if 'mode=survivor-duel' in c['value']}),'maxCombinedReturnedSamples':combined,'maxMeleeStoreBound':maxStore,'conservativeTotalQueryRowUpperBound':queryBound,'commonGunRatingCohorts':[{"round":k[0],"target":k[1],"counts":v} for k,v in rates.items()],'modelSHA256':hashlib.sha256(model).hexdigest(),'hashes':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in [p/'FACET_trace.jsonl',p/'FACET_metrics.jsonl',p/'FACET_ContractAlpha.jsonl',p/'FACET_ContractBeta.jsonl',p/'battle.json',p/'Facet/Facet.jar']}}
 assert item['hashes']['Facet.jar']=='78f460de523f8eb15f7998b7f94d75ec150f1ed8e3ef4b6a0a5b4d4957afc1b9'
 report['games'].append(item)
assert report['games'][0]['gameId']!=report['games'][1]['gameId']
report['status']='passed-acquired-integration-with-declared-query-instrumentation-limit';report['scheduledRounds']=report['completedRounds']=4;report['competitiveRounds']=0
(DEST/'live-independent-validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
