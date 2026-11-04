import json,math,hashlib
from pathlib import Path
from collections import Counter
P=Path('/tmp/facet-acquisition/extracted/build/facet-registered-300')
games=[]
for g in sorted(x for x in P.iterdir() if x.is_dir()):
 if not (g/'battle.json').exists():continue
 official=json.loads((g/'battle.json').read_text()); f=next(x for x in official['results'] if x['name']=='Facet');o=next(x for x in official['results'] if x['name']!='Facet')
 rows=[json.loads(x) for x in (g/'FACET_metrics.jsonl').read_text().splitlines()];r=rows[-1]
 trace=[json.loads(x) for x in (g/'FACET_trace.jsonl').read_text().splitlines()]
 obs=r['observationsCountSumMinMax']; counters=r['counters']
 stats={k:dict(count=v[0],sum=v[1],min=v[2],max=v[3],mean=v[1]/v[0] if v[0] else None) for k,v in obs.items()}
 failureOnset={}
 prev=Counter()
 for row in rows:
  for k,v in row['counters'].items():
   if 'failure' in k.lower() and v>prev[k]:failureOnset.setdefault(k,[]).append([row['round'],v-prev[k]])
  prev.update({k:v-prev[k] for k,v in row['counters'].items()})
 game=dict(game=g.name,opponent=o['name'],officialRounds=official['numberOfRounds'],facet=f,opponentOfficial=o,scoreShare=f['totalScore']/(f['totalScore']+o['totalScore']),metricRows=len(rows),missingRounds=sorted(set(range(1,21))-{x['round'] for x in rows}),finalMetricRound=r['round'],cumulativeTelemetry={k:r[k] for k in ['shotsRequested','shotsAccepted','shotsConfirmed','shotsCancelled','unmatchedConfirmations','neighborBytes','neighborEvictions','waveEvictions','surfTruncations','geometryFallbacks','gcCount','gcMillis']},counters=counters,observations=stats,timingLog2Nanos=r['timingLog2Nanos'],trace=dict(rows=len(trace),lastRound=trace[-1]['round'],lastTurn=trace[-1]['turn'],failureEvents=[x for x in trace if x['kind']=='failure']),failureOnset=failureOnset)
 games.append(game)
def cohort(gs):
 c=Counter();t=Counter();obs={};h={}
 for g in gs:
  c.update(g['counters']);t.update(g['cumulativeTelemetry'])
  for k,v in g['observations'].items():
   a=obs.setdefault(k,dict(count=0,sum=0,min=math.inf,max=-math.inf));a['count']+=v['count'];a['sum']+=v['sum'];a['min']=min(a['min'],v['min']);a['max']=max(a['max'],v['max'])
  for k,v in g['timingLog2Nanos'].items():
   a=h.setdefault(k,[0]*32)
   for i,x in enumerate(v):a[i]+=x
 for v in obs.values():v['mean']=v['sum']/v['count'] if v['count'] else None
 quantiles={}
 for k,v in h.items():
  n=sum(v);q={}
  for p in [.5,.95,.99]:
   target=math.ceil(n*p);s=0
   for i,x in enumerate(v):
    s+=x
    if s>=target:q[str(p)]=[2**i/1e6,2**(i+1)/1e6];break
  quantiles[k]=dict(count=n,bucketIntervalMillis=q)
 return dict(games=len(gs),officialRounds=sum(g['officialRounds'] for g in gs),firstPlaces=sum(g['facet']['firstPlaces'] for g in gs),facetTotalScore=sum(g['facet']['totalScore'] for g in gs),opponentTotalScore=sum(g['opponentOfficial']['totalScore'] for g in gs),equalGameScoreShare=sum(g['scoreShare'] for g in gs)/len(gs),officialFacetBulletDamage=sum(g['facet']['bulletDamage'] for g in gs),officialOpponentBulletDamage=sum(g['opponentOfficial']['bulletDamage'] for g in gs),metricRows=sum(g['metricRows'] for g in gs),summedCumulativeTelemetry=dict(t),counters=dict(c),observations=obs,latency=quantiles)
r=dict(status='invalid-acquisition-all-physical-games-retained',physicalRounds=140,games=games,cohorts={name:cohort([g for g in games if g['opponent']==name]) for name in ['AISurvivor','Retcon']},all=cohort(games))
Path('/tmp/facet-review/acquisition-1/independent-analysis.json').write_text(json.dumps(r,indent=2)+'\n')
for g in games:print(g['game'],g['facet']['firstPlaces'],g['facet']['totalScore'],g['opponentOfficial']['totalScore'],round(g['scoreShare'],6),'missing',g['missingRounds'],'failureOnset',g['failureOnset'],'traceFailure',g['trace']['failureEvents'])
for name,c in r['cohorts'].items():
 print(name,json.dumps({k:c[k] for k in ['firstPlaces','facetTotalScore','opponentTotalScore','equalGameScoreShare','officialFacetBulletDamage','officialOpponentBulletDamage','summedCumulativeTelemetry','counters']}))
 print('obs',json.dumps({k:c['observations'][k] for k in ['range','scanAge','damageGiven','damageTaken']}),'decisionLatency',c['latency']['decision'])
