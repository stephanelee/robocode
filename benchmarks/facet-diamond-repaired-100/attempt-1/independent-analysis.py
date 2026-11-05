"""Independent public official/own FACET-only analysis; no runner/opponent logs."""
import json,hashlib,math,statistics,argparse
from collections import Counter
from pathlib import Path
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root',type=Path,required=True,help='Selected own/public extracted result directory')
parser.add_argument('--output',type=Path,required=True,help='Analysis JSON destination')
args=parser.parse_args()
P=args.root.resolve()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def lines(p):return [json.loads(s) for s in p.read_text().splitlines()]
def latency(h):
 out={}
 for k,bins in h.items():
  n=sum(bins);q={}
  for frac in [.5,.95,.99,1.]:
   s=0
   for i,v in enumerate(bins):
    s+=v
    if s>=math.ceil(frac*n):q[str(frac)]=[2**i/1e6,2**(i+1)/1e6];break
  out[k]=dict(count=n,quantileBucketIntervalsMillis=q)
 return out
reg=read(P/'registration.json');report=read(P/'results.json');assert report['status']=='complete';assert sha(P/'registration.json')=='121bfbcca6e9958a2bcaa67ff361d865e5834f2e22c1652a32ae76dd2e93aaff';assert len(report['games'])==5
games=[];ids=set();totals=Counter();counters=Counter();obs={};hist={}
for i,record in enumerate(report['games'],1):
 g=P/f'diamond-port-g{i:02d}';official=read(g/'battle.json');rows=lines(g/'FACET_metrics.jsonl');r=rows[-1];critical=lines(g/'FACET_metrics.jsonl.critical.jsonl')
 assert record['status']=='complete' and record['stage']=='complete' and record['initialFacetModel']=='absent' and record['seed']==49666+i
 assert record['launchOrder']==(['Facet','diamond-port'] if i%2 else ['diamond-port','Facet'])
 assert record['official']==official and official['numberOfRounds']==20
 assert [x['round'] for x in rows]==list(range(1,21));assert all(x['schema']==3 and all(x[k]=='success' for k in ['snapshotStatus','terminalObservation','terminalHash']) for x in rows)
 assert all(x['gameId']==r['gameId'] for x in rows);assert r['gameId'] not in ids;ids.add(r['gameId'])
 assert record['finalTelemetry']==r and record['facetModelSHA256']==sha(g/'FACET_model.dat')
 assert r['shotsRequested']==r['shotsAccepted']==r['shotsConfirmed']
 f=next(x for x in official['results'] if x['name']=='Facet');o=next(x for x in official['results'] if x['name']=='Diamond Port');assert o['version']=='1.8.28' and f['version']=='0.1.0'
 criticalKinds=Counter(x['kind'] for x in critical);assert criticalKinds['completion.persisted']==20 and criticalKinds['model.roundHash']==20
 traceN=0;traceLast=None;failures=[]
 with (g/'FACET_trace.jsonl').open() as file:
  for line in file:
   x=json.loads(line);traceN+=1;traceLast=dict(round=x['round'],turn=x['turn'])
   if x['kind']=='failure':failures.append(x)
 numeric={k:r[k] for k in ['shotsRequested','shotsAccepted','shotsConfirmed','shotsCancelled','unmatchedConfirmations','neighborEvictions','waveEvictions','surfTruncations','geometryFallbacks','gcCount','gcMillis']};totals.update(numeric);counters.update(r['counters'])
 for k,v in r['observationsCountSumMinMax'].items():
  a=obs.setdefault(k,[0,0,math.inf,-math.inf]);a[0]+=v[0];a[1]+=v[1];a[2]=min(a[2],v[2]);a[3]=max(a[3],v[3])
 for k,v in r['timingLog2Nanos'].items():
  a=hist.setdefault(k,[0]*len(v))
  for j,n in enumerate(v):a[j]+=n
 games.append(dict(game=i,seed=record['seed'],launchOrder=record['launchOrder'],gameId=r['gameId'],official=official,scoreShare=f['totalScore']/(f['totalScore']+o['totalScore']),finalCumulativeTelemetry=r,modelSHA256=record['facetModelSHA256'],modelBytes=(g/'FACET_model.dat').stat().st_size,criticalKindCounts=dict(criticalKinds),criticalFailureEvents=[x for x in critical if 'failure' in x['kind'] or 'quarantin' in x['kind']],traceLines=traceN,lastTraceIdentity=traceLast,traceFailureEvents=failures,latency=latency(r['timingLog2Nanos'])))
share=statistics.mean(g['scoreShare'] for g in games);assert abs(share-report['summary']['diamond-port']['equalGameMeanScoreShare'])<1e-15
observation={k:dict(count=v[0],sum=v[1],min=v[2],max=v[3],mean=v[1]/v[0] if v[0] else None) for k,v in obs.items()}
first=sum(next(r['firstPlaces'] for r in g['official']['results'] if r['name']=='Facet') for g in games)
score={name:dict(totalScore=sum(next(r['totalScore'] for r in g['official']['results'] if r['name']==name) for g in games),bulletDamage=sum(next(r['bulletDamage'] for r in g['official']['results'] if r['name']==name) for g in games)) for name in ['Facet','Diamond Port']}
result=dict(status='valid-complete-screen-reject-profile',registrationSHA256=sha(P/'registration.json'),sourceSHA256=reg['facetSourceSHA256'],jarSHA256=reg['facetJarSHA256'],configSHA256=reg['facetConfigSHA256'],candidateCommit=reg['candidateCommit'],prospectiveProtocolCommit=reg['prospectiveProtocolCommit'],officialRounds=100,coldGameCount=5,uniqueGameIds=len(ids),officialFirstPlaces=first,equalGameMeanScoreShare=share,officialScoreTotals=score,cumulativeTelemetryTotals=dict(totals),counters=dict(counters),observations=observation,latency=latency(hist),perProcessNeighborBytes=[g['finalCumulativeTelemetry']['neighborBytes'] for g in games],perProcessGC=[dict(count=g['finalCumulativeTelemetry']['gcCount'],millis=g['finalCumulativeTelemetry']['gcMillis']) for g in games],registeredGateAssessment=report['summary']['diamond-port']['registeredGateAssessment'],observedRuntimeBeforeGames=report['observedRuntimeBeforeGames'],runnerSHA256=report['runnerSHA256'],retconPrepared=report['retcon'] is not None,games=games)
result['observationInterpretation'] = {'shadowFallbacks':'Repeatedly sampled game-cumulative counter; sum over samples is not number of fallback events. Max is per-game observed cumulative maximum.','neighborBytes':'Per-process payload gauge; summed gauges are not peak memory.','latency':'Powers-of-two bucket intervals, not exact quantiles.'}
result['perGameShadowFallbackGaugeMax'] = [g['finalCumulativeTelemetry']['observationsCountSumMinMax']['shadowFallbacks'][3] for g in games]
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(result,indent=2)+'\n')
for g in games:
 f=next(x for x in g['official']['results'] if x['name']=='Facet');o=next(x for x in g['official']['results'] if x['name']!='Facet');r=g['finalCumulativeTelemetry'];print(g['game'],f['firstPlaces'],f['totalScore'],o['totalScore'],f['bulletDamage'],o['bulletDamage'],g['scoreShare'],'shots/hits',r['shotsConfirmed'],r['counters'].get('shot.hit',0),'gaps/unmatched',r['counters'].get('turn.gap',0),r['counters'].get('incoming.unmatched',0),'critical',g['criticalKindCounts'])
print('TOTAL',first,share,score,dict(totals),dict(counters));print('RANGE/AGE',observation['range'],observation['scanAge']);print('LATENCY',result['latency']['decision']);print('GAUGES',result['perProcessNeighborBytes'])
