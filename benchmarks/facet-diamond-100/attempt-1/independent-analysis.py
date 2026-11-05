import json,math,hashlib
from pathlib import Path
P=Path('/tmp/facet-diamond-acquisition/extracted/build/facet-diamond-100');G=P/'diamond-port-g01'
def read(p):return json.loads(p.read_text())
def lines(p):return [json.loads(x) for x in p.read_text().splitlines()]
reg=read(P/'registration.json');report=read(P/'results.json');official=read(G/'battle.json');rows=lines(G/'FACET_metrics.jsonl');critical=lines(G/'FACET_metrics.jsonl.critical.jsonl');trace=lines(G/'FACET_trace.jsonl');r=rows[-1]
f=next(x for x in official['results'] if x['name']=='Facet');o=next(x for x in official['results'] if x['name']!='Facet')
assert [r['round'] for r in rows]==list(range(1,21))
assert all(row[k]=='success' for row in rows for k in ['snapshotStatus','terminalObservation','terminalHash'])
obs={k:dict(count=v[0],sum=v[1],min=v[2],max=v[3],mean=v[1]/v[0] if v[0] else None) for k,v in r['observationsCountSumMinMax'].items()}
latency={}
for name,bins in r['timingLog2Nanos'].items():
 q={};n=sum(bins)
 for quantile in [.5,.95,.99,1.0]:
  cumulative=0
  for i,x in enumerate(bins):
   cumulative+=x
   if cumulative>=math.ceil(quantile*n):q[str(quantile)]=[2**i/1e6,2**(i+1)/1e6];break
 latency[name]=dict(count=n,quantileBucketIntervalsMillis=q)
result=dict(acquisitionStatus=report['status'],error=report.get('error'),gameStatus=report['games'][0]['status'],stage=report['games'][0]['stage'],registrationSHA256=hashlib.sha256((P/'registration.json').read_bytes()).hexdigest(),physicalRounds=official['numberOfRounds'],unrunRounds=80,facet=f,opponent=o,equalGameScoreShare=f['totalScore']/(f['totalScore']+o['totalScore']),metricRows=len(rows),missingRounds=[],allTerminalStatusesSuccess=True,finalCumulativeTelemetry=r,observations=obs,latency=latency,criticalKindCounts={k:sum(x['kind']==k for x in critical) for k in sorted({x['kind'] for x in critical})},criticalTail=critical[-6:],traceRows=len(trace),lastTraceIdentity=dict(round=trace[-1]['round'],turn=trace[-1]['turn']),traceFailureEvents=[x for x in trace if x['kind']=='failure'],traceGapEvents=[x for x in trace if x['kind']=='turn.gap'],criticalFailureEvents=[x for x in critical if 'fail' in x['kind'].lower() or 'export' in x['kind'].lower()],ownFinalModelPresent=(G/'FACET_model.dat').exists())
Path('/tmp/facet-review/diamond100/acquisition-analysis.json').write_text(json.dumps(result,indent=2)+'\n')
print('share',result['equalGameScoreShare'],'critical kinds',result['criticalKindCounts'],'trace last',result['lastTraceIdentity'])
print('range',obs['range'],'scanAge',obs['scanAge'],'latency',latency['decision'])
