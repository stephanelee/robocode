"""Streaming own/public review. Never opens runner logs or opponent/model internals."""
import argparse,json,hashlib,math,statistics,zipfile,io,re
from pathlib import Path
from collections import Counter
p=argparse.ArgumentParser();p.add_argument('--zip',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--checkout',type=Path);args=p.parse_args()
PREFIX='build/facet-prepared-cohort-100/'
def filehash(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(65536),b''):h.update(b)
 return h.hexdigest()
def angular(a,b):return abs(math.atan2(math.sin(a-b),math.cos(a-b)))
def number(s,k):return float(re.search(r'\b'+re.escape(k)+r'=([-+0-9.eE]+)(?= |,|$)',s)[1])
def quantiles(h):
 out={}
 for k,v in h.items():
  q={};n=sum(v)
  for f in [.5,.95,.99,1.]:
   acc=0
   for j,x in enumerate(v):
    acc+=x
    if acc>=math.ceil(n*f):q[str(f)]=[2**j/1e6,2**(j+1)/1e6];break
  out[k]=dict(count=n,bucketIntervalsMillis=q)
 return out
assert filehash(args.zip)=='ef24b160d461008783fb7bb293c114addbcaa066ebcd40ca48ea383d74143ac4'
with zipfile.ZipFile(args.zip) as z:
 def raw(name):return z.read(PREFIX+name)
 def read(name):return json.loads(raw(name))
 def rows(name):
  with z.open(PREFIX+name) as f:
   for line in io.TextIOWrapper(f,encoding='utf-8'):yield json.loads(line)
 report=read('results.json');reg=read('registration.json');regsha=hashlib.sha256(raw('registration.json')).hexdigest()
 assert regsha=='1713338d6b6cd9d0d271173a4243f238642913ebe3e717d72b471feaf07ecb58'
 pinValidation=None
 if args.checkout:
  checkout=args.checkout.resolve()
  for relative,expected in reg['fileSHA256'].items():
   path=(checkout/relative).resolve();assert path.is_relative_to(checkout) and filehash(path)==expected
  sourcefiles={}
  for relative,expected in reg['facetSourceFiles'].items():
   path=(checkout/'experiments/Facet'/relative).resolve();assert path.is_relative_to(checkout/'experiments/Facet')
   digest=filehash(path);assert digest==expected;sourcefiles[relative]=digest
  encoded=''.join(f'{name}\0{digest}\n' for name,digest in sorted(sourcefiles.items()))
  assert hashlib.sha256(encoded.encode()).hexdigest()==reg['facetSourceSHA256']
  pinValidation=dict(filePins=len(reg['fileSHA256']),sourceInputs=len(sourcefiles),allMatch=True)
 assert report['status']=='complete' and len(report['games'])==5 and reg['requirePreparedCohort'] is True
 games=[];ids=set();counters=Counter();totals=Counter();obs={};hist={};criticaltotals=Counter()
 for i,record in enumerate(report['games'],1):
  d=f'diamond-port-g{i:02d}/';official=read(d+'battle.json');ms=list(rows(d+'FACET_metrics.jsonl'));last=ms[-1]
  assert record['status']=='complete' and record['stage']=='complete' and record['initialFacetModel']=='absent' and record['seed']==49666+i
  assert record['launchOrder']==(['Facet','diamond-port'] if i%2 else ['diamond-port','Facet'])
  assert official==record['official'] and official['numberOfRounds']==20
  assert [x['round'] for x in ms]==list(range(1,21));assert len({x['gameId'] for x in ms})==1 and last['gameId'] not in ids;ids.add(last['gameId'])
  for row in ms:
   assert row['schema']==3 and all(row[k]=='success' for k in ['snapshotStatus','terminalObservation','terminalHash'])
   s=row['shotsConfirmed'];c=row['counters'];o=row['observationsCountSumMinMax']
   assert 0<=s<=row['shotsAccepted']<=row['shotsRequested']
   assert c.get('gun.cohort.confirmed',0)==s==c.get('gun.cohort.common',0)+c.get('gun.cohort.analyticOnly',0)
   assert not c.get('gun.cohort.missing',0) and not c.get('gun.cohort.failure',0)
   for k,lo,hi in [('headingErrorRadians',0,math.radians(.5)+1e-9),('requestAge',1,2)]:
    v=o.get('gun.cohort.'+k)
    if s:assert v[0]==s and all(math.isfinite(x) for x in v) and lo<=v[2]<=v[3]<=hi
  assert last==record['finalTelemetry']
  h=hashlib.sha256();size=0
  with z.open(PREFIX+d+'FACET_model.dat') as f:
   for b in iter(lambda:f.read(65536),b''):h.update(b);size+=len(b)
  assert h.hexdigest()==record['facetModelSHA256'] and 0<size<=134217728
  kinds=Counter();sampleKinds=Counter();sampleIds=set();byround=Counter();maxerror=0;versionDiverged=0
  for row in rows(d+'FACET_metrics.jsonl.critical.jsonl'):
   kinds[row['kind']]+=1
   if row['kind']=='gun.cohort.confirmed':
    text=row['value'];selected=re.search(r'\bselected=(analytic|long|recent)\b',text)[1]
    predictions={m[1]:(int(m[2]),int(m[3]),float(m[4]),float(m[5]),int(m[6])) for m in re.finditer(r'(analytic|long|recent)=Prediction\[gun=\1, wave=(-?\d+), turn=(\d+), angle=([-+0-9.eE]+), power=([-+0-9.eE]+), version=(\d+)\]',text)}
    assert set(predictions) in [{'analytic'},{'analytic','long','recent'}] and selected in predictions
    sampleKinds['common' if len(predictions)==3 else 'analyticOnly']+=1
    wave=int(number(text,'physicalWave'));source=int(number(text,'preparedWave'));prep=int(number(text,'preparedTurn'));request=int(number(text,'request'));creation=int(number(text,'creation'))
    assert (row['round'],wave) not in sampleIds;sampleIds.add((row['round'],wave));byround[row['round']]+=1
    assert 1<=request-prep<=2 and int(number(text,'expectedOriginTurn'))==prep+1 and creation==request+1==row['turn']
    assert all(x[0]==source and x[1]==prep for x in predictions.values())
    error=angular(number(text,'actualAngle'),predictions[selected][2]);assert error<=math.radians(.5)+1e-9;maxerror=max(maxerror,error)
    views={m[1]:[int(v.strip()) for v in m[2].split(',') if v.strip()] for m in re.finditer(r'(analytic|long|recent)=\[([^\]]*)\]',text.split(' modelViews=')[1])}
    assert set(views)==set(predictions) and len(views['analytic'])==0
    if 'long' in views:assert len(views['long'])==1 and len(views['recent'])==3
    if 'recent' in views and len(set(views['recent']))>1:versionDiverged+=1
  assert kinds['completion.persisted']==kinds['model.roundHash']==20 and all(v<=16 for v in byround.values())
  criticaltotals.update(kinds)
  traceN=0;lasttrace=None
  for row in rows(d+'FACET_trace.jsonl'):traceN+=1;lasttrace=dict(round=row['round'],turn=row['turn'])
  assert traceN==100000
  for k in ['shotsRequested','shotsAccepted','shotsConfirmed','shotsCancelled','unmatchedConfirmations','neighborEvictions','waveEvictions','surfTruncations','geometryFallbacks','gcCount','gcMillis']:totals[k]+=last[k]
  counters.update(last['counters'])
  for k,v in last['observationsCountSumMinMax'].items():
   a=obs.setdefault(k,[0,0,math.inf,-math.inf]);a[0]+=v[0];a[1]+=v[1];a[2]=min(a[2],v[2]);a[3]=max(a[3],v[3])
  for k,v in last['timingLog2Nanos'].items():
   a=hist.setdefault(k,[0]*len(v))
   for j,x in enumerate(v):a[j]+=x
  facet=next(x for x in official['results'] if x['name']=='Facet');op=next(x for x in official['results'] if x['name']=='Diamond Port');assert op['version']=='1.8.28' and facet['version']=='0.1.0'
  games.append(dict(game=i,gameId=last['gameId'],seed=record['seed'],launchOrder=record['launchOrder'],facet=facet,opponent=op,scoreShare=facet['totalScore']/(facet['totalScore']+op['totalScore']),modelSHA256=h.hexdigest(),modelBytes=size,finalCumulativeTelemetry=last,criticalKinds=dict(kinds),cohortSamples=len(sampleIds),cohortSampleKinds=dict(sampleKinds),sampleRounds=dict(byround),sampleHeadingErrorMaxRadians=maxerror,samplesWithDivergentRecentVersions=versionDiverged,traceRows=traceN,lastTraceIdentity=lasttrace,latency=quantiles(last['timingLog2Nanos'])))
 share=statistics.mean(g['scoreShare'] for g in games);assert abs(share-report['summary']['diamond-port']['equalGameMeanScoreShare'])<1e-15
 result=dict(status='valid-complete-reject-profile',pinValidation=pinValidation,registrationSHA256=regsha,sourceSHA256=reg['facetSourceSHA256'],jarSHA256=reg['facetJarSHA256'],configSHA256=reg['facetConfigSHA256'],candidateCommit=reg['candidateCommit'],officialRounds=100,firstPlaces=sum(g['facet']['firstPlaces'] for g in games),equalGameScoreShare=share,officialFacetScore=sum(g['facet']['totalScore'] for g in games),officialOpponentScore=sum(g['opponent']['totalScore'] for g in games),officialFacetBulletDamage=sum(g['facet']['bulletDamage'] for g in games),officialOpponentBulletDamage=sum(g['opponent']['bulletDamage'] for g in games),cumulativeTelemetryTotals=dict(totals),counters=dict(counters),observations={k:dict(count=v[0],sum=v[1],min=v[2],max=v[3],mean=v[1]/v[0] if v[0] else None) for k,v in obs.items()},latency=quantiles(hist),criticalKinds=dict(criticaltotals),cohortSamples=sum(g['cohortSamples'] for g in games),registeredGateAssessment=report['summary']['diamond-port']['registeredGateAssessment'],observedRuntimeBeforeGames=report['observedRuntimeBeforeGames'],runnerSHA256=report['runnerSHA256'],retconPrepared=report['retcon'] is not None,games=games)
args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
for g in games:print(g['game'],g['facet']['firstPlaces'],g['facet']['totalScore'],g['opponent']['totalScore'],g['scoreShare'],'damage',g['facet']['bulletDamage'],g['opponent']['bulletDamage'],'shot/hit',g['finalCumulativeTelemetry']['shotsConfirmed'],g['finalCumulativeTelemetry']['counters'].get('shot.hit',0),'samples',g['cohortSamples'])
print('TOTAL',result['firstPlaces'],share,dict(totals),dict(counters));print('observations',{k:result['observations'][k] for k in ['range','scanAge','damageGiven','damageTaken','gun.cohort.headingErrorRadians','gun.cohort.requestAge']});print('latency',result['latency']['decision']);print('sampletotal',result['cohortSamples'])
