"""Read-only exploratory geometry on retained 0.5 shots; no engine/gameplay."""
import argparse,bisect,collections,gzip,hashlib,json,math,statistics
from pathlib import Path

def sha(p):
    with p.open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def norm(a):return(a+180)%360-180
def analyze(root,out):
    out.mkdir(parents=True,exist_ok=False)
    result=json.loads((root/'results.json').read_text());games=[g for g in result['games']if g['scenario']=='melee'and g['arm']=='candidate'];assert len(games)==8
    inputs=[root/'results.json',root/'freeze.json']+[p for g in games for p in [root/g['recording'],root/'games'/g['id']/'helm-metrics.txt']]
    manifest=dict(sourceSHA256=sha(Path(__file__)),inputs={str(p):sha(p)for p in inputs});(out/'inputs.json').write_text(json.dumps(manifest,indent=2)+'\n')
    rows=[];counts=collections.Counter()
    for game in games:
        traces={}
        for line in(root/'games'/game['id']/'helm-metrics.txt').read_text().splitlines():
            if ' learntrace 'not in line:continue
            roundnum,text=line.split(' learntrace ')
            for item in text.split(';'):
                if not item:continue
                t,bid,target,power,actual,desired,circular,flag=item.split(',');key=(int(roundnum),int(bid));assert key not in traces
                traces[key]=dict(turn=int(t),target=int(target),power=float(power),actual=float(actual),desired=float(desired),circular=float(circular),learned=int(flag))
        ticks={};shots={};scans=collections.defaultdict(dict);names={}
        with gzip.open(root/game['recording'],'rt')as f:
            for line in f:
                e=json.loads(line)
                if e['type']=='GameStartedEventForObserver':names={b['id']:b['name']for b in e['participants']};own=next(i for i,n in names.items()if n=='KeelHelm')
                if e['type']!='TickEventForObserver':continue
                r,t=e['roundNumber'],e['turnNumber'];ticks[r,t]={b['id']:b for b in e['botStates']}
                for v in e['events']:
                    if v['type']=='BulletFiredEvent'and v['bullet']['ownerId']==own:
                        key=(r,v['bullet']['bulletId']);assert key not in shots;shots[key]=(t,v['bullet'])
                    if v['type']=='ScannedBotEvent'and v['scannedByBotId']==own:
                        scans[r,v['scannedBotId']].setdefault(v['turnNumber'],v)
        for key,trace in traces.items():
            r,bid=key;counts['capturedTraces']+=1
            if key not in shots:counts['unmatchedTrace']+=1;continue
            f,b=shots[key];assert f==trace['turn']and abs(b['power']-trace['power'])<1e-8 and abs(norm(b['direction']-trace['actual']))<1e-7
            row=dict(game=game['id'],round=r,bulletID=bid,fireTurn=f,target=names[trace['target']],learned=trace['learned']);rows.append(row);counts['physicalMatched']+=1
            prev=ticks.get((r,f-1),{}).get(own)
            if prev is not None:assert math.hypot(prev['x']-b['x'],prev['y']-b['y'])<1e-7;counts['originVerifiedAgainstPreviousPose']+=1
            series=scans[r,trace['target']];turns=sorted(t for t in series if t<=f-1)
            if not turns or prev is None:row['context']='missing-own-scan-or-previous-pose';counts['missingContext']+=1
            else:
                s=series[turns[-1]];age=f-1-s['turnNumber'];distance=math.hypot(prev['x']-s['x'],prev['y']-s['y']);wall=min(s['x']-18,982-s['x'],s['y']-18,982-s['y'])
                rate=None if len(turns)<2 else norm(s['direction']-series[turns[-2]]['direction'])/(turns[-1]-turns[-2])
                row.update(scanAge=age,range=distance,wallDistance=wall,headingRate=rate,rangeBin='<250'if distance<250 else'250–500'if distance<500 else'≥500',ageBin='0–3'if age<=3 else'4–7'if age<=7 else'8–12'if age<=12 else'>12',wallBin='<60'if wall<60 else'≥60',turnBin='unknown'if rate is None else'≥3'if abs(rate)>=3 else'<3')
            t=f
            while (r,t)in ticks:
                target=ticks[r,t].get(trace['target'])
                if target is None:break
                distance=math.hypot(target['x']-b['x'],target['y']-b['y'])
                if(t-f+1)*(20-3*b['power'])>=distance:
                    bearing=math.degrees(math.atan2(target['y']-b['y'],target['x']-b['x']))
                    row['crossingTurn']=t;row['angularErrors']={k:abs(norm(trace[k]-bearing))for k in ['actual','desired','circular']};counts['rangeCrossingObserved']+=1;break
                t+=1
            if 'angularErrors'not in row:row['censored']='target absent or no range crossing before observed round end';counts['crossingCensored']+=1
    groups={}
    for dimension in ['overall','target','rangeBin','ageBin','wallBin','turnBin']:
        selected=collections.defaultdict(list)
        for row in rows:
            if 'angularErrors'in row:selected['all'if dimension=='overall'else row.get(dimension,'unknown')].append(row)
        groups[dimension]={k:dict(shots=len(v),**{model:dict(meanAbsDegrees=statistics.mean(x['angularErrors'][model]for x in v),medianAbsDegrees=statistics.median(x['angularErrors'][model]for x in v))for model in ['actual','desired','circular']})for k,v in selected.items()}
    summary=dict(games=8,newGames=0,newRounds=0,counts=dict(counts),groups=groups,limitations=['Exploratory conditional actual-trajectory discrete range-crossing error, not hit probability or counterfactual strength.','Own scans reconstructed as recorded available through f−1; callback completion is not established.','Rejected learned-gun0.5 data are planning only; no reuse as fresh validation or movement tuning.'])
    (out/'shots.json').write_text(json.dumps(rows,indent=2)+'\n');(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(dict(counts=counts,overall=groups['overall']),indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('out',type=Path);a=p.parse_args();analyze(a.root,a.out)
