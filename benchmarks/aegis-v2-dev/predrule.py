# HISTORICAL exploratory script: reads local scratch paths and skips missing runs.
# Superseded by analyses.py, which reads the retained raw/ inputs and fails on missing data.
import json,math,sys
def sgn(x):return (x>0)-(x<0)
def v2(vm,v0):
    if abs(v0-vm)<1e-9:return v0
    if v0!=0 and vm!=0 and sgn(v0)!=sgn(vm):return max(-8,min(8,v0+sgn(v0)))
    if abs(v0)>abs(vm):return max(-8,min(8,v0+sgn(v0)))
    if v0==0:return -sgn(vm) if abs(vm)<=2 else 0
    if abs(v0)>2:return v0-2*sgn(v0)
    return -(1-abs(v0)/2)*sgn(v0)
def v1rule(v0,v1):
    s=1 if v0>0 else -1
    if abs(v0)>2.0001 and (abs(v1)<1e-9 or (abs(abs(v1)-(abs(v0)/2-1))<1e-6 and abs(v0)-abs(v1)>2.0001)):return v0-2*s
    return v1
err1=[];err2=[]
for run in ['instr4','aegis22','aegis20','aegis18','aegis10','instr3','instr2','instr1','aegis9','aegis8']:
    try:
        D=[json.loads(l) for l in open(f'dev/runs/{run}/diamond.jsonl') if '"aim"' in l]
        A=[json.loads(l) for l in open(f'dev/runs/{run}/trace.jsonl') if '"scan"' in l or '"round"' in l]
    except Exception: continue
    rounds=[];cur=None
    for r in A:
        if r['ev']=='round':cur={};rounds.append(cur)
        elif cur is not None:cur[r['turn']]=r
    dr=[];cur=[];last=-1
    for a in D:
        if a['t']<last-50:dr.append(cur);cur=[]
        last=a['t'];cur.append(a)
    dr.append(cur)
    for sc,aims in zip(rounds,dr):
        for a in aims:
            T=a['t'];s0=sc.get(T);s1=sc.get(T+1);sm=sc.get(T-1)
            if not(s0 and s1 and sm):continue
            h=math.radians(s1['dir'])
            for rule,acc in ((v1rule(s0['v'],s1['v']),err1),(v2(sm['v'],s0['v']),err2)):
                px=s0['x']+math.cos(h)*rule;py=s0['y']+math.sin(h)*rule
                acc.append(math.hypot(px-a['sx'],py-a['sy']))
import statistics
for name,e in (('frozen rule',err1),('new rule',err2)):
    print(name,'n',len(e),'mean',round(statistics.mean(e),3),'>1px',round(sum(x>1 for x in e)/len(e),4),'>2px',round(sum(x>2 for x in e)/len(e),4),'max',round(max(e),2))
