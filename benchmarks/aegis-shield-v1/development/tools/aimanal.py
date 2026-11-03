import json,math,sys
rows=[json.loads(l) for l in open(sys.argv[1])]
rounds=[];cur=None
for r in rows:
    if r['ev']=='round':cur={'own':{},'scan':{},'fire':[],'hit':[]};rounds.append(cur);continue
    if r['ev']=='own':cur['own'][r['turn']]=(r['x'],r['y'])
    elif r['ev']=='scan':cur['scan'][r['turn']]=(r['x'],r['y'],r['e'])
    else:cur[r['ev']].append(r)
from collections import Counter
off=Counter();nf=0;nh=0;unm=0
for R in rounds:
    nf+=len(R['fire']);nh+=len(R['hit'])
    for h in R['hit']:
        d=math.radians(h['dir']);ux,uy=math.cos(d),math.sin(d)
        def perp(p):return -uy*(p[0]-h['bx'])+ux*(p[1]-h['by'])
        best=None
        for f in R['fire']:
            if abs(f['power']-h['power'])>0.02 or f['turn']>h['turn']:continue
            N=f['turn'];o=R['scan'].get(N-1)
            if not o:continue
            pe=abs(perp(o))
            if best is None or pe<best[0]:best=(pe,N)
        if not best or best[0]>1.0:unm+=1;continue
        N=best[1]
        ds={m-N:round(perp(R['own'][m]),2) for m in range(N-4,N+1) if m in R['own']}
        moving=len(set(R['own'][m] for m in range(N-4,N+1) if m in R['own']))>1
        if moving:print('N',N,'origin perp',round(best[0],3),ds)
        z=[k for k,v in ds.items() if abs(v)<0.05]
        off[tuple(z)]+=1
print('fires',nf,'hits',nh,'unmatched',unm,off)
for i,R in enumerate(rounds):
    out=[]
    for h in R['hit']:
        d=math.radians(h['dir']);ux,uy=math.cos(d),math.sin(d)
        perp=lambda p:-uy*(p[0]-h['bx'])+ux*(p[1]-h['by'])
        f=min((f for f in R['fire'] if abs(f['power']-h['power'])<0.02 and f['turn']<=h['turn'] and (f['turn']-1) in R['scan']),key=lambda f:abs(perp(R['scan'][f['turn']-1])))
        N=f['turn'];out.append((N,round(perp(R['own'][N-2]),2)))
    print(i,out)
