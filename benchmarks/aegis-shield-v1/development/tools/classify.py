import json,sys
from collections import Counter
c=Counter();rnd=0;com={};sh=set();fires={}
rows=[json.loads(l) for l in open(sys.argv[1]+'/trace.jsonl')]
for r in rows:
    if r['ev']=='round':rnd=r['round']
    if r['ev']=='commit':com[(rnd,r['fireTurn'])]=r
    if r['ev']=='fire':fires[(rnd,r['turn'])]=r
rnd=0;resolved=set()
for r in rows:
    if r['ev']=='round':rnd=r['round']
    if r['ev'] in('hitBy','blocked') and r.get('matched',-1)>0:
        k=(rnd,r['matched']);resolved.add(k)
        if r['ev']=='blocked':c['blocked']+=1;continue
        cm=com.get(k)
        e=abs(r.get('solverError',0))
        if not cm:c['hit:unshielded'+(':dodge' if r.get('dodge') else '')]+=1
        elif e>4:c['hit:aimerr>4 cov%d'%cm['covered'] if cm['covered']<5 else 'hit:aimerr>4 goodcov']+=1
        else:c['hit:inmodel cov%s'%('<5' if cm['covered']<5 else '>=5')]+=1
    if r['ev']=='hitBy' and r.get('matched',-1)<=0:c['hit:unmatched']+=1
c['other(miss/pending)']=len(fires)-len(resolved)
tot=sum(c.values());print(tot);[print(f'{k:28s}{v:5d} {v/tot:.3f}') for k,v in sorted(c.items(),key=lambda x:-x[1])]
