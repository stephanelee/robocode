import json,math,sys,statistics
from collections import Counter,defaultdict
D=[json.loads(l) for l in open(sys.argv[1]+'/diamond.jsonl')]
V=[v for v in D if v['ev']=='vg']
hits=[v['hit'] for v in V if v['gun'].startswith('Main')]
print('firing hit mean',round(statistics.mean(hits),4),'abs',round(statistics.mean(abs(h) for h in hits),4),'n',len(hits))
dis=Counter()
for a,b in zip(V,V[1:]):
    if a['ft']==b['ft'] and a['gun']!=b['gun']:
        m,s=(a,b) if a['gun'].startswith('Main') else (b,a)
        if round(m['fa']/0.0207)!=round(s['fa']/0.0207):dis['Main' if m['score']>s['score'] else 'Anti']+=1
print('disagreements won by',dict(dis))
rounds=[];cur=[];last=-1
for r in D:
    if r['ev'] not in('aim','break'):continue
    if r['t']<last-50:rounds.append(cur);cur=[]
    last=r['t'];cur.append(r)
rounds.append(cur)
for i,rr in enumerate(rounds,1):
    aims=[r for r in rr if r['ev']=='aim']
    c=Counter(max(-2,min(2,round(((r['fa']-r['absb']+math.pi)%(2*math.pi)-math.pi)*r['od']/0.0207))) for r in aims)
    print(i,'bin0',round(c[0]/max(1,len(aims)),3),dict(Counter(r['gun'][:4] for r in aims)))
