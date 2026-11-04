import gzip,json,sys,math
from pathlib import Path
from collections import defaultdict
def norm(a): return math.remainder(a,360)
def run(d,name='Keel'):
    rec=list(Path(d).glob('*.battle.gz'))[0]
    ev=[json.loads(l) for l in gzip.open(rec,'rt')]
    uid=next(r for r in ev[-1]['results'] if r['name']==name)['id']
    fired=[];passed=[];hitg=[]
    for rnd in sorted({t['roundNumber'] for t in ev if t['type']=='TickEventForObserver'}):
        ticks={t['turnNumber']:t for t in ev if t['type']=='TickEventForObserver' and t['roundNumber']==rnd}
        pos={T:next(b for b in t['botStates'] if b['id']==uid) for T,t in ticks.items() if any(b['id']==uid for b in t['botStates'])}
        waves=[]
        for T in sorted(ticks):
            for e in ticks[T]['events']:
                if e['type']=='BulletFiredEvent' and e['bullet']['ownerId']!=uid:
                    b=e['bullet']; p=pos.get(T-2) or pos.get(T-1)
                    if not p: continue
                    # origin: shooter position at T-1
                    sh=next((x for x in ticks.get(T-1,{'botStates':[]})['botStates'] if x['id']!=uid),None)
                    if not sh: continue
                    ox,oy=sh['x'],sh['y']; direct=math.degrees(math.atan2(p['y']-oy,p['x']-ox))
                    lat=p['speed']*math.sin(math.radians(p['direction']-direct)); ld=1 if lat>=0 else -1
                    s=20-3*b['power']; mea=math.degrees(math.asin(8/s))
                    g=norm(b['direction']-direct)/mea*ld
                    fired.append(g); waves.append([T,ox,oy,s,direct,ld,mea,g,b['bulletId']])
                if e['type']=='BulletHitBotEvent' and e.get('victimId')==uid:
                    w=next((w for w in waves if w[8]==e['bullet']['bulletId']),None) if 'bulletId' in e['bullet'] else None
                    if w: hitg.append(w[7])
            p=pos.get(T)
            if p:
                for w in list(waves):
                    r=(T-w[0]+1)*w[3]
                    if r>=math.hypot(p['x']-w[1],p['y']-w[2]):
                        passed.append(norm(math.degrees(math.atan2(p['y']-w[2],p['x']-w[1]))-w[4])/w[6]*w[5]); waves.remove(w)
    def hist(v):
        h=[0]*11
        for g in v: h[min(10,max(0,int(round((g+1.25)/2.5*10))))]+=1
        return h
    print(d,'n',len(fired),len(passed),len(hitg))
    print(' fired ',hist(fired)); print(' passed',hist(passed)); print(' hits  ',hist(hitg))
for d in sys.argv[1:]: run(d)
