import gzip,json,sys
from pathlib import Path
rec=list(Path(sys.argv[1]).glob('*.battle.gz'))[0]
ev=[json.loads(l) for l in gzip.open(rec,'rt')]
res=ev[-1]['results']; uid=next(r for r in res if r['name']==sys.argv[2])['id']
st={}
for t in ev:
    if t['type']=='TickEventForObserver':
        r=t['roundNumber']; s=st.setdefault(r,dict(turns=0,sh=0,h=0,d=0.,es=0,eh=0,et=0.,myE=None,eE=None))
        s['turns']=t['turnNumber']
        for b in t['botStates']:
            if b['id']==uid: s['myE']=round(b['energy'],1)
            else: s['eE']=round(b['energy'],1)
        for e in t['events']:
            k=e['type'];b=e.get('bullet',{})
            if k=='BulletFiredEvent':
                if b['ownerId']==uid: s['sh']+=1
                else: s['es']+=1
            elif k=='BulletHitBotEvent':
                if b['ownerId']==uid: s['h']+=1;s['d']+=e['damage']
                else: s['eh']+=1;s['et']+=e['damage']
for r,s in st.items(): print(r,s)
