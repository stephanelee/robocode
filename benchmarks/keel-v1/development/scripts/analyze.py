import gzip,json,sys
from pathlib import Path
def analyze(d,name):
    rec=list(Path(d).glob('*.battle.gz'))[0]
    ev=[json.loads(l) for l in gzip.open(rec,'rt')]
    end=ev[-1]; res=end['results']; me=next(r for r in res if r['name']==name); uid=me['id']
    o=dict(shots=0,hits=0,dealt=0.,taken=0.,eshots=0,ehits=0,walls=0,rams=0)
    rounds=[]; cur={}
    for t in ev:
        if t['type']=='RoundEndedEventForObserver':
            rr=t['results']; m=next(r for r in rr if r['id']==uid); oth=next(r for r in rr if r['id']!=uid)
            rounds.append(round(m['totalScore']/max(1,m['totalScore']+oth['totalScore'])*100))
        if t['type']!='TickEventForObserver': continue
        for e in t['events']:
            k=e['type'];b=e.get('bullet',{})
            if k=='BulletFiredEvent': 
                if b['ownerId']==uid: o['shots']+=1
                else: o['eshots']+=1
            elif k=='BulletHitBotEvent':
                if b['ownerId']==uid: o['hits']+=1;o['dealt']+=e['damage']
                elif e['victimId']==uid: o['ehits']+=1;o['taken']+=e['damage']
            elif k=='BotHitWallEvent' and e['victimId']==uid:o['walls']+=1
            elif k=='BotHitBotEvent' and e['botId']==uid:o['rams']+=1
    share=me['totalScore']/sum(r['totalScore'] for r in res)
    print(f"share={share*100:.1f}% wins={me['firstPlaces']}/{len(rounds)} hit={o['hits']}/{o['shots']}={o['hits']/max(1,o['shots'])*100:.1f}% ehit={o['ehits']}/{o['eshots']}={o['ehits']/max(1,o['eshots'])*100:.1f}% dealt={o['dealt']:.0f} taken={o['taken']:.0f} walls={o['walls']} rams={o['rams']} rounds%={rounds}")
analyze(sys.argv[1],sys.argv[2])
