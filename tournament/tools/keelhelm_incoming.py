"""Attribute observed incoming contacts to Crossfire's preceding state."""
from collections import Counter, defaultdict
import gzip
import json
import math
from pathlib import Path
import sys
from melee_damage import AUDIT, HELM, phase, sha

def bucket(prior, own, w, h):
    s = prior[own]
    wall = min(s["x"]-18,s["y"]-18,w-18-s["x"],h-18-s["y"])
    near = min((math.hypot(b["x"]-s["x"],b["y"]-s["y"]) for i,b in prior.items() if i!=own),default=math.inf)
    return [("speed","0–<4" if abs(s["speed"])<4 else "4–<8" if abs(s["speed"])<8 else "8"),
            ("wall clearance","<80" if wall<80 else "80+"),
            ("nearest opponent","<250" if near<250 else "250–<500" if near<500 else "500+"),
            ("absolute turn","<2" if abs(s["turnRate"])<2 else "2+")]

def measure(path):
    out=defaultdict(Counter)
    shots={}
    struck=set()
    prior={}
    for line in gzip.open(path,"rt"):
        m=json.loads(line)
        if m["type"]=="GameStartedEventForObserver":
            names={b["id"]:b["name"] for b in m["participants"]}
            own=next(i for i,n in names.items() if n==HELM)
            w,h=(m["gameSetup"][k] for k in ("arenaWidth","arenaHeight"))
        if m["type"]!="TickEventForObserver":continue
        r,t=m["roundNumber"],m["turnNumber"]
        if t==1:prior={}
        context=[]
        hitphase="unknown"
        if own in prior:
            hitphase=phase(prior[own]["enemyCount"])
            context=[hitphase]+[f"{hitphase} / {k} / {v}" for k,v in bucket(prior,own,w,h)]
            for key in context:out[key]["atRiskTicks"]+=1
        ram_pairs=set()
        for e in m["events"]:
            kind=e["type"]
            if kind=="BulletFiredEvent":
                b=e["bullet"]; shooter=b["ownerId"]; ps=prior.get(shooter)
                firephase=phase(ps["enemyCount"]) if ps else "unknown"
                shots[r,b["bulletId"]]=(firephase, math.hypot(b["x"]-prior[own]["x"],b["y"]-prior[own]["y"]) if own in prior else None)
            elif kind=="BulletHitBotEvent" and e["victimId"]==own:
                b=e["bullet"]; sid=b["ownerId"]; key=r,b["bulletId"]
                firephase,dist=shots[key]
                extra=[f"{hitphase} / shooter / {names[sid]}",f"fire {firephase} / impact {hitphase}",
                       f"{hitphase} / firing distance / "+("unknown" if dist is None else "<250" if dist<250 else "250–<500" if dist<500 else "500+")]
                d=e["damage"]; credit=max(0,min(d,e["energy"]+d))
                for k in (context or ["unknown"])+extra:
                    out[k].update(hitEvents=1,uniqueHitBullets=key not in struck,rawDamage=d,credit=credit,
                                  lethal=(e["energy"]<0 and e["energy"]+d>=0))
                struck.add(key)
            elif kind=="BotHitWallEvent" and e.get("victimId")==own:
                out[hitphase]["wallContacts"]+=1
            elif kind=="BotHitBotEvent" and own in (e.get("botId"),e.get("victimId")):
                out[hitphase]["ramEventRecords"]+=1
                ram_pairs.add(tuple(sorted((e["botId"],e["victimId"]))))
        if ram_pairs:out[hitphase]["ramContacts"]+=len(ram_pairs)
        dead={e["victimId"] for e in m["events"] if e["type"]=="BotDeathEvent"}
        prior={s["id"]:s for s in m["botStates"] if s["id"] not in dead}
    return out

def main(evidence):
    games={}
    aggregate=defaultdict(Counter)
    for g in json.loads(AUDIT.read_text())["games"]:
        if not g["game"].endswith("-candidate"):continue
        rec,=(evidence/"games"/g["game"]/"recordings").glob("*.battle.gz")
        assert sha(rec)==g["recordingSHA256"]
        data=measure(rec)
        games[g["game"]]=data
        for k,v in data.items():aggregate[k].update(v)
    assert len(games)==12
    print(json.dumps(dict(games=games,aggregate=aggregate),indent=2,sort_keys=True))
if __name__=="__main__":
    main(Path(sys.argv[1]))
