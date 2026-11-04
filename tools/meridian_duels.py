"""Execute the manifest once in a fresh workspace. Summaries use official observer events."""
import gzip
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import benchmark
import retcon
import strong_melee_opponents as native
from meridian_profile import CACHE, OUT, ROOT

REG=ROOT/"benchmarks/meridian-v1"
def save(path,obj):
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+"\n")

def metrics(path,game):
    recordings=list(path.glob("*.battle.gz"))
    if len(recordings)!=1: raise ValueError("Expected one raw recording")
    with gzip.open(recordings[0],"rt") as fd: events=[json.loads(line) for line in fd]
    end=events[-1]
    if end["type"]!="GameEndedEventForObserver" or end["numberOfRounds"]!=game["rounds"]: raise ValueError("Missing game endpoint")
    setup=events[0]["gameSetup"]
    if setup["gameType"]!="1v1" or setup["numberOfRounds"]!=game["rounds"] or setup["turnTimeout"]!=100000: raise ValueError("Wrong setup")
    results=end["results"]
    if sorted((r["name"],r["version"]) for r in results)!=sorted([("Meridian",game.get("meridianVersion","0.1.0")),(game["enemy"],game["version"])]): raise ValueError("Wrong participants")
    me=next(r for r in results if r["name"]=="Meridian"); uid=me["id"]
    markers=(path/"markers.txt").read_text().splitlines()
    rounds=list(range(1,game["rounds"]+1))
    for tag in ("start","end"):
        if [int(m.split()[0]) for m in markers if m.endswith(" "+tag)]!=rounds: raise ValueError("Incomplete private round markers")
    endpoints=[e for e in events if e["type"]=="RoundEndedEventForObserver"]
    if [e["roundNumber"] for e in endpoints]!=rounds: raise ValueError("Incomplete official rounds")
    shots=hits=spent=dealt=taken=walls=rams=0
    enemy_shots=hits_taken=0
    turns={r:[] for r in rounds}
    for tick in events:
        if tick["type"]!="TickEventForObserver": continue
        turns[tick["roundNumber"]].append(tick["turnNumber"])
        for e in tick["events"]:
            kind=e["type"]; b=e.get("bullet",{})
            if kind=="BulletFiredEvent" and b["ownerId"]==uid:
                shots+=1;spent+=b["power"]
            elif kind=="BulletFiredEvent":
                enemy_shots+=1
            elif kind=="BulletHitBotEvent":
                if b["ownerId"]==uid: hits+=1;dealt+=e["damage"]
                elif e["victimId"]==uid:
                    taken+=e["damage"]; hits_taken+=1
            elif kind=="BotHitWallEvent" and e["victimId"]==uid: walls+=1
            elif kind=="BotHitBotEvent" and e["botId"]==uid:rams+=1
    for r,seq in turns.items():
        if not seq or seq!=list(range(seq[0],seq[-1]+1)) or seq[0]!=1 or seq[-1]!=endpoints[r-1]["turnNumber"]: raise ValueError("Missing/duplicate official ticks")
    if game["enemy"]=="AISurvivor":
        ais=[json.loads(x) for x in (path/"ais.jsonl").read_text().splitlines()]
        if [x["round"] for x in ais]!=rounds or any(x["mode"]!="train" or x["seed"]!=game["seed"] or x["checkpointLoaded"] is not False for x in ais) or int.from_bytes((path/"ais-model.dat").read_bytes()[20:24],"big")!=game["rounds"]: raise ValueError("AIS cold model or round evidence missing")
    extended = {}
    if "meridianVersion" in game:
        if not enemy_shots:
            raise ValueError("Undefined incoming damage rate")
        extended = dict(enemyShots=enemy_shots, hitsTaken=hits_taken, damagePer100EnemyShots=100*taken/enemy_shots)
        if game["meridianVersion"] in ("0.2.0", "0.3.0", "0.4.0", "0.5.0"):
            labels = {label: 0 for label in ("ACCEPTED", "AMBIGUOUS", "UNMATCHED", "UNORIENTED", "OUTSIDE")}
            for line in markers:
                fields = line.split()
                if len(fields) == 3 and fields[1] == "hit":
                    labels[fields[2]] += 1
            extended["hitLabels"] = labels
            extended["observerHitsWithoutCallback"] = hits_taken - sum(labels.values())
        if game["meridianVersion"] in ("0.3.0", "0.4.0", "0.5.0"):
            extended["learnedFireRequests"] = sum(line.endswith(" aim learned") for line in markers)
            extended["fallbackFireRequests"] = sum(line.endswith(" aim fallback") for line in markers)
            extended["crossingLabels"] = sum(int(line.split()[2]) for line in markers if " crossings " in line)
        if game["meridianVersion"] == "0.6.0":
            extended["priorFireRequests"] = sum(line.endswith(" aim prior") for line in markers)
        if game["meridianVersion"] == "0.5.0":
            warnings = {(int(r), int(t)) for r, tag, t in (m.split() for m in markers if " warn " in m)}
            observed = {(tick["roundNumber"], tick["turnNumber"]) for tick in events
                        if tick["type"] == "TickEventForObserver"
                        and any(e["type"] == "BulletFiredEvent" and e["bullet"]["ownerId"] != uid
                                for e in tick["events"])}
            choices = [line.split() for line in markers if " surf " in line]
            if [int(fields[0]) for fields in choices] != rounds:
                raise ValueError("Missing surf counters")
            extended.update(warnings=len(warnings), warnedShots=len(warnings & observed),
                            stopTurns=sum(int(c[2]) for c in choices), twoFrontTurns=sum(int(c[3]) for c in choices))
    return dict(**extended,share=me["totalScore"]/sum(r["totalScore"] for r in results),wins=me["firstPlaces"],shots=shots,hits=hits,
                efficiency=hits/shots if shots else None,damageDealt=dealt,damageTaken=taken,shotEnergy=spent,wallContacts=walls,
                botContacts=rams,skips=sum(" skip " in m for m in markers),results=results,
                recordingSHA256=benchmark.digest(recordings[0]),turnsByRound=[len(s) for s in turns.values()])

def play(game,profile,*,bundle=ROOT/"experiments/Meridian"):
    data=OUT/game["id"];data.mkdir()
    stage=ROOT/"build/meridian-staging"/game["id"];stage.mkdir(parents=True)
    own=stage/"Meridian";shutil.copytree(bundle/"bot",own)
    shutil.copy2(bundle/"Meridian.jar",own/"Meridian.jar")
    env={k:os.environ[k] for k in ("PATH","HOME","JAVA_HOME","LANG") if k in os.environ}
    env.update(MERIDIAN_SEED=str(game["seed"]),MERIDIAN_METRICS=str(data/"markers.txt"),PYTHONDONTWRITEBYTECODE="1")
    enemy=game["enemy"]
    if enemy=="Retcon":
        src,api,rc=retcon.prepare(CACHE);other=stage/"Retcon";benchmark.isolate_opponent(src,other)
        if retcon.source_hash(other)!=rc["sourceSHA256"]:raise ValueError("Retcon tree mismatch")
        env["RUMBLE_JAVA_CACHE"]=str(api)
    elif enemy=="Diamond Port":
        base=CACHE/"strong-python"/profile["python"]["commit"];src=base/"source"
        env.update(native.stage(src,stage,base/"venv/bin/python"))
        other=stage/"opponents/diamond-port";benchmark.isolate_opponent(src/"bots/ports/diamond-port",other)
        native.verify_staged(stage,["diamond-port"])
    elif enemy == "Walls":
        name = "sample-bots-Java-1.4.0.zip"
        archive = benchmark.download(CACHE, name, profile["sampleArchive"])
        samples = stage / "official-samples"
        benchmark.extract(archive, samples)
        descriptor, = samples.rglob("Walls.json")
        other = stage / "Walls"
        benchmark.isolate_opponent(descriptor.parent, other)
    elif enemy=="AISurvivor":
        other=stage/"AISurvivor";shutil.copytree(ROOT/"AISurvivor",other)
        env.update(AISURVIVOR_MODEL_PATH=str(data/"ais-model.dat"),AISURVIVOR_METRICS_PATH=str(data/"ais.jsonl"),
                   AISURVIVOR_SEED=str(game["seed"]),AISURVIVOR_MODE="train")
    else: raise ValueError("Unregistered opponent")
    command=["java","-cp",str(CACHE/"robocode-tankroyale-runner-1.4.0.jar"),str(REG/"MeridianBattle.java"),
             str(game["port"]),str(game["rounds"]),str(data),str(own),str(other)]
    save(data/"attempt.json",dict(game=game,command=command,endpoint="ws://localhost:"+str(game["port"]),execution=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()))
    benchmark.run_process(command,stage,env,data/"runner.log",1680)
    return metrics(data,game)

def main():
    profile=json.loads((OUT/"profile.json").read_bytes())
    if (OUT/"profile.json").read_bytes()!=(REG/"profile.json").read_bytes():raise ValueError("Wrong runtime")
    plan=json.loads((REG/"games.json").read_bytes())
    report=dict(games=[],decision="pending"); dest=OUT/"decision.json"
    if dest.exists():raise FileExistsError(dest)
    save(dest,report)
    for stage in ("screen","confirmation"):
        rows=[]
        for game in [g for g in plan if g["stage"]==stage]:
            try: row=dict(**game,status="valid",**play(game,profile))
            except Exception as e: row=dict(**game,status="invalid",error=str(e))
            report["games"].append(row);rows.append(row);save(dest,report)
        valid=all(r["status"]=="valid" and r["skips"]==0 for r in rows)
        if stage=="screen":
            if not valid or not all(r["share"]>.35 and r["wins"]>=1 for r in rows):
                report["decision"]="stop at screen";break
        else:
            for enemy in {g["enemy"] for g in plan}:
                group=[r for r in rows if r["enemy"]==enemy]
                valid=valid and statistics.mean(r["share"] for r in group)>.50 and sum(r["wins"] for r in group)>30 and sum(r["share"]>.50 for r in group)>=2
            report["decision"]="confirmed" if valid else "reject"
    save(dest,report)
if __name__=="__main__":main()
