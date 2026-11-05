"""Bounded two-game top-four exhibition; championship history is read-only."""
import argparse
import gzip
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import tournament

def top_four(base):
    reports = {n: json.loads((base/f"playoff-{n}.json").read_text()) for n in (4,5,6)}
    if (base/"playoff-7.json").exists():
        reports[7] = json.loads((base/"playoff-7.json").read_text())
    upper = next(p for p in reports[4]["draw"] if p["matchId"] == "U7")["winner"]
    if 7 not in reports and reports[6]["draw"][0]["winner"] != upper:
        raise ValueError("Required championship reset is incomplete")
    final = reports.get(7, reports[6])
    if any(r["status"] != "complete" for r in reports.values()):
        raise ValueError("Championship must be complete")
    last = final["draw"][0]
    loser = lambda p: next(n for n in (p["a"],p["b"]) if n != p["winner"])
    lower = reports[5]["draw"][0]
    fourth = next(p for p in reports[4]["draw"] if p["matchId"] == "L5")
    names = [last["winner"],loser(last),loser(lower),loser(fourth)]
    if len(set(names)) != 4:
        raise ValueError("Top-four coverage differs")
    return names

def validate_result(data, identities):
    rows = data.get("results", [])
    if data.get("numberOfRounds") != 20 or len(rows) != 4:
        raise ValueError("Expected four official participants and twenty rounds")
    if sorted((r["name"],r["version"]) for r in rows) != sorted(identities):
        raise ValueError("Official identity mismatch")
    if len({r["id"] for r in rows}) != 4:
        raise ValueError("Repeated official ID")
    for row in rows:
        for key in ("totalScore","firstPlaces","bulletDamage","survival"):
            if type(row[key]) is not int or row[key] < 0:
                raise ValueError("Invalid official "+key)
        if row["firstPlaces"] > 20 or type(row["rank"]) is not int or not 1 <= row["rank"] <= 4:
            raise ValueError("Official rank/round coverage differs")

def audit_recording(run, output, official):
    recording = tournament.collect_recording(run/"recordings", output)
    starts = ends = 0
    rounds, tick_rounds, ids = [], set(), set()
    with gzip.open(output/recording["path"], "rt") as stream:
        for line in stream:
            event = json.loads(line)
            kind = event["type"]
            if kind == "GameStartedEventForObserver":
                starts += 1
                ids = {r["id"] for r in event["participants"]}
                if len(ids) != 4 or event["gameSetup"]["numberOfRounds"] != 20 or event["gameSetup"]["gameType"] != "melee":
                    raise ValueError("Observer setup differs")
                if sorted((r["name"],r["version"]) for r in event["participants"]) != sorted((r["name"],r["version"]) for r in official["results"]):
                    raise ValueError("Observer participant identities differ")
            elif kind == "TickEventForObserver":
                n = event["roundNumber"]
                present = {b["id"] for b in event["botStates"]}
                if not present <= ids or (n not in tick_rounds and present != ids):
                    raise ValueError("Observer bot coverage differs")
                tick_rounds.add(n)
            elif kind == "RoundEndedEventForObserver":
                rounds.append(event["roundNumber"])
            elif kind == "GameEndedEventForObserver":
                ends += 1
                observed = {r["id"]:r for r in event["results"]}
                for row in official["results"]:
                    if any(observed[row["id"]].get(k) != v for k,v in row.items()):
                        raise ValueError("Observer final scores differ")
    if starts != 1 or ends != 1 or rounds != list(range(1,21)) or tick_rounds != set(range(1,21)):
        raise ValueError("Incomplete observer game/round coverage")
    return recording

def totals(report):
    rows = {n:dict(name=n,totalScore=0,firstPlaces=0,games=0) for n in report["entrants"]}
    for game in report["games"]:
        if game["status"] == "complete":
            for r in game["official"]["results"]:
                row = rows[r["name"]]
                row["totalScore"] += r["totalScore"]
                row["firstPlaces"] += r["firstPlaces"]
                row["games"] += 1
    ordered = sorted(rows.values(), key=lambda r:(-r["totalScore"],-r["firstPlaces"],r["name"]))
    for i,row in enumerate(ordered):
        row["rank"] = 1+sum((r["totalScore"],r["firstPlaces"])>(row["totalScore"],row["firstPlaces"]) for r in ordered)
    return ordered

def main(args):
    source, base, output = args.source.resolve(),args.tournament.resolve(),args.output.resolve()
    names = top_four(base)
    sys.path.insert(1,str(source/"tools"))
    import benchmark, strong_melee_opponents
    java, python = args.java.resolve(),args.python.absolute()
    if 'version "25.' not in subprocess.check_output([str(java),"-version"],stderr=subprocess.STDOUT,text=True):
        raise ValueError("JDK25 required")
    bases = {n:source/"experiments"/n/"bot" for n in names if n != "Diamond Port"}
    bases["Diamond Port"] = source/"guest-source/bots/ports/diamond-port"
    expected = json.loads((base/"entry-manifests.json").read_text())
    for name in names:
        paths = list(bases[name].rglob("*"))
        if name in tournament.NAMES[1:8]:
            paths += list((source/"experiments"/name/"src").rglob("*"))+[source/"experiments"/name/"build.gradle",source/"experiments"/name/(name+".jar")]
        actual = {str(p.relative_to(source)):tournament.digest(p) for p in paths if p.is_file()}
        if actual != expected[name]:
            raise ValueError("Frozen entrant inputs changed: "+name)
    guest = source/"guest-source"
    for entry in strong_melee_opponents.manifest()["files"]:
        if tournament.digest(guest/entry["path"]) != entry["sha256"]:
            raise ValueError("Frozen guest/shared code changed")
    if tournament.collect_shared_inputs(source,java,python) != json.loads((base/"shared-inputs.json").read_text()):
        raise ValueError("Frozen shared runtime changed")
    identities = [tuple(json.loads(next(bases[n].glob("*.json")).read_text())[k] for k in ("name","version")) for n in names]
    runner = source/".benchmark-cache/robocode-tankroyale-runner-1.4.0.jar"
    if tournament.digest(runner) != benchmark.ASSETS[runner.name]:
        raise ValueError("Runner pin differs")
    harness = Path(__file__).with_name("TournamentBattle.java").resolve()
    if tournament.recording_harness_inputs(harness) != json.loads((base/"recording-harness.json").read_text()):
        raise ValueError("Recording adapter changed")
    inputs = dict(importedTournamentSHA256=tournament.digest(Path(tournament.__file__)),runtimeHelpersSHA256={name:tournament.digest(source/"tools"/name) for name in ("benchmark.py","strong_melee_opponents.py")},scriptSHA256=tournament.digest(Path(__file__)),harnessSHA256=tournament.digest(harness),entryManifestSHA256=tournament.digest(base/"entry-manifests.json"),sharedInputsSHA256=tournament.digest(base/"shared-inputs.json"),championshipSHA256={p.name:tournament.digest(p) for p in base.glob("playoff-*.json") if p.stem in {f"playoff-{i}" for i in range(1,8)}})
    output.mkdir(parents=True,exist_ok=True)
    state = output/"report.json"
    if state.exists():
        if not args.resume_unplayed:
            raise ValueError("Existing exhibition; explicit unplayed continuation required")
        report = json.loads(state.read_text())
        if report["inputs"] != inputs or report["entrants"] != names or any(g["status"] != "complete" for g in report["games"]):
            raise ValueError("Cannot resume changed inputs or an attempted incomplete game")
        if len(report["games"]) >= 2:
            raise ValueError("Exhibition already completed")
    else:
        report = dict(kind="post-tournament-melee",status="scheduled",entrants=names,plannedGames=2,roundsPerGame=20,completedGames=0,completedRounds=0,started=tournament.stamp(),games=[],inputs=inputs,scoring="Sum official scores; then first places; exact ties shared; no extra games",championshipUnchanged=True)
        tournament.save(state,report)
    if args.register_only:
        print("REGISTERED "+", ".join(names)+"; no games launched",flush=True)
        return
    for game in (1,2):
        if any(g["game"]==game for g in report["games"]):
            continue
        if shutil.disk_usage(output).free < 2*1024**3:
            raise ValueError("2GiB launch guard failed; no new game started")
        run = output/f"g{game}"
        run.mkdir()
        participants = names if game==1 else list(reversed(names))
        env = {k:v for k,v in os.environ.items() if not k.startswith(tournament.PREFIXES) and k not in ("RETCON_STATS_DIR","RUMBLE_JAVA_CACHE","JAVA_TOOL_OPTIONS","JDK_JAVA_OPTIONS","JAVA_HOME","SERVER_URL","BOT_SECRET","SERVER_SECRET")}
        env.update(JAVA_HOME=str(java.parent.parent),PATH=str(java.parent)+os.pathsep+env.get("PATH",""),ROBOCODE_PYTHON_BIN=str(python),PYTHONNOUSERSITE="1",PYTHONDONTWRITEBYTECODE="1",ROBOCODE_ENV_FILE=str(run/"absent.env"),ROBOCODE_GUN_ENV_FILE=str(run/"absent-guns.env"))
        dirs=[]
        for name in participants:
            target=run/"bots"/(name if name in tournament.NAMES[:8] else bases[name].name)
            benchmark.isolate_opponent(bases[name],target)
            if name in tournament.NAMES[1:8]:
                shutil.copy2(source/"experiments"/name/(name+".jar"),target/(name+".jar"))
            for sh in target.glob("*.sh"):sh.chmod(0o755)
            dirs.append(target)
        shutil.copytree(guest/"bots/bot_core",run/"bots/bot_core")
        shutil.copy2(guest/"LICENSE",run/"bots/LICENSE.shared-Apache-2.0")
        for prefix in ("KEEL","KEEL_HELM","PRISM"):env[prefix+"_SEED"]="20261104"
        for name,key in (("Keel","KEEL_METRICS"),("KeelHelm","KEEL_HELM_METRICS"),("Prism","PRISM_TELEMETRY_PATH")):
            env[key]=str(run/(name+"-metrics.jsonl"))
        (run/"recordings").mkdir()
        command=[str(java),"-Dtournament.recordingDir="+str(run/"recordings"),"-cp",str(runner),str(harness),"20","900",str(run/"battle.json"),*map(str,dirs)]
        row=dict(game=game,status="running",participants=participants,directory=run.name,started=tournament.stamp())
        report["games"].append(row);report["status"]="running";tournament.save(state,report)
        tournament.save(run/"invocation.json",dict(command=command,configuredEnvironment={k:v for k,v in env.items() if k.startswith(tournament.PREFIXES) or k=="JAVA_HOME"}))
        print(f"START melee game{game}: {participants}",flush=True);start=time.monotonic()
        try:
            benchmark.run_process(command,source,env,run/"runner.log",960)
            official=json.loads((run/"battle.json").read_text());validate_result(official,identities)
            recording=audit_recording(run,output,official)
            row.update(status="complete",official=official,recording=recording)
            report["completedGames"]+=1;report["completedRounds"]+=20
        except Exception as error:
            row.update(status="failed",error=str(error));report["status"]="incomplete"
            if (run/"battle.json").exists():row["unvalidatedOfficial"]=json.loads((run/"battle.json").read_text())
        row.update(elapsedSeconds=round(time.monotonic()-start,3),finished=tournament.stamp())
        report["standings"]=totals(report);tournament.save(state,report)
        print(f"END melee game{game}: {row['status']}",flush=True)
        if row["status"]!="complete":return
    report.update(status="complete",finished=tournament.stamp());tournament.save(state,report)
    print("EXHIBITION COMPLETE; championship unchanged",flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,required=True);p.add_argument("--tournament",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--java",type=Path,required=True);p.add_argument("--python",type=Path,required=True)
    p.add_argument("--register-only",action="store_true");p.add_argument("--resume-unplayed",action="store_true")
    main(p.parse_args())
