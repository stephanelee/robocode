"""Prepare and run the registered serial firepower benchmark using the reviewed paired transport."""
import argparse
import gzip
import json
import math
import os
from pathlib import Path
import random
import shutil
import time
import zipfile

import keelhelm_risk as risk
from keelhelm_crossfire import SERVER_SHA

core = risk.core
ROOT = risk.ROOT
PROJECT = ROOT / "benchmarks/keelhelm-long-range-power/candidate-project"

def schedule():
    rng = random.Random(145202611)
    games = []
    for scenario, count, rounds in [("melee",4,5), ("Diamond Port",2,5), ("Retcon",1,2), ("Walls",1,2)]:
        names = risk.NAMES if scenario == "melee" else ["KeelHelm",scenario]
        w,h = (1000,1000) if scenario == "melee" else (800,600)
        for layout in range(count):
            while True:
                points = [[rng.randint(80,w-80),rng.randint(80,h-80),rng.randrange(360)] for _ in names]
                if all(math.dist(a[:2],b[:2]) >= 250 for i,a in enumerate(points) for b in points[:i]):
                    break
            pair = len(games)//2
            for arm in (("control","candidate") if pair%2 == 0 else ("candidate","control")):
                seat = layout%len(names)
                games.append(dict(id=f"power145-{pair+1:02d}-{arm}",arm=arm,pair=pair,mode="fixed",
                                  scenario=scenario,rounds=rounds,seed=145202611+pair,
                                  order=names[seat:]+names[:seat],positions=dict(zip(names,points))))
    return games

def prepare(a):
    out = a.evidence.resolve()
    out.mkdir(parents=True,exist_ok=False)
    parent = ROOT/"experiments/KeelHelm"
    shutil.copytree(parent,out/"control-source",ignore=shutil.ignore_patterns("build",".gradle","__pycache__"))
    jar = PROJECT/"KeelHelm.jar"
    if jar.read_bytes() != (PROJECT/"build/bot/KeelHelm/KeelHelm.jar").read_bytes():
        raise ValueError("Candidate build parity failed")
    with zipfile.ZipFile(a.source/".benchmark-cache/robocode-tankroyale-runner-1.4.0.jar") as runner:
        (out/"server.jar").write_bytes(runner.read("robocode-tankroyale-server.jar"))
    if core.sha(out/"server.jar") != SERVER_SHA: raise ValueError("Server hash mismatch")
    extras = [Path(__file__),out/"server.jar",ROOT/"docs/improvements/keelhelm-long-range-power-protocol.md",
              ROOT/"tournament/scripts/PairedBattle.java", ROOT/"tools/keelhelm_resources.py"]
    extras += [ROOT/"tournament/tools"/p for p in
               ("paired_transport.py","keelhelm_risk.py","keelhelm_crossfire.py","keelhelm_calibration.py","keelhelm_randomization.py")]
    f = core.collect(a.source.resolve(),out,a.pins.resolve(),a.java.absolute(),a.python.absolute(),
                     control_jar_sha=risk.BASE,schedule_rows=schedule(),extra_paths=extras,candidate_project=PROJECT)
    f.update(serverJar=str(out/"server.jar"),candidateSHA256=core.sha(jar),controlSHA256=risk.BASE)
    for game in f["schedule"]:
        run = out/"preflight"/game["id"]
        dirs,ids = risk.stage(game,f,out,run)
        version = "0.2.1-power145" if game["arm"]=="candidate" else "0.2.1"
        assert ("KeelHelm",version) in ids and len(dirs)==len(game["order"])
        expected = f["candidateSHA256"] if game["arm"]=="candidate" else risk.BASE
        assert core.sha(run/"bots/KeelHelm/KeelHelm.jar")==expected
    core.save(out/"freeze.json",f)
    print(core.sha(out/"freeze.json"))

def phase_damage(recording,helm):
    total = melee = 0.
    previous = None
    origins = {}
    with gzip.open(recording,"rt") as stream:
        for line in stream:
            msg=json.loads(line)
            if msg["type"]!="TickEventForObserver": continue
            r,t=msg["roundNumber"],msg["turnNumber"]
            for e in msg["events"]:
                b=e.get("bullet")
                if not b or b["ownerId"]!=helm["id"]:continue
                key=(r,b["bulletId"])
                if e["type"]=="BulletFiredEvent":
                    assert previous and previous[:2]==(r,t-1)
                    origins[key]=previous[2]>1
                if e["type"]=="BulletHitBotEvent":
                    credit=max(0,min(e["damage"],e["energy"]+e["damage"]))
                    total+=credit
                    if origins[key]:melee+=credit
            own=next((b for b in msg["botStates"] if b["id"]==helm["id"]),None)
            previous=(r,t,own["enemyCount"]) if own else None
    if math.floor(total+0.5)!=helm["bulletDamage"]: raise ValueError("Official damage disagrees")
    return melee

def run(a):
    out=a.evidence.resolve()
    if core.sha(out/"freeze.json")!=a.approved_sha: raise ValueError("Unreviewed freeze")
    f=json.loads((out/"freeze.json").read_text())
    if f["schedule"]!=schedule() or f["evidence"]!=str(out): raise ValueError("Wrong run")
    core.verify_inputs(f)
    with (out/"acquisition-receipt.json").open("x") as receipt:
        json.dump(dict(freezeSHA256=a.approved_sha,executionHead=f["parent"]),receipt)
        receipt.flush();os.fsync(receipt.fileno())
    report=dict(status="running",games=[],freezeSHA256=a.approved_sha)
    try:
        for game in f["schedule"]:
            core.verify_inputs(f)
            while True:
                ad=risk.resources.read_resources()
                ad["diskFreeBytes"]=shutil.disk_usage(out).free
                core.save(out/(game["id"]+"-admission.json"),ad)
                if ad["diskFreeBytes"]<2*1024**3: raise ValueError("Insufficient disk")
                if ad["availablePercent"]>=35: break
                time.sleep(15)
            monitor=risk.resources.Monitor(out/(game["id"]+"-resources.jsonl"))
            monitor.start()
            try: row=risk.play(game,f,out,monitor,risk_metrics=False)
            finally: monitor.close()
            report["games"].append(row)
            row["resourceSamples"]=monitor.samples
            if monitor.error or row["status"]!="complete": raise ValueError(monitor.error or row.get("error"))
            row["meleePhaseDamage"]=phase_damage(out/row["recording"],row["helm"])
            core.save(out/"results.json",report)
        report["status"]="complete"
    except BaseException as error:
        report.update(status="invalid",error=repr(error));raise
    finally:core.save(out/"results.json",report)

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=["prepare","run"])
    p.add_argument("--evidence",type=Path,required=True)
    for name in ("source","pins","java","python"):p.add_argument("--"+name,type=Path)
    p.add_argument("--approved-sha")
    args=p.parse_args()
    (prepare if args.action=="prepare" else run)(args)
