"""Registered Keel v1 experiment (docs/improvements/keel-v1.md).

  python tools/keel_experiment.py profile        # zero games: acquire and hash the exact runtime
  python tools/keel_experiment.py run STAGE      # play every registered game of STAGE once, in order

Scores and combat counts come only from the official observer recording. Never reruns or replaces a game.
"""
import gzip
import hashlib
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

import benchmark
import retcon
import strong_melee_opponents as native

ROOT = benchmark.ROOT
CACHE = ROOT / ".benchmark-cache"
OUT = ROOT / "build/keel"
REG = ROOT / "benchmarks/keel-v1"
RUNNER = "robocode-tankroyale-runner-1.4.0.jar"
BOTS = {"Keel": ROOT / "experiments/Keel", "Meridian": REG / "control/Meridian"}


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def tree_hashes(directory):
    return {p.relative_to(ROOT).as_posix(): benchmark.digest(p) for p in sorted(directory.rglob("*")) if p.is_file()}


def profile():
    OUT.mkdir(parents=True, exist_ok=True)
    source, python, py = native.prepare(CACHE, sys.executable)
    _, _, rc = retcon.prepare(CACHE)
    jar = benchmark.download(CACHE, RUNNER, benchmark.ASSETS[RUNNER])
    java = Path(os.environ["JAVA_HOME"])
    files = {}
    for d in (ROOT / "experiments/Keel/src", ROOT / "experiments/Keel/bot", REG / "control", ROOT / "AISurvivor"):
        files.update(tree_hashes(d))
    for f in (ROOT / "experiments/Keel/Keel.jar", REG / "KeelBattle.java", REG / "games.json", Path(__file__),
              ROOT / "tools/opponents/strong-melee.lock.json", ROOT / "tools/opponents/retcon.lock.json"):
        files[f.relative_to(ROOT).as_posix()] = benchmark.digest(f)
    data = dict(assets=files, runner=benchmark.digest(jar),
                java25={f: benchmark.digest(java / f) for f in ("bin/java", "lib/modules", "release")},
                javaVersion=subprocess.run([str(java / "bin/java"), "-version"], capture_output=True, text=True).stderr,
                buildJava=(OUT / "build-java.txt").read_text() if (OUT / "build-java.txt").exists() else None,
                os=Path("/etc/os-release").read_text(), arch=platform.machine(), image=os.environ.get("ImageVersion"),
                pythonBinary=benchmark.digest(python.resolve()), python=py, retcon=rc)
    save(OUT / "profile.json", data)
    print("Profile:", hashlib.sha256((OUT / "profile.json").read_bytes()).hexdigest(), flush=True)


def metrics(path, game, name, version):
    recordings = list(path.glob("*.battle.gz"))
    if len(recordings) != 1:
        raise ValueError("Expected one raw recording")
    with gzip.open(recordings[0], "rt") as fd:
        events = [json.loads(line) for line in fd]
    end = events[-1]
    if end["type"] != "GameEndedEventForObserver" or end["numberOfRounds"] != game["rounds"]:
        raise ValueError("Missing game endpoint")
    setup = events[0]["gameSetup"]
    if setup["gameType"] != "1v1" or setup["numberOfRounds"] != game["rounds"] or setup["turnTimeout"] != 100000:
        raise ValueError("Wrong setup")
    results = end["results"]
    expected = sorted([(name, version), (game["enemy"], game["enemyVersion"])])
    if sorted((r["name"], r["version"]) for r in results) != expected:
        raise ValueError("Wrong participants")
    me = next(r for r in results if r["name"] == name)
    uid = me["id"]
    rounds = list(range(1, game["rounds"] + 1))
    markers = (path / "markers.txt").read_text().splitlines()
    for tag in ("start", "end"):
        if [int(m.split()[0]) for m in markers if m.endswith(" " + tag)] != rounds:
            raise ValueError("Incomplete private round markers")
    endpoints = [e for e in events if e["type"] == "RoundEndedEventForObserver"]
    if [e["roundNumber"] for e in endpoints] != rounds:
        raise ValueError("Incomplete official rounds")
    c = dict(shots=0, hits=0, shotEnergy=0.0, damageDealt=0.0, enemyShots=0, enemyHits=0, damageTaken=0.0,
             enemyShotEnergy=0.0, wallContacts=0, botContacts=0, bulletCollisions=0)
    turns = {r: [] for r in rounds}
    per_round = {r: dict(hits=0, shots=0, enemyHits=0, enemyShots=0) for r in rounds}
    for tick in events:
        if tick["type"] != "TickEventForObserver":
            continue
        r = tick["roundNumber"]
        turns[r].append(tick["turnNumber"])
        for e in tick["events"]:
            kind, b = e["type"], e.get("bullet", {})
            if kind == "BulletFiredEvent":
                if b["ownerId"] == uid:
                    c["shots"] += 1; c["shotEnergy"] += b["power"]; per_round[r]["shots"] += 1
                else:
                    c["enemyShots"] += 1; c["enemyShotEnergy"] += b["power"]; per_round[r]["enemyShots"] += 1
            elif kind == "BulletHitBotEvent":
                if b["ownerId"] == uid:
                    c["hits"] += 1; c["damageDealt"] += e["damage"]; per_round[r]["hits"] += 1
                elif e["victimId"] == uid:
                    c["enemyHits"] += 1; c["damageTaken"] += e["damage"]; per_round[r]["enemyHits"] += 1
            elif kind == "BulletHitBulletEvent":
                c["bulletCollisions"] += 1
            elif kind == "BotHitWallEvent" and e["victimId"] == uid:
                c["wallContacts"] += 1
            elif kind == "BotHitBotEvent" and e["botId"] == uid:
                c["botContacts"] += 1
    for r, seq in turns.items():
        if not seq or seq != list(range(seq[0], seq[-1] + 1)) or seq[0] != 1 or seq[-1] != endpoints[r - 1]["turnNumber"]:
            raise ValueError("Missing/duplicate official ticks")
    if not c["enemyShots"]:
        raise ValueError("Undefined incoming hit rate")
    if game["enemy"] == "AISurvivor":
        ais = [json.loads(x) for x in (path / "ais.jsonl").read_text().splitlines()]
        if [x["round"] for x in ais] != rounds or any(x["mode"] != "train" or x["seed"] != game["seed"] or x["checkpointLoaded"] is not False for x in ais):
            raise ValueError("AIS cold model or round evidence missing")
    round_shares, round_winners = [], []
    previous, previous_firsts = {}, {}
    for e in endpoints:
        totals = {x["id"]: x["totalScore"] for x in e["results"]}
        delta = {k: v - previous.get(k, 0) for k, v in totals.items()}
        previous = totals
        mine, theirs = delta.get(uid, 0), sum(v for k, v in delta.items() if k != uid)
        round_shares.append(mine / (mine + theirs) if mine + theirs else None)
        firsts = {x["id"]: x["firstPlaces"] for x in e["results"]}
        round_winners.append(firsts[uid] - previous_firsts.get(uid, 0) == 1)
        previous_firsts = firsts
    return dict(share=me["totalScore"] / sum(r["totalScore"] for r in results), wins=me["firstPlaces"],
                hitRate=c["hits"] / c["shots"] if c["shots"] else None, enemyHitRate=c["enemyHits"] / c["enemyShots"],
                **c, skips=sum(" skip " in m for m in markers), results=results, roundShares=round_shares,
                perRound=per_round, roundWins=round_winners, recordingSHA256=benchmark.digest(recordings[0]),
                turnsByRound=[len(s) for s in turns.values()])


def play(game, profile_data):
    name = game["bot"]
    bundle = BOTS[name]
    data = OUT / game["id"]
    data.mkdir()
    stage = ROOT / "build/keel-staging" / game["id"]
    stage.mkdir(parents=True)
    own = stage / name
    shutil.copytree(bundle / "bot" if name == "Keel" else bundle, own)
    if name == "Keel":
        shutil.copy2(bundle / "Keel.jar", own / "Keel.jar")
    version = json.loads((own / f"{name}.json").read_text())["version"]
    env = {k: os.environ[k] for k in ("PATH", "HOME", "JAVA_HOME", "LANG") if k in os.environ}
    marker_var = "KEEL_METRICS" if name == "Keel" else "MERIDIAN_METRICS"
    env.update({marker_var: str(data / "markers.txt"), "PYTHONDONTWRITEBYTECODE": "1",
                ("KEEL_SEED" if name == "Keel" else "MERIDIAN_SEED"): str(game["seed"])})
    enemy = game["enemy"]
    if enemy == "Retcon":
        src, api, rc = retcon.prepare(CACHE)
        other = stage / "Retcon"
        benchmark.isolate_opponent(src, other)
        if retcon.source_hash(other) != rc["sourceSHA256"]:
            raise ValueError("Retcon tree mismatch")
        env["RUMBLE_JAVA_CACHE"] = str(api)
    elif enemy == "Diamond Port":
        base = CACHE / "strong-python" / profile_data["python"]["commit"]
        src = base / "source"
        env.update(native.stage(src, stage, base / "venv/bin/python"))
        other = stage / "opponents/diamond-port"
        benchmark.isolate_opponent(src / "bots/ports/diamond-port", other)
        native.verify_staged(stage, ["diamond-port"])
    elif enemy == "AISurvivor":
        other = stage / "AISurvivor"
        shutil.copytree(ROOT / "AISurvivor", other)
        env.update(AISURVIVOR_MODEL_PATH=str(data / "ais-model.dat"), AISURVIVOR_METRICS_PATH=str(data / "ais.jsonl"),
                   AISURVIVOR_SEED=str(game["seed"]), AISURVIVOR_MODE="train")
    else:
        raise ValueError("Unregistered opponent")
    command = ["java", "-cp", str(CACHE / RUNNER), str(REG / "KeelBattle.java"), str(game["port"]), str(game["rounds"]),
               str(data), str(own), str(other)]
    save(data / "attempt.json", dict(game=game, command=command, endpoint="ws://localhost:" + str(game["port"]),
                                     staged={k: v for k, v in tree_hashes(own).items()},
                                     execution=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()))
    benchmark.run_process(command, stage, env, data / "runner.log", 3300)
    return metrics(data, game, name, version)


def decide(stage, rows):
    valid = all(r["status"] == "valid" for r in rows)
    keel = [r for r in rows if r["bot"] == "Keel"]
    by = lambda bot, enemy: [r for r in rows if r["bot"] == bot and r["enemy"] == enemy]
    checks = {"allValid": valid, "zeroKeelSkips": valid and all(r["skips"] == 0 for r in keel)}
    if not valid:
        return dict(decision="invalid", checks=checks)
    kd, md = by("Keel", "Diamond Port"), by("Meridian", "Diamond Port")
    mean = lambda rs: statistics.mean(r["share"] for r in rs)
    if stage == "screen":
        checks.update(
            diamondMeanAbove25=mean(kd) > 0.25,
            diamondMeanAboveControl=mean(kd) > mean(md),
            diamondWinsAtLeast2=sum(r["wins"] for r in kd) >= 2,
            retconAbove60=mean(by("Keel", "Retcon")) > 0.60,
            aisurvivorAbove50=mean(by("Keel", "AISurvivor")) > 0.50)
        progress = all(checks.values())
        confirm = progress and mean(kd) > 0.45
        return dict(decision="progress; confirmation open" if confirm else "progress; confirmation closed" if progress else "reject",
                    checks=checks, diamondMean=mean(kd), controlMean=mean(md))
    checks.update(
        diamondMeanAbove50=mean(kd) > 0.50,
        diamondMajorityRounds=sum(r["wins"] for r in kd) > sum(g["rounds"] for g in kd) / 2,
        diamondGamesAbove50=sum(r["share"] > 0.5 for r in kd) >= 4)
    return dict(decision="confirmed" if all(checks.values()) else "not confirmed", checks=checks, diamondMean=mean(kd))


def run(stage):
    profile_data = json.loads((OUT / "profile.json").read_bytes())
    if (OUT / "profile.json").read_bytes() != (REG / "profile.json").read_bytes():
        raise ValueError("Runtime differs from the registered profile")
    plan = [g for g in json.loads((REG / "games.json").read_bytes()) if g["stage"] == stage]
    if not plan:
        raise ValueError("Unknown stage")
    dest = OUT / f"decision-{stage}.json"
    if dest.exists():
        raise FileExistsError(dest)
    report = dict(stage=stage, games=[], decision="pending")
    save(dest, report)
    for game in plan:
        try:
            row = dict(**game, status="valid", **play(game, profile_data))
        except Exception as e:
            row = dict(**game, status="invalid", error=f"{type(e).__name__}: {e}")
        report["games"].append(row)
        save(dest, report)
        print(game["id"], row["status"], row.get("share"), row.get("wins"), flush=True)
    report.update(decide(stage, report["games"]))
    save(dest, report)
    print("Decision:", report["decision"], json.dumps(report["checks"]), flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["profile"]:
        profile()
    elif len(sys.argv) == 3 and sys.argv[1] == "run":
        run(sys.argv[2])
    else:
        raise SystemExit(__doc__)
