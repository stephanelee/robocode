"""Attribute physical bullet outcomes to the decision phase of their original shooter."""
import collections
import gzip
import hashlib
import json
import math
import sys
from pathlib import Path

AUDIT = Path("benchmarks/keelhelm-crossfire/stopped-attempt/independent-outcome-review.json")
STANDINGS = AUDIT.with_name("results.json")
HELM = "KeelHelm Crossfire"
DIAMOND = "Diamond Port"

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def phase(enemy_count):
    return "melee" if enemy_count > 1 else "lastTwo" if enemy_count == 1 else "other"

def range_bin(distance):
    return "under250" if distance < 250 else "250to499" if distance < 500 else "500plus"

def measure(recording, intents, official):
    contexts = {(v["round"], v["fireTurn"]): v for v in intents}
    rounds = {}
    shots = {}
    turns = collections.Counter()
    reconciled = set()
    previous = {}
    previous_round = None
    with gzip.open(recording,"rt") as stream:
        for line in stream:
            msg = json.loads(line)
            kind = msg["type"]
            if kind == "GameStartedEventForObserver":
                names = {p["id"]:p["name"] for p in msg["participants"]}
            elif kind == "RoundEndedEventForObserver":
                rounds[msg["roundNumber"]] = {p["name"]:p["bulletDamage"] for p in msg["results"]}
            elif kind == "GameEndedEventForObserver":
                observed = {p["id"]:p for p in msg["results"]}
                assert len(observed) == len(official["results"])
                for placement in official["results"]:
                    assert all(observed[placement["id"]][key] == value for key,value in placement.items())
            elif kind == "TickEventForObserver":
                r,t = msg["roundNumber"],msg["turnNumber"]
                if r != previous_round:
                    previous = {}
                    previous_round = r
                for i,s in previous.items():
                    turns[names[i],phase(s["enemyCount"])] += 1
                deaths = set()
                for e in msg["events"]:
                    et = e["type"]
                    if et == "BotDeathEvent":
                        deaths.add(e["victimId"])
                    if "bullet" not in e:
                        continue
                    b = e["bullet"]
                    key = (r,b["ownerId"],b["bulletId"])
                    if et == "BulletFiredEvent":
                        assert key not in shots and b["ownerId"] in previous and e["turnNumber"] == t
                        state = previous[b["ownerId"]]
                        owner = names[b["ownerId"]]
                        band = phase(state["enemyCount"])
                        shot = dict(owner=owner,phase=band,power=b["power"],hits=[],wall=False,collision=False)
                        shots[key] = shot
                        if owner == HELM and band == "melee":
                            c = contexts[r,t]
                            assert math.dist((b["x"],b["y"]),(c["x"],c["y"])) < 1e-6
                            assert abs(c["power"]-b["power"]) < 1e-9 and abs(math.remainder(b["direction"]-c["actualRay"],360)) < 1e-6
                            shot.update(intent=names[c["target"]],rangeBand=range_bin(c["range"]))
                            reconciled.add((r,t))
                    elif et == "BulletHitBotEvent":
                        # Event energy is victim energy AFTER subtracting the full physical damage.
                        credit = max(0,min(e["damage"],e["energy"] + e["damage"]))
                        shots[key]["hits"].append((names[e["victimId"]],e["damage"],credit))
                    elif et == "BulletHitWallEvent":
                        shots[key]["wall"] = True
                    elif et == "BulletHitBulletEvent":
                        shots[key]["collision"] = True
                        other = e["hitBullet"]
                        shots[r,other["ownerId"],other["bulletId"]]["collision"] = True
                previous = {s["id"]:s for s in msg["botStates"] if s["id"] not in deaths}
    assert set(contexts) == reconciled and len(rounds) == 5
    bins = collections.defaultdict(collections.Counter)
    for (name,band),n in turns.items():
        bins[name+" / "+band]["liveTurns"] += n
    per_round = collections.defaultdict(collections.Counter)
    for (r,_,_),v in shots.items():
        owner = v["owner"]
        hits = v["hits"]
        credited = sum(h[2] for h in hits)
        per_round[r][owner] += credited
        keys = [owner+" / "+v["phase"]]
        if "intent" in v:
            keys += ["range / "+v["rangeBand"],"intent / "+v["intent"]]
        for k in keys:
            c = bins[k]
            c.update(shots=1,power=v["power"],hitShots=bool(hits),hitEvents=len(hits),
                     multiHitShots=len(hits)>1,rawDamage=sum(h[1] for h in hits),credit=credited,
                     wall=v["wall"],collision=v["collision"],unresolved=not (hits or v["collision"] or v["wall"]))
            for victim,_,credit in hits:
                c["creditTo "+victim] += credit
                if "intent" in v:
                    c["intendedHits" if victim == v["intent"] else "collateralHits"] += 1
    # Round-ended scoreboards are cumulative. Preserve every owner and the actual rounding residual.
    running = collections.Counter()
    reconciliation = {}
    for r,by_name in rounds.items():
        running.update(per_round[r])
        if any(math.floor(running[n] + 0.5) != o for n,o in by_name.items()):
            raise ValueError(f"Official damage mismatch in {recording.name} round {r}")
        reconciliation[r] = {n:dict(official=o,credit=running[n],difference=o-running[n]) for n,o in by_name.items()}
    return dict(bins=bins,roundScores=reconciliation)

def main(evidence,intent_path):
    audit = json.loads(AUDIT.read_text())
    games = {g["id"]:g for g in json.loads(STANDINGS.read_text())["games"]}
    intents = json.loads(intent_path.read_text())
    out = {}
    for a in audit["games"]:
        game = a["game"]
        if game not in intents:
            assert not game.endswith("-candidate")
            continue
        path, = (evidence/"games"/game/"recordings").glob("*.battle.gz")
        assert sha(path) == a["recordingSHA256"]
        out[game] = measure(path,intents[game],games[game]["official"])
    assert len(out) == 12
    print(json.dumps(out,indent=2,sort_keys=True))

if __name__ == "__main__":
    main(*map(Path,sys.argv[1:]))
