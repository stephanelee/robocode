"""Score the preregistered frozen target memories on the separate v3 Diamond game."""
import collections
import gzip
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "benchmarks"
TRAIN = ROOT / "meridian-aim/raw/meridian-aim-v3-diamond-screen"
TEST = ROOT / "meridian-economy/raw/meridian-economy-v3-control-diamond-screen"
OUTPUT = ROOT / "meridian-motion-memory/memory-result.json"


def lateral(scan, bearing):
    return scan["speed"] * math.sin(math.radians(scan["direction"]) - bearing)


def wall_distance(scan, width, height):
    travel = math.radians(scan["direction"] + (180 if scan["speed"] < 0 else 0))
    walls = []
    for pos, slope, size in ((scan["x"], math.cos(travel), width), (scan["y"], math.sin(travel), height)):
        if abs(slope) > 1e-9:
            walls.append(((size - 18 if slope > 0 else 18) - pos) / slope)
    return max(0, min(1, min(walls) / 180))


def example(bullet, launch, ticks, scans, own_id, arena):
    decision = launch - 1
    if any(t not in scans for t in (decision, decision - 4, decision - 8)):
        return "missing_decision_history", None
    own = next(bot for bot in ticks[decision]["botStates"] if bot["id"] == own_id)
    x, y = bullet["x"], bullet["y"]
    if math.hypot(x - own["x"], y - own["y"]) > 1e-4:
        return "origin_mismatch", None
    p = scans[decision]
    dx, dy = p["x"] - x, p["y"] - y
    bearing = math.atan2(dy, dx)
    distance = math.hypot(dx, dy)
    speed = 20 - 3 * bullet["power"]
    escape = math.asin(8 / speed)
    side = -1 if lateral(p, bearing) < 0 else 1
    features = [abs(lateral(p, bearing)) / 3,
                p["speed"] * math.cos(math.radians(p["direction"]) - bearing) / 4,
                wall_distance(p, *arena) / .35, distance / speed / 15]
    for lag in (4, 8):
        features.append(side * lateral(scans[decision - lag], bearing) / 4)
    for t in range(launch, max(ticks) + 1):
        if t not in scans:
            continue
        dx, dy = scans[t]["x"] - x, scans[t]["y"] - y
        arrival_range = math.hypot(dx, dy)
        if arrival_range > speed * (t - launch + 2):
            continue
        before = scans.get(t - 1)
        if before is None or math.hypot(before["x"] - x, before["y"] - y) <= speed * (t - launch + 1):
            return "unbracketed", None
        angle = math.atan2(dy, dx)
        return "eligible", dict(features=features, offset=side * math.remainder(angle-bearing, 2*math.pi) / escape,
                                bearing=bearing, side=side, escape=escape, distance=distance, actual=angle,
                                radius=arrival_range, recorded=math.radians(bullet["direction"]), turn=launch)
    return "round_ended", None


def read(folder):
    path, = folder.glob("*.battle.gz")
    with gzip.open(path, "rt") as stream:
        events = [json.loads(line) for line in stream]
    uid = next(bot["id"] for bot in events[-1]["results"] if (bot["name"], bot["version"]) == ("Meridian", "0.3.0"))
    setup = events[0]["gameSetup"]
    counts = collections.Counter()
    rows = []
    for r in range(1, setup["numberOfRounds"] + 1):
        ticks = {t["turnNumber"]: t for t in events if t["type"] == "TickEventForObserver" and t["roundNumber"] == r}
        scans = {t: e for t, tick in ticks.items() for e in tick["events"]
                 if e["type"] == "ScannedBotEvent" and e["scannedByBotId"] == uid}
        for t, tick in ticks.items():
            for e in tick["events"]:
                if e["type"] != "BulletFiredEvent" or e["bullet"]["ownerId"] != uid:
                    continue
                reason, row = example(e["bullet"], t, ticks, scans, uid, (setup["arenaWidth"], setup["arenaHeight"]))
                counts[reason] += 1
                if row is not None:
                    rows.append(dict(row, round=r))
    return rows, dict(counts=counts, source=str(path.relative_to(ROOT)), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def forecast(bank, query, dimensions):
    bins = [0.] * 51
    footprint = max(.05, math.atan2(18, query["distance"]) / query["escape"])
    for row in bank:
        distance = sum((x-y)**2 for x, y in zip(row["features"][:dimensions], query["features"][:dimensions]))
        weight = 1 / (1 + distance)**2
        for index in range(51):
            gap = abs(index * .05 - 1.25 - row["offset"])
            bins[index] += weight * max(0, 1 - gap / footprint)
    best = 25
    for i, score in enumerate(bins):
        if score > bins[best]:
            best = i
    return query["bearing"] + query["side"] * query["escape"] * (best * .05 - 1.25)


def main():
    train, training = read(TRAIN)
    bank = [sample for sample in train if abs(sample["offset"]) <= 1.25]
    test, evaluation = read(TEST)
    rows = []
    for shot in test:
        angles = dict(recorded=shot["recorded"], state=forecast(bank, shot, 4), sequence=forecast(bank, shot, 6))
        misses = {key: 2 * shot["radius"] * abs(math.sin((angle - shot["actual"])/2)) for key, angle in angles.items()}
        rows.append(dict(round=shot["round"], turn=shot["turn"], miss=misses))
    scores = {}
    for arm in ("recorded", "state", "sequence"):
        miss = [r["miss"][arm] for r in rows]
        scores[arm] = dict(within18=sum(d <= 18 for d in miss), fraction=sum(d <= 18 for d in miss)/len(miss),
                           meanMiss=sum(miss)/len(miss))
    passed = all(scores["sequence"]["fraction"] >= scores[c]["fraction"] + .02
                 and scores["sequence"]["meanMiss"] < scores[c]["meanMiss"] for c in ("recorded", "state"))
    result = dict(training=training, bankSize=len(bank), evaluation=evaluation, scores=scores, advance=passed, rows=rows)
    OUTPUT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k:v for k,v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
