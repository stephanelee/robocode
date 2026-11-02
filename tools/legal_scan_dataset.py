"""Build state-only forecast examples from legal AIS observations; no collection/training.

Input splits are assigned by whole game BEFORE windows/labels. Never interpolate
future scans, join across rounds, use current decisions or fit preprocessing here.
"""
import argparse
import copy
from contextlib import ExitStack
import hashlib
import json
import math
from pathlib import Path
import re

SPLITS = ("train", "dev", "withheld", "transfer")
MAX_RECORDS = 200_000
MAX_EVENT_BYTES = 256 * 1024 * 1024


def object_keys(value, required, optional=()):
    if not isinstance(value, dict) or not set(required) <= value.keys() or value.keys() - set(required) - set(optional):
        raise ValueError(f"Expected keys {required}, optional {optional}")


def integer(value, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError("Invalid integer")
    return value


def number(value, minimum=-math.inf, maximum=math.inf):
    if type(value) not in (int, float) or not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError("Invalid finite number")
    return value


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", value):
        raise ValueError("Invalid identifier")
    return value


def no_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_json(text):
    return json.loads(text, object_pairs_hook=no_duplicate_keys,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"Invalid constant {value}")))


def validate_manifest(manifest):
    object_keys(manifest, ("schemaVersion", "datasetId", "provenance", "games"))
    if type(manifest["schemaVersion"]) is not int or manifest["schemaVersion"] != 1:
        raise ValueError("Unsupported manifest schema")
    identifier(manifest["datasetId"])
    provenance = manifest["provenance"]
    object_keys(provenance, ("sourceCommit", "botJarSha256", "engineVersion", "collectorVersion"))
    for key, length in (("sourceCommit", 40), ("botJarSha256", 64)):
        if not isinstance(provenance[key], str) or not re.fullmatch(rf"[0-9a-f]{{{length}}}", provenance[key]):
            raise ValueError(f"Invalid {key}")
    identifier(provenance["engineVersion"])
    identifier(provenance["collectorVersion"])
    games = manifest["games"]
    if not isinstance(games, list) or not 1 <= len(games) <= 1000:
        raise ValueError("Expected 1..1000 preassigned games")
    index, acquisitions = {}, set()
    for game in games:
        object_keys(game, ("gameId", "acquisitionId", "split", "opponent", "opponentPin", "arenaWidth", "arenaHeight"), ("aisSeed",))
        game_id = identifier(game["gameId"])
        acquisition = identifier(game["acquisitionId"])
        identifier(game["opponent"])
        identifier(game["opponentPin"])
        for dimension in ("arenaWidth", "arenaHeight"):
            if integer(game[dimension], 400) > 5000:
                raise ValueError("Invalid arena size")
        if "aisSeed" in game and (type(game["aisSeed"]) is not int or not -(2 ** 63) <= game["aisSeed"] < 2 ** 63):
            raise ValueError("Invalid AIS seed")
        if game["split"] not in SPLITS or game_id in index or acquisition in acquisitions:
            raise ValueError("Invalid split or repeated game/acquisition (whole-game leakage)")
        index[game_id] = game
        acquisitions.add(acquisition)
    return index


def state_fields(raw, own=False):
    required = ("x", "y", "direction", "speed") + (("gunDirection",) if own else ("observedTurn",))
    optional = ("energy", "radarDirection") if own else ("energy",)
    object_keys(raw, required, optional)
    result = {"x": number(raw["x"], 0, 5000), "y": number(raw["y"], 0, 5000),
              "direction": number(raw["direction"], 0, 360), "speed": number(raw["speed"], -8, 8)}
    if own:
        result["gunDirection"] = number(raw["gunDirection"], 0, 360)
        result["radarDirection"] = None if raw.get("radarDirection") is None else number(raw["radarDirection"], 0, 360)
    else:
        result["observedTurn"] = integer(raw["observedTurn"])
    result["energy"] = None if raw.get("energy") is None else number(raw["energy"])
    return result


def validate_record(raw):
    object_keys(raw, ("schemaVersion", "gameId", "round", "turn", "targetId", "own", "scan", "previousIntent", "confirmedShots"))
    if type(raw["schemaVersion"]) is not int or raw["schemaVersion"] != 1:
        raise ValueError("Unsupported event schema")
    result = {key: integer(raw[key]) for key in ("round", "turn", "targetId")}
    result["gameId"] = identifier(raw["gameId"])
    result["own"] = state_fields(raw["own"], own=True)
    scan = None if raw["scan"] is None else state_fields(raw["scan"])
    if scan is not None and scan["observedTurn"] > result["turn"]:
        raise ValueError("Future scan cannot enter features")
    result["scan"] = scan
    intent = raw["previousIntent"]
    if intent is not None:
        object_keys(intent, ("submittedTurn",), ("targetSpeed", "turnRate", "gunTurnRate", "radarTurnRate", "firepower", "action"))
        submitted = integer(intent["submittedTurn"], 0)
        if submitted >= result["turn"]:
            raise ValueError("Current/future intent is not available at state-only forecast time")
        intent = {"submittedTurn": submitted, **{key: None if intent.get(key) is None else number(intent[key])
                  for key in ("targetSpeed", "turnRate", "gunTurnRate", "radarTurnRate", "firepower")},
                  "action": None if intent.get("action") is None else integer(intent["action"], 0)}
    result["previousIntent"] = intent
    shots = raw["confirmedShots"]
    if not isinstance(shots, list) or len(shots) > 1000:
        raise ValueError("Invalid confirmed own-shot list")
    seen, result_shots = set(), []
    for shot in shots:
        object_keys(shot, ("turn", "bulletId", "power", "direction", "x", "y"))
        event_turn, bullet = integer(shot["turn"]), integer(shot["bulletId"])
        if event_turn > result["turn"] or bullet in seen:
            raise ValueError("Future or duplicate own bullet event")
        seen.add(bullet)
        result_shots.append(dict(turn=event_turn, bulletId=bullet, power=number(shot["power"], .1, 3),
                                 direction=number(shot["direction"], 0, 360), x=number(shot["x"], 0, 5000),
                                 y=number(shot["y"], 0, 5000)))
    result["confirmedShots"] = sorted(result_shots, key=lambda shot: (shot["turn"], shot["bulletId"]))
    return result


def validated_groups(records, manifest):
    games = validate_manifest(manifest)  # partition before feature windows
    groups, cached_scans, shots, own_snapshots = {}, {}, {}, {}
    for count, raw in enumerate(records, 1):
        if count > MAX_RECORDS:
            raise ValueError("Event count exceeds bounded builder limit")
        row = validate_record(raw)
        if row["gameId"] not in games:
            raise ValueError("Event game was not assigned before processing")
        game = games[row["gameId"]]
        for position in (row["own"], row["scan"], *row["confirmedShots"]):
            if position is not None and (position["x"] > game["arenaWidth"] or position["y"] > game["arenaHeight"]):
                raise ValueError("Position outside declared arena")
        own_key = (row["gameId"], row["round"], row["turn"])
        snapshot = {key: row[key] for key in ("own", "previousIntent", "confirmedShots")}
        if own_key in own_snapshots and own_snapshots[own_key] != snapshot:
            raise ValueError("Conflicting own snapshot across targets")
        own_snapshots[own_key] = snapshot
        group_key = (row["gameId"], row["round"], row["targetId"])
        group = groups.setdefault(group_key, {})
        if row["turn"] in group:
            raise ValueError("Duplicate game/round/turn/target event")
        group[row["turn"]] = row
        if row["scan"] is not None:
            scan_key = (*group_key, row["scan"]["observedTurn"])
            if scan_key in cached_scans and cached_scans[scan_key] != row["scan"]:
                raise ValueError("Conflicting values for one observed scan")
            cached_scans[scan_key] = row["scan"]
        for shot in row["confirmedShots"]:
            key = (row["gameId"], row["round"], shot["bulletId"])
            if key in shots and shots[key] != shot:
                raise ValueError("Conflicting own bullet event")
            shots[key] = shot
    for turns in groups.values():
        last_observed = 0
        for _, row in sorted(turns.items()):
            if row["scan"] is not None:
                observed = row["scan"]["observedTurn"]
                if observed < last_observed:
                    raise ValueError("Cached scan observation turn regresses")
                last_observed = observed
    return games, groups


def validate_settings(horizons, history):
    if type(history) is not int or not 1 <= history <= 64:
        raise ValueError("History must be 1..64 turns")
    if not horizons or len(horizons) > 8 or len(set(horizons)) != len(horizons):
        raise ValueError("Expected 1..8 unique horizons")
    for horizon in horizons:
        integer(horizon)
        if horizon > 150:
            raise ValueError("Horizon exceeds 150 turns")


def build_samples(records, manifest, horizons=(1, 3, 8), history=8):
    validate_settings(horizons, history)
    games, groups = validated_groups(records, manifest)
    for (game_id, round_number, target_id), turns in sorted(groups.items()):
        split = games[game_id]["split"]
        for anchor, row in sorted(turns.items()):
            if row["scan"] is None:
                continue  # No previously observed target; no target-state forecast.
            frames = []
            for turn in range(max(1, anchor - history + 1), anchor + 1):
                prior = turns.get(turn)
                if prior is None:
                    frames.append({"turn": turn, "recordPresent": False, "own": None,
                                   "scan": None, "scanAge": None, "scanFresh": False,
                                   "previousIntent": None, "confirmedShots": []})
                else:
                    scan = prior["scan"]
                    age = None if scan is None else turn - scan["observedTurn"]
                    frames.append({"turn": turn, "recordPresent": True, "own": prior["own"],
                                   "scan": scan, "scanAge": age, "scanFresh": age == 0,
                                   "previousIntent": prior["previousIntent"],
                                   "confirmedShots": prior["confirmedShots"]})
            for horizon in sorted(horizons):
                target_turn = anchor + horizon
                future = turns.get(target_turn)
                scan = None if future is None else future["scan"]
                observed = scan is not None and scan["observedTurn"] == target_turn
                yield dict(schemaVersion=1, gameId=game_id, split=split, round=round_number,
                           anchorTurn=anchor, targetId=target_id, horizon=horizon,
                           features={"history": copy.deepcopy(frames)}, label={"observed": observed,
                           "turn": target_turn, "x": scan["x"] if observed else None,
                           "y": scan["y"] if observed else None})


def load_events(path):
    if path.stat().st_size > MAX_EVENT_BYTES:
        raise ValueError("Event input exceeds bounded size limit")
    with path.open(encoding="utf-8") as stream:
        count = 0
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            if len(line) > 65536:
                raise ValueError(f"Oversized event at line {line_number}")
            count += 1
            if count > MAX_RECORDS:
                raise ValueError("Event count exceeds bounded builder limit")
            try:
                yield parse_json(line)
            except (ValueError, TypeError) as error:
                raise ValueError(f"Invalid event JSON at line {line_number}") from error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizons", default="1,3,8")
    parser.add_argument("--history", type=int, default=8)
    args = parser.parse_args()
    if args.manifest.stat().st_size > 2 * 1024 * 1024:
        raise ValueError("Manifest too large")
    manifest = parse_json(args.manifest.read_text(encoding="utf-8"))
    records = list(load_events(args.events))
    horizons = tuple(int(h) for h in args.horizons.split(","))
    validate_settings(horizons, args.history)
    # Materialize validation before opening outputs, without materializing all windows.
    validated_groups(records, manifest)
    if args.output.exists():
        raise ValueError("Output directory must be new; preserve published datasets")
    args.output.mkdir(parents=True)
    counts = {split: {"examples": 0, "observedLabels": 0} for split in SPLITS}
    coverage = {split: {str(h): {"examples": 0, "observedLabels": 0} for h in sorted(horizons)} for split in SPLITS}
    game_coverage = {game["gameId"]: {"examples": 0, "observedLabels": 0} for game in manifest["games"]}
    with ExitStack() as stack:
        streams = {split: stack.enter_context((args.output / f"{split}.jsonl").open("w", encoding="utf-8", newline="\n")) for split in SPLITS}
        for sample in build_samples(records, manifest, horizons, args.history):
            streams[sample["split"]].write(json.dumps(sample, allow_nan=False) + "\n")
            counts[sample["split"]]["examples"] += 1
            counts[sample["split"]]["observedLabels"] += sample["label"]["observed"]
            cell = coverage[sample["split"]][str(sample["horizon"])]
            cell["examples"] += 1
            cell["observedLabels"] += sample["label"]["observed"]
            game_coverage[sample["gameId"]]["examples"] += 1
            game_coverage[sample["gameId"]]["observedLabels"] += sample["label"]["observed"]
    result = dict(schemaVersion=1, manifest=manifest, history=args.history, horizons=horizons,
                  toolSha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  eventsSha256=hashlib.sha256(args.events.read_bytes()).hexdigest(), counts=counts,
                  coverageByHorizon=coverage,
                  coverageByGame=game_coverage,
                  gameInputRecords={game["gameId"]: sum(row["gameId"] == game["gameId"] for row in records) for game in manifest["games"]},
                  inputRecords=len(records), scope="legal-observation label preparation only; no acquisition, normalization or training")
    (args.output / "dataset.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
