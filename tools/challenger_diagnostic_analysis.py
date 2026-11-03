"""Offline geometry for legal Challenger diagnostics; passage containment is NOT actual hit rate."""
import argparse
import json
import math
from pathlib import Path


def angle_delta(a, b):
    """Signed shortest difference, including wrap around the east-facing zero."""
    return (a - b + 180.0) % 360.0 - 180.0


def bearing(origin_x, origin_y, target_x, target_y):
    return math.degrees(math.atan2(target_y - origin_y, target_x - origin_x))


def passage_geometry(origin_x, origin_y, target_x, target_y, direction, radius=18.0):
    """Point-scan proxy at passage, independent of any observed bullet hit event."""
    distance = math.hypot(target_x - origin_x, target_y - origin_y)
    error = abs(angle_delta(direction, bearing(origin_x, origin_y, target_x, target_y)))
    half_width = math.degrees(math.asin(min(1.0, radius / max(radius, distance))))
    return {"range": distance, "absoluteAngularError": error,
            "bodyHalfWidth": half_width, "angularMargin": half_width - error,
            "contained": error <= half_width + 1e-9}


def segment_intersects_body(origin_x, origin_y, direction, inner, outer, x, y, radius=18.0):
    """Exact finite straight bullet segment versus a circle at ONE recorded own/scan position."""
    if inner < 0 or outer < inner:
        raise ValueError("invalid radial interval")
    theta = math.radians(direction)
    dx, dy = x - origin_x, y - origin_y
    along = dx * math.cos(theta) + dy * math.sin(theta)
    projection = min(outer, max(inner, along))
    return math.hypot(dx - projection * math.cos(theta), dy - projection * math.sin(theta)) <= radius + 1e-9


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def distribution(values):
    """Missing observations stay missing; an empty series cannot become a zero error."""
    values = [float(v) for v in values if finite_number(v)]
    if not values:
        return {"count": 0, "mean": None, "median": None, "p90": None}
    values.sort()
    return {"count": len(values), "mean": sum(values) / len(values),
            "median": (values[(len(values)-1)//2]+values[len(values)//2])/2,
            "p90": values[max(0, math.ceil(.9*len(values))-1)]}


def analyze(records):
    """Join confirmed shot IDs within rounds; never manufacture hit labels from virtual rays."""
    shots, passages, virtuals, hits, rounds = {}, [], [], [], []
    event_counts = {}
    for record in records:
        kind = record.get("type", record.get("event", record.get("eventtype")))
        event_counts[kind] = event_counts.get(kind, 0) + 1
        if kind == "shot":
            key = (record["round"], record["shotId"])
            if key in shots:
                raise ValueError(f"duplicate confirmed shot {key}")
            shots[key] = record
        elif kind == "passage": passages.append(record)
        elif kind == "virtual": virtuals.append(record)
        elif kind == "hit": hits.append(record)
        elif kind == "round": rounds.append(record)
    aiming, firing, gf_errors, origin_shift, raw_gfs = [], [], [], [], []
    model_errors = {}
    containment = {"predicted": [0, 0], "actualRay": [0, 0]}
    joined, unknown = 0, 0
    learned = {"passages": 0, "contained": 0}
    for p in passages:
        s = shots.get((p["round"], p.get("shotId"))) if p.get("shotId") is not None else None
        if s is None:
            unknown += 1
            continue
        joined += 1
        xy = [s.get("originX"), s.get("originY"), p.get("x"), p.get("y")]
        if all(finite_number(v) for v in xy):
            for label, direction in [("predicted", p.get("predictedDirection")), ("actualRay", s.get("direction"))]:
                if finite_number(direction):
                    g = passage_geometry(*xy, direction)
                    containment[label][0] += int(g["contained"])
                    containment[label][1] += 1
                    (aiming if label == "predicted" else firing).append(g["absoluteAngularError"])
                    if label == "predicted": model_errors.setdefault(str(s.get("model")), []).append(g["absoluteAngularError"])
                    if label == "predicted" and s.get("learned"):
                        learned["passages"] += 1
                        learned["contained"] += int(g["contained"])
        if finite_number(p.get("predictedGF")) and finite_number(p.get("rawObservedGF")):
            gf_errors.append(abs(p["predictedGF"]-p["rawObservedGF"]))
        if finite_number(p.get("rawObservedGF")): raw_gfs.append(p["rawObservedGF"])
        if finite_number(s.get("referenceBearing")) and finite_number(p.get("referenceBearing")):
            origin_shift.append(abs(angle_delta(s["referenceBearing"], p["referenceBearing"])))
    phase, aim_creation = [], []
    for s in shots.values():
        if finite_number(s.get("direction")) and finite_number(s.get("decisionGunDirection")):
            phase.append(abs(angle_delta(s["direction"], s["decisionGunDirection"])))
        if finite_number(s.get("direction")) and finite_number(s.get("aimDirection")):
            aim_creation.append(abs(angle_delta(s["direction"], s["aimDirection"])))
    actual_hit_keys = {(h["round"], h.get("shotId")) for h in hits if h.get("shotId") is not None}
    matched_hits = actual_hit_keys & shots.keys()
    selected_virtual = []
    for v in virtuals:
        s = shots.get((v["round"], v.get("shotId"))) if v.get("shotId") is not None else None
        if s is not None and v.get("model") == s.get("model"):
            selected_virtual.append((bool(v["hit"]), (v["round"],v["shotId"]) in actual_hit_keys))
    return {
        "warning": "Legal passage angular containment and virtual intersections are proxies, not actual hit rate. Truncated/censored cohorts stay missing.",
        "eventCounts": event_counts, "confirmedShotsRecorded": len(shots),
        "joinedPassages": joined, "unjoinedPassages": unknown,
        "passageAngularContainment": {k: {"contained":v[0],"count":v[1],"fraction":v[0]/v[1] if v[1] else None} for k,v in containment.items()},
        "learnedPassageContainment": learned,
        "predictedAngularErrorDegrees": distribution(aiming), "actualRayAngularErrorDegrees": distribution(firing),
        "absoluteRawGFError": distribution(gf_errors), "confirmedOriginReferenceShiftDegrees": distribution(origin_shift),
        "rawPassageGFOutsideUnitInterval": {"count": sum(abs(v)>1 for v in raw_gfs), "observed":len(raw_gfs)},
        "angularErrorBySelectedModel": {k:distribution(v) for k,v in model_errors.items()},
        "confirmedShotSupport": distribution([s.get("support") for s in shots.values()]),
        "confirmedShotKthDistance": distribution([s.get("kthDistance") for s in shots.values()]),
        "decisionToCreationGunPhaseDegrees": distribution(phase), "actualRayToRequestedAimDegrees": distribution(aim_creation),
        "actualHitEvents": len(hits), "distinctMatchedActualHitShots": len(matched_hits),
        "unmatchedActualHitShots": len(actual_hit_keys-shots.keys()),
        "selectedLegacyVirtualVsActual": {"count":len(selected_virtual), "virtualHitActualHit":sum(a and b for a,b in selected_virtual),
            "virtualHitNoActualHit":sum(a and not b for a,b in selected_virtual), "virtualMissActualHit":sum(not a and b for a,b in selected_virtual)},
        "roundSummaries": rounds,
        "recordedPowerSpent": sum(s["power"] for s in shots.values() if finite_number(s.get("power"))),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    records = []
    for path in args.inputs:
        with path.open(encoding="utf-8-sig") as stream:
            for number, line in enumerate(stream, 1):
                if line.strip():
                    try: records.append(json.loads(line, parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v))))
                    except (ValueError, json.JSONDecodeError) as exc: raise ValueError(f"{path}:{number}: {exc}") from exc
    result = json.dumps(analyze(records), indent=2, allow_nan=False)+"\n"
    if args.output: args.output.write_text(result, encoding="utf-8")
    else: print(result, end="")


if __name__ == "__main__": main()
