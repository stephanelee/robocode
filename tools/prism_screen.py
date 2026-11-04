#!/usr/bin/env python3
"""Registered Prism 200-round screen (docs/prism-screen-protocol.md). Python 3.10+.

The harness refuses to run unless the frozen plan, the committed JAR hash and an explicit
independent premeasurement review reference all match. It never retunes, reruns or launches a
confirmation; every game (valid or invalid) is preserved in decision.json.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
ROOT = TOOLS.parent
PLAN = ROOT / "docs/prism/screen-plan.json"
CACHE = ROOT / ".benchmark-cache"
BUNDLE = ROOT / "experiments/Prism"
_NATIVE_PREPARED = None  # static pinned code/runtime only; every battle still starts fresh bot processes


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_plan(path):
    """Selects the registered plan to validate or run. Its repository-relative path is the only
    approval-record path, and its declared screenId names the durable receipt (default v1)."""
    global PLAN, PLAN_REL
    PLAN = (ROOT / path).resolve() if not Path(path).is_absolute() else Path(path)
    PLAN_REL = PLAN.relative_to(ROOT).as_posix()


def load_plan():
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    games = plan["games"]
    if len(games) != 13 or sum(g["rounds"] for g in games) != 200:
        raise ValueError("plan must contain the registered 13 games / 200 rounds")
    ids = [g["id"] for g in games]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate game id")
    for opp, n, rounds in (("Diamond Port", 3, 20), ("Retcon", 3, 20), ("Adaptive Prime", 3, 20), ("SpinBot", 2, 5), ("Walls", 2, 5)):
        g = [x for x in games if x["opponent"] == opp]
        if len(g) != n or any(x["rounds"] != rounds for x in g):
            raise ValueError("cohort mismatch: " + opp)
    return plan


def share(prism, other):
    total = prism + other
    return prism / total if total > 0 else 0.5


def telemetry_game(path):
    if not path.exists():
        return None
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    games = [r for r in rows if r.get("event") == "game"]
    return games[-1] if games else None


def gates(rows):
    """Specification 13.3 screen gates; failing any gate rejects the profile."""
    out = {}
    valid = all(r["status"] == "valid" for r in rows)
    out["allGamesValid"] = valid
    if not valid:
        out["decision"] = "reject"
        return out
    out["noFailures"] = all(r["failures"] == 0 for r in rows)
    by = {}
    for r in rows:
        by.setdefault(r["opponent"], []).append(r)
    out["skipRateBelow0.5pct"] = all(sum(r["skipped"] for r in g) < 0.005 * sum(r["turns"] for r in g) for g in by.values())
    out["collisionsBelow1pct"] = sum(r["collisions"] for r in rows) < 0.01 * sum(r["turns"] for r in rows)
    mean = {k: statistics.mean(r["share"] for r in v) for k, v in by.items()}
    out["meanShare"] = mean
    out["diamondAbove40"] = mean["Diamond Port"] > 0.40
    out["retconAbove40"] = mean["Retcon"] > 0.40
    out["adaptiveAtLeast50"] = mean["Adaptive Prime"] >= 0.50
    out["samplesAtLeast50"] = mean["SpinBot"] >= 0.50 and mean["Walls"] >= 0.50
    keys = [k for k in out if k not in ("meanShare",)]
    out["decision"] = "passed screen" if all(out[k] for k in keys) else "reject"
    return out


def stage_opponent(name, stage, env):
    import benchmark
    if name == "Retcon":
        import retcon
        src, api, rc = retcon.prepare(CACHE)
        other = stage / "Retcon"
        benchmark.isolate_opponent(src, other)
        if retcon.source_hash(other) != rc["sourceSHA256"]:
            raise ValueError("Retcon tree mismatch")
        env["RUMBLE_JAVA_CACHE"] = str(api)
        return other
    if name in ("Diamond Port", "Adaptive Prime"):
        import strong_melee_opponents as native
        key = "diamond-port" if name == "Diamond Port" else "adaptive-prime"
        lock = native.manifest()
        python = os.environ.get("PRISM_OPPONENT_PYTHON", "python3.13")
        global _NATIVE_PREPARED
        if _NATIVE_PREPARED is None:
            _NATIVE_PREPARED = native.prepare(CACHE, python)
        src, executable, provenance = _NATIVE_PREPARED
        if provenance["commit"] != lock["commit"]:
            raise ValueError("Prepared native source differs from the pinned lock")
        env.update(native.stage(src, stage, executable))
        other = stage / "opponents" / key
        spec = next(bot for bot in lock["opponents"] if Path(bot["descriptor"]).stem == key)
        benchmark.isolate_opponent(src / spec["sourcePath"], other)
        native.verify_staged(stage, [key])
        return other
    if name in ("SpinBot", "Walls"):
        archive = benchmark.download(CACHE, "sample-bots-Java-1.4.0.zip", benchmark.ASSETS["sample-bots-Java-1.4.0.zip"])
        samples = stage / "official-samples"
        benchmark.extract(archive, samples)
        descriptor, = samples.rglob(name + ".json")
        other = stage / name
        benchmark.isolate_opponent(descriptor.parent, other)
        return other
    raise ValueError("unregistered opponent " + name)


def play(game, out):
    import benchmark
    data = out / game["id"]
    data.mkdir(parents=True)
    stage = ROOT / "build/prism-staging" / game["id"]
    if stage.exists():
        raise FileExistsError(stage)
    stage.mkdir(parents=True)
    own = stage / "Prism"
    shutil.copytree(BUNDLE / "bot", own)
    shutil.copy2(BUNDLE / "Prism.jar", own / "Prism.jar")
    env = {k: os.environ[k] for k in ("PATH", "HOME", "JAVA_HOME", "LANG", "SystemRoot", "TEMP", "TMP") if k in os.environ}
    env.update(PRISM_SEED=str(game["seed"]), PRISM_TELEMETRY_PATH=str(data / "prism-telemetry.jsonl"), PYTHONDONTWRITEBYTECODE="1")
    other = stage_opponent(game["opponent"], stage, env)
    pair = [own, other] if game["seat"] == "prism-first" else [other, own]
    runner = benchmark.download(CACHE, "robocode-tankroyale-runner-1.4.0.jar", benchmark.ASSETS["robocode-tankroyale-runner-1.4.0.jar"])
    command = ["java", "-Dfile.encoding=UTF-8", "-cp", str(runner), str(TOOLS / "PrismBattle.java"),
               str(game["rounds"]), str(240 * game["rounds"]), str(data / "results.json"), str(pair[0]), str(pair[1])]
    (data / "attempt.json").write_text(json.dumps(dict(game=game, command=command,
        execution=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        prismJarSHA256=sha256(own / "Prism.jar")), indent=1), encoding="utf-8")
    benchmark.run_process(command, stage, env, data / "runner.log", 300 * game["rounds"] + 300)
    res = json.loads((data / "results.json").read_text(encoding="utf-8"))
    if res["numberOfRounds"] != game["rounds"]:
        raise ValueError("incomplete game")
    p = [r for r in res["results"] if r["name"] == "Prism"]
    o = [r for r in res["results"] if r["name"] != "Prism"]
    if len(p) != 1 or len(o) != 1:
        raise ValueError("identity mismatch")
    tel = telemetry_game(data / "prism-telemetry.jsonl")
    if tel is None:
        raise ValueError("missing Prism game telemetry")
    c = tel["counters"]
    turns = sum(v for k, v in c.items() if k.startswith("move.action.")) + c.get("move.reason.min-risk", 0)
    return dict(prismScore=p[0]["totalScore"], opponentScore=o[0]["totalScore"], share=share(p[0]["totalScore"], o[0]["totalScore"]),
                prismFirstPlaces=p[0]["firstPlaces"], opponentFirstPlaces=o[0]["firstPlaces"],
                prismBulletDamage=p[0]["bulletDamage"], opponentBulletDamage=o[0]["bulletDamage"],
                shotsRequested=c.get("fire.requested", 0), shotsConfirmed=c.get("fire.confirmed", 0), shotsHit=c.get("fire.hit", 0),
                collisions=c.get("collision.wall", 0) + c.get("collision.bot", 0), skipped=c.get("turns.skipped", 0), turns=turns,
                failures=sum(v for k, v in c.items() if k.startswith("failure.")), latency=tel.get("latencyMicros"),
                counters=c)


# ---------------------------------------------------------------- clearance and durable receipt

SCREEN_ID = "prism-screen-v1"
PLAN_REL = "docs/prism/screen-plan.json"


def _run(cmd):
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    return p.returncode, p.stdout.strip()


def receipt_ref(plan):
    """Durable receipt for this frozen screen: one tag per frozen candidate, never reused."""
    return "refs/tags/%s-%s" % (plan.get("screenId", SCREEN_ID), plan["freeze"]["prismJarSHA256"][:16])


APPROVAL_FIELDS = ("approvedReview", "approvedHead")


def frozen_content(plan):
    """Every plan field except the explicitly permitted approval record."""
    return {k: v for k, v in plan.items() if k not in APPROVAL_FIELDS}


def verify_clearance(plan, review, head, run=_run):
    """The review must be recorded, and the execution head may differ from the exact reviewed head
    only by the approval record: the plan file is the only changed path, and its contents equal
    the plan at the reviewed head after removing only the approval fields. Games, IDs, seeds,
    seats, opponent schedule and candidate/profile pins must be identical to what was reviewed."""
    if plan.get("approvedReview") in (None, "", "pending") or plan["approvedReview"] != review:
        raise SystemExit("independent premeasurement review not recorded in the frozen plan")
    reviewed = plan.get("approvedHead")
    if not reviewed or reviewed == "pending" or len(reviewed) != 40:
        raise SystemExit("exact reviewed head not recorded in the frozen plan")
    code, _ = run(["git", "merge-base", "--is-ancestor", reviewed, head])
    if code != 0:
        raise SystemExit("execution head does not descend from the reviewed head")
    code, out = run(["git", "diff", "--name-only", reviewed, head])
    changed = {l for l in out.splitlines() if l.strip()}
    if code != 0 or not changed <= {PLAN_REL}:
        raise SystemExit("execution head changes more than the approval record: %s" % sorted(changed - {PLAN_REL}))
    code, text = run(["git", "show", "%s:%s" % (reviewed, PLAN_REL)])
    if code != 0:
        raise SystemExit("the plan cannot be read at the reviewed head")
    try:
        reviewed_plan = json.loads(text)
    except ValueError:
        raise SystemExit("the plan at the reviewed head is not valid JSON")
    if frozen_content(reviewed_plan) != frozen_content(plan):
        diff = sorted(k for k in set(reviewed_plan) | set(plan)
                      if k not in APPROVAL_FIELDS and reviewed_plan.get(k) != plan.get(k))
        raise SystemExit("frozen plan contents differ from the reviewed head beyond the approval record: %s" % diff)


def consume_receipt(plan, repo, head, run_id, run=_run, receipt_file=None):
    """Atomically create the receipt tag at the execution head before any game. Creation fails if
    the tag already exists, so a second dispatch (or a re-run attempt) is refused. The receipt is
    never deleted, including after failure."""
    ref = receipt_ref(plan)
    code, out = run(["gh", "api", "-X", "POST", "repos/%s/git/refs" % repo, "-f", "ref=" + ref, "-f", "sha=" + head])
    if code != 0:
        raise SystemExit("screen receipt %s already consumed or unavailable; refusing to launch any game" % ref)
    receipt = dict(ref=ref, head=head, runId=str(run_id), repo=repo)
    path = receipt_file or (ROOT / "build/prism-screen-receipt.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8") as f:  # exclusive: one receipt per workspace
        json.dump(receipt, f)
    return receipt


def verify_receipt(plan, head, run_id, run=_run, receipt_file=None):
    """Before launching games: this run consumed the receipt, and the remote tag points at this head."""
    path = receipt_file or (ROOT / "build/prism-screen-receipt.json")
    if not path.exists():
        raise SystemExit("no receipt consumed by this run; refusing to launch any game")
    r = json.loads(path.read_text(encoding="utf-8"))
    ref = receipt_ref(plan)
    if r.get("ref") != ref or r.get("head") != head or r.get("runId") != str(run_id):
        raise SystemExit("receipt does not belong to this run and head")
    code, out = run(["git", "ls-remote", "origin", ref])
    if code != 0 or not out.startswith(head):
        raise SystemExit("remote receipt does not point at the execution head")
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--review", required=True, help="URL of the recorded independent premeasurement review")
    ap.add_argument("--head", required=True, help="exact execution commit (GITHUB_SHA)")
    ap.add_argument("--plan", default="docs/prism/screen-plan.json", help="registered plan (repository-relative)")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true", help="validate plan, pins and clearance; run nothing")
    mode.add_argument("--consume-receipt", metavar="OWNER/REPO", help="atomically consume the durable receipt; run nothing")
    a = ap.parse_args()
    select_plan(a.plan)
    plan = load_plan()
    freeze = plan["freeze"]
    if sha256(BUNDLE / "Prism.jar") != freeze["prismJarSHA256"]:
        raise SystemExit("Prism.jar does not match the frozen hash")
    verify_clearance(plan, a.review, a.head)
    run_id = os.environ.get("GITHUB_RUN_ID", "")
    if a.check_only:
        print("plan, pins and clearance valid; no games run")
        return
    if a.consume_receipt:
        if not run_id:
            raise SystemExit("receipts are consumed only inside the registered workflow")
        print("consumed", consume_receipt(plan, a.consume_receipt, a.head, run_id)["ref"])
        return
    verify_receipt(plan, a.head, run_id)
    out = a.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit("output directory must be empty")
    out.mkdir(parents=True, exist_ok=True)
    report = dict(plan=str(PLAN.relative_to(ROOT)), planSHA256=sha256(PLAN), head=a.head, receipt=receipt_ref(plan),
                  games=[], unplayedGameIds=[], decision="pending")
    dest = out / "decision.json"
    for index, game in enumerate(plan["games"]):
        try:
            row = dict(**game, status="valid", **play(game, out))
        except Exception as e:  # preserved as evidence; never rerun
            row = dict(**game, status="invalid", error=repr(e))
        report["games"].append(row)
        if row["status"] == "invalid":
            report["unplayedGameIds"] = [g["id"] for g in plan["games"][index + 1:]]
            dest.write_text(json.dumps(report, indent=1), encoding="utf-8")
            break  # acquisition/integrity failure only; ordinary score/safety losses never stop the budget
        dest.write_text(json.dumps(report, indent=1), encoding="utf-8")
    report["gates"] = gates(report["games"])
    report["decision"] = report["gates"]["decision"]
    dest.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(report["decision"])


if __name__ == "__main__":
    main()
