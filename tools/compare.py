#!/usr/bin/env python3
"""Compare immutable AISurvivor versions against pinned Retcon without retraining."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess

import benchmark
import retcon

BOT_FILES = ("AISurvivor.jar", "AISurvivor.json", "AISurvivor.sh", "AISurvivor.cmd")
THRESHOLDS = dict(meanShareGain=0.05, minimumImprovedBlocks=2, maximumBlockShareLoss=0.05,
                  maximumWinsLost=3, maximumWallRateRatio=1.2, maximumSkippedTurns=0,
                  earlyShareLoss=0.15, earlyPreviousWins=4)


def seeds_value(value):
    seeds = [int(part) for part in value.split(",")]
    if len(seeds) != 3 or len(set(seeds)) != 3 or any(not -(2**63) <= seed < 2**63 for seed in seeds):
        raise argparse.ArgumentTypeError("exactly three distinct signed 64-bit seeds are required")
    return seeds


def snapshot_bots(previous, candidate, output):
    """Snapshot both complete bundles before invoking any external battle process."""
    for name in BOT_FILES[1:]:
        if (previous / name).read_bytes() != (candidate / name).read_bytes():
            raise ValueError(f"Bot configuration/launcher differs between arms: {name}")
    snapshots, hashes = {}, {}
    for arm, source in (("previous", previous), ("candidate", candidate)):
        target = output / "artifacts" / arm
        target.mkdir(parents=True)
        hashes[arm] = {}
        for name in BOT_FILES:
            shutil.copy2(source / name, target / name)
            hashes[arm][name] = benchmark.digest(target / name)
        snapshots[arm] = target
    return snapshots, hashes


def verify_snapshots(snapshots, hashes):
    for arm, source in snapshots.items():
        for name, expected in hashes[arm].items():
            if benchmark.digest(source / name) != expected:
                raise ValueError(f"Immutable version snapshot changed: {arm}/{name}")


def copy_checkpoint(source, destination, expected):
    shutil.copy2(source, destination)
    if benchmark.digest(destination) != expected:
        raise ValueError("Checkpoint copy does not match the immutable input")


def score_share(battle):
    total = sum(row["totalScore"] for row in battle["results"])
    if not math.isfinite(total) or total <= 0:
        raise ValueError("Official total score must be positive and finite")
    score = battle["totalScore"]
    if not math.isfinite(score) or score < 0:
        raise ValueError("Official bot score must be nonnegative and finite")
    return score / total


def wilson(wins, rounds):
    """Descriptive interval; within-game adaptation means rounds are not IID trials."""
    if not 0 <= wins <= rounds or rounds <= 0:
        raise ValueError("Invalid wins/rounds")
    z = 1.959963984540054
    p = wins / rounds
    denominator = 1 + z * z / rounds
    center = (p + z * z / (2 * rounds)) / denominator
    half = z * math.sqrt(p * (1 - p) / rounds + z * z / (4 * rounds * rounds)) / denominator
    return dict(lower=max(0, center - half), upper=min(1, center + half))


def aggregate(blocks, arm):
    rows = [block[arm] for block in blocks]
    rounds = sum(row["rounds"] for row in rows)
    wins = sum(row["firstPlaces"] for row in rows)
    totals = {key: sum(row["metrics"][key] for row in rows)
              for key in ("turns", "shotsFired", "bulletHits", "damageDealt", "damageTaken",
                          "wallCollisions", "skippedTurns")}
    if totals["turns"] <= 0:
        raise ValueError("Comparison requires measured positive bot turns")
    return dict(rounds=rounds, wins=wins, winRate=wins / rounds,
                winRateWilson95=wilson(wins, rounds),
                meanScoreShare=statistics.mean(score_share(row) for row in rows),
                hitRate=totals["bulletHits"] / totals["shotsFired"] if totals["shotsFired"] else None,
                wallsPer1000Turns=1000 * totals["wallCollisions"] / totals["turns"], **totals)


def decision(blocks):
    previous = aggregate(blocks, "previous")
    candidate = aggregate(blocks, "candidate")
    deltas = [score_share(block["candidate"]) - score_share(block["previous"]) for block in blocks]
    checks = dict(
        complete=len(blocks) == 3,
        meanShareGain=candidate["meanScoreShare"] - previous["meanScoreShare"] >= THRESHOLDS["meanShareGain"] - 1e-12,
        improvedBlocks=sum(delta > 0 for delta in deltas) >= THRESHOLDS["minimumImprovedBlocks"],
        maximumBlockLoss=min(deltas) >= -THRESHOLDS["maximumBlockShareLoss"] - 1e-12,
        winsNotMateriallyLower=candidate["wins"] >= previous["wins"] - THRESHOLDS["maximumWinsLost"],
        noSkippedTurns=candidate["skippedTurns"] == 0,
        wallRate=candidate["wallsPer1000Turns"] <= previous["wallsPer1000Turns"] * THRESHOLDS["maximumWallRateRatio"] + 1e-12)
    return dict(status="retain" if all(checks.values()) else "reject" if len(blocks) == 3 else "incomplete",
                checks=checks, previous=previous, candidate=candidate, scoreShareDeltas=deltas)


def catastrophic_first_block(block):
    return (score_share(block["previous"]) - score_share(block["candidate"]) >= THRESHOLDS["earlyShareLoss"]
            and block["candidate"]["firstPlaces"] == 0
            and block["previous"]["firstPlaces"] >= THRESHOLDS["earlyPreviousWins"])


def save_report(output, report):
    path = output / "comparison.json"
    temporary = output / "comparison.json.part"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def run(args):
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("output must be empty so prior evidence is never overwritten")
    output.mkdir(parents=True, exist_ok=True)
    snapshots, hashes = snapshot_bots(args.previous_bot_dir.resolve(), args.candidate_bot_dir.resolve(), output)
    plan = args.plan.read_bytes()
    report = dict(schemaVersion=1, status="preparing", policy=args.policy, seeds=args.seeds,
                  purpose=args.purpose,
                  battleTimeoutSeconds=args.timeout,
                  roundsPerBlock=args.rounds, trainingRounds=args.training_rounds if args.policy == "trained" else 0,
                  previousSource=args.previous_source, candidateSource=args.candidate_source,
                  preregistrationCommit=args.plan_commit, preregistrationSHA256=hashlib.sha256(plan).hexdigest(),
                  thresholds=THRESHOLDS, bundleSHA256=hashes, blocks=[],
                  limitations="Retcon adaptation only. Starts and scheduling are unpaired/unseeded. Seeds initialize matching cold networks, not matching game states. Three game blocks provide limited uncertainty evidence. Wilson win intervals are descriptive because rounds share an adaptive opponent.")
    (output / "preregistered-plan.md").write_bytes(plan)
    checkpoint = None
    if args.policy == "trained":
        checkpoint = output / "artifacts" / "input-checkpoint.dat"
        shutil.copy2(args.checkpoint.resolve(), checkpoint)
        report["checkpointSHA256"] = benchmark.digest(checkpoint)
    save_report(output, report)
    try:
        runtime = subprocess.run([args.java, "-XshowSettings:properties", "-version"], capture_output=True, text=True, timeout=15, check=True)
        import re
        major = re.search(r"java\.specification\.version\s*=\s*(\d+)", runtime.stderr)
        if not major or int(major.group(1)) < 22:
            raise ValueError("Retcon source launcher requires JDK22+; use the pinned JDK25 CI runtime")
        report["runtimeJava"] = (runtime.stderr + runtime.stdout).strip()
        cache = args.cache.resolve()
        source, api_cache, provenance = retcon.prepare(cache)
        runner = benchmark.download(cache, "robocode-tankroyale-runner-1.4.0.jar", benchmark.ASSETS["robocode-tankroyale-runner-1.4.0.jar"])
        report.update(opponentProvenance=[provenance], tankRoyaleVersion=benchmark.VERSION,
                      runnerSHA256=benchmark.digest(runner), status="running")
        save_report(output, report)
        for index, seed in enumerate(args.seeds):
            order = ["previous", "candidate"] if index % 2 == 0 else ["candidate", "previous"]
            block = dict(seed=seed, order=order)
            report["inProgressBlock"] = dict(seed=seed, order=order)
            save_report(output, report)
            for arm in order:
                verify_snapshots(snapshots, hashes)
                arm_output = output / f"seed-{seed}" / arm
                arm_output.mkdir(parents=True)
                model = arm_output / "checkpoint.dat"
                if checkpoint:
                    copy_checkpoint(checkpoint, model, report["checkpointSHA256"])
                battle_args = argparse.Namespace(java=args.java, bot_dir=snapshots[arm], seed=seed,
                    timeout=args.timeout, training_rounds=args.training_rounds, require_metrics=True,
                    opponent_env={"RUMBLE_JAVA_CACHE": str(api_cache)}, retcon_source=source)
                result = benchmark.battle(battle_args, arm_output, runner, output, "Retcon", ["Retcon"], args.policy, args.rounds, model)
                result["scoreShare"] = score_share(result)
                block[arm] = result
                report["inProgressBlock"][arm] = result
                save_report(output, report)
            report["blocks"].append(block)
            del report["inProgressBlock"]
            report["decision"] = decision(report["blocks"])
            if args.purpose == "diagnostic":
                report["decision"]["status"] = "diagnostic"
            if args.purpose == "primary" and index == 0 and args.rounds >= 20 and catastrophic_first_block(block):
                report.update(status="early-rejected", earlyStopReason="Preregistered catastrophic first-block loss")
                report["decision"]["status"] = "reject"
                break
            save_report(output, report)
            print(f"seed {seed}: previous={block['previous']['scoreShare']:.2%}, candidate={block['candidate']['scoreShare']:.2%}", flush=True)
        else:
            report["status"] = "complete"
        verify_snapshots(snapshots, hashes)
        if checkpoint and benchmark.digest(checkpoint) != report["checkpointSHA256"]:
            raise ValueError("Immutable input checkpoint changed")
        save_report(output, report)
        print(f"Decision: {report['decision']['status']}; see {output / 'comparison.json'}", flush=True)
    except BaseException as error:
        report.update(status="failed", failure=dict(type=type(error).__name__, message=str(error)))
        save_report(output, report)
        raise
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-bot-dir", type=Path, required=True)
    parser.add_argument("--candidate-bot-dir", type=Path, required=True)
    parser.add_argument("--previous-source", required=True)
    parser.add_argument("--candidate-source", required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-commit", required=True)
    parser.add_argument("--policy", choices=("untrained", "fixed-minrisk", "trained"), default="untrained")
    parser.add_argument("--purpose", choices=("primary", "diagnostic"), default="primary")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--training-rounds", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--seeds", type=seeds_value, default=seeds_value("20261102,20261103,20261104"))
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--java", default="java")
    parser.add_argument("--cache", type=Path, default=benchmark.ROOT / ".benchmark-cache")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.rounds, args.timeout, args.training_rounds) <= 0:
        parser.error("round counts and timeout must be positive")
    if (args.checkpoint is not None) != (args.policy == "trained"):
        parser.error("--checkpoint is required only for trained evaluation")
    run(args)


if __name__ == "__main__":
    main()
