#!/usr/bin/env python3
"""Three fixed-policy blocks observing unchanged Retcon's native diagnostics."""
import argparse
import json
import os
from pathlib import Path
import subprocess

import benchmark
import retcon
import retcon_stats


def aggregate(blocks):
    complete = [block["retconStats"] for block in blocks if block["retconStats"]["status"] == "complete"]
    if len(complete) != len(blocks):
        raise ValueError("Cannot aggregate missing or partial Retcon diagnostics")
    shots = sum(row["data"]["gun"]["bulletsFired"] for row in complete)
    hits = sum(row["data"]["gun"]["bulletHits"] for row in complete)
    virtual = {}
    for row in complete:
        for gun in row["virtualGuns"]:
            total = virtual.setdefault(gun["model"], dict(hits=0, shots=0))
            total["hits"] += gun["hits"]
            total["shots"] += gun["shots"]
    for row in virtual.values():
        row["hitRate"] = row["hits"] / row["shots"] if row["shots"] else None
    return dict(capturesComplete=len(complete), roundsRecorded=sum(row["data"]["roundsRecorded"] for row in complete),
                actualGun=dict(shots=shots, hits=hits, hitRate=hits / shots if shots else None), virtualGuns=virtual,
                waves={key: sum(row["data"]["waves"][key] for row in complete) for key in
                       ("firesDetected", "unanchoredShots", "hitsTaken", "hitsOnWave", "wavePowerCorrected",
                        "waveHitsExact", "waveHitsOffByOneStep", "waveHitsOtherResidual")},
                movement={key: sum(row["data"]["movement"][key] for row in complete) for key in
                          ("moveTurnsChecked", "moveMatches", "moveWallHitsChecked", "moveNotChecked")},
                skippedTurns=sum(row["data"]["skippedTurns"] for row in complete),
                overBudgetTurns=sum(row["data"]["overBudgetTurns"] for row in complete))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--java", default="java")
    parser.add_argument("--bot-dir", type=Path, default=benchmark.ROOT / "AISurvivor")
    parser.add_argument("--cache", type=Path, default=benchmark.ROOT / ".benchmark-cache")
    parser.add_argument("--output", type=Path, default=benchmark.ROOT / "build" / "retcon-observability")
    parser.add_argument("--expected-bot-sha256", required=True)
    parser.add_argument("--expected-ubuntu", help="Require the actual Linux os-release Ubuntu VERSION_ID")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--seeds", default="20261102,20261103,20261104")
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    operating_system = retcon_stats.os_provenance()
    if args.expected_ubuntu:
        release = operating_system.get("osRelease", {})
        if release.get("ID") != "ubuntu" or release.get("VERSION_ID") != args.expected_ubuntu:
            raise ValueError("Diagnostic host does not match the explicitly required Ubuntu release")
    args.bot_dir, args.output = args.bot_dir.resolve(), args.output.resolve()
    if args.rounds < 1 or args.timeout < 1:
        parser.error("rounds and timeout must be positive")
    seeds = [int(value) for value in args.seeds.split(",")]
    if not seeds or len(set(seeds)) != len(seeds):
        parser.error("supply distinct block seeds")
    actual = benchmark.digest(args.bot_dir / "AISurvivor.jar")
    if actual != args.expected_bot_sha256:
        raise ValueError("Diagnostic AISurvivor JAR differs from the explicitly approved baseline")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("output must be empty")
    args.output.mkdir(parents=True, exist_ok=True)
    args.retcon_source, api_cache, provenance = retcon.prepare(args.cache.resolve())
    args.opponent_env = {"RUMBLE_JAVA_CACHE": str(api_cache)}
    args.require_metrics = args.require_retcon_stats = True
    args.training_rounds = 0
    runner_name = "robocode-tankroyale-runner-1.4.0.jar"
    runner = benchmark.download(args.cache.resolve(), runner_name, benchmark.ASSETS[runner_name])
    version = subprocess.run([args.java, "-version"], capture_output=True, text=True, timeout=15, check=True)
    report = dict(schemaVersion=1, purpose="unchanged-opponent observability; no gameplay change",
                  sourceCommit=os.environ.get("GITHUB_SHA"), botJarSHA256=actual, opponentProvenance=provenance,
                  runtimeOS=operating_system,
                  runtimeJava=(version.stderr + version.stdout).strip(), roundsPerBlock=args.rounds,
                  blocks=[], limitations="Retcon's own observed counters, not causal AISurvivor defect frequencies. "
                  "Virtual shots are completed wave evaluations and differ from actual fired bullets. "
                  "Starting positions, opponent randomness, and scheduling are unpaired and unseeded.")
    for number, seed in enumerate(seeds, 1):
        args.seed = seed
        output = args.output / f"block-{number}"
        output.mkdir()
        result = benchmark.battle(args, output, runner, None, "Retcon", ["Retcon"], "fixed-minrisk",
                                  args.rounds, output / "never-loaded.dat")
        report["blocks"].append(dict(block=number, seed=seed, **result))
        report["aggregateRetconStats"] = aggregate(report["blocks"])
        (args.output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"Retcon diagnostic block {number}: {args.rounds} rounds, native stats complete", flush=True)


if __name__ == "__main__":
    main()
