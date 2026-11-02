#!/usr/bin/env python3
"""Five registered fixed-policy integration rounds, not a strength experiment."""
import argparse
import json
from pathlib import Path
import subprocess
import benchmark
import retcon
import retcon_stats


def run(args):
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Sanity output must be empty")
    output.mkdir(parents=True, exist_ok=True)
    cache = args.cache.resolve()
    runner = benchmark.download(cache, "robocode-tankroyale-runner-1.4.0.jar", benchmark.ASSETS["robocode-tankroyale-runner-1.4.0.jar"])
    archive = benchmark.download(cache, "sample-bots-Java-1.4.0.zip", benchmark.ASSETS["sample-bots-Java-1.4.0.zip"])
    samples = output / "official-samples"
    benchmark.extract(archive, samples)
    source, api_cache, opponent = retcon.prepare(cache)
    java = subprocess.run([args.java, "-XshowSettings:properties", "-version"], capture_output=True, text=True, timeout=15, check=True)
    plan = subprocess.check_output(["git", "show", "c7f64d3be25220d4574f1b5b31bfe9d8bc6c6db7:docs/targeting-refactor.md"], cwd=benchmark.ROOT)
    (output / "registered-plan.md").write_bytes(plan)
    jar_hash = benchmark.digest(args.bot_dir / "AISurvivor.jar")
    report = dict(schemaVersion=1, status="running", purpose="refactor compatibility; no strength inference",
                  sourceCommit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=benchmark.ROOT, text=True).strip(),
                  characterizationCommit="74bcb6f8e79a97513c93a77425d8e631ba34cc64",
                  planCommit="c7f64d3be25220d4574f1b5b31bfe9d8bc6c6db7", jarSHA256=jar_hash,
                  planSHA256=benchmark.digest(output / "registered-plan.md"),
                  runnerSHA256=benchmark.digest(runner), sampleArchiveSHA256=benchmark.digest(archive),
                  opponentProvenance=opponent, runtimeJava=java.stdout + java.stderr,
                  runtimeOS=retcon_stats.os_provenance(), plannedRounds=5, validatedRounds=0, battles=[])
    path = output / "sanity.json"
    try:
        for name, rounds in (("Retcon", 2), ("Target", 1), ("SpinBot", 1), ("Walls", 1)):
            if benchmark.digest(args.bot_dir / "AISurvivor.jar") != jar_hash:
                raise ValueError("Candidate JAR changed during the sanity suite")
            options = argparse.Namespace(java=args.java, bot_dir=args.bot_dir, seed=20261102,
                timeout=180, training_rounds=20, require_metrics=True, require_retcon_stats=name == "Retcon",
                opponent_env={"RUMBLE_JAVA_CACHE": str(api_cache)}, retcon_source=source)
            game = benchmark.battle(options, output, runner, samples, name, [name], "fixed-minrisk", rounds,
                                    output / (name + "-absent-model.dat"))
            report["battles"].append(game)
            report["validatedRounds"] += rounds
            if game["metrics"]["skippedTurns"] != 0 or game["modelSHA256"] is not None:
                raise ValueError("Refactor sanity requires zero skipped turns and no fixed-mode model")
            path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        if report["validatedRounds"] != 5:
            raise ValueError("Incomplete registered five-round budget")
        report.update(status="complete", decision="compatibility passed; deterministic parity remains primary")
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--java", default="java")
    parser.add_argument("--bot-dir", type=Path, default=benchmark.ROOT / "AISurvivor")
    parser.add_argument("--cache", type=Path, default=benchmark.ROOT / ".benchmark-cache")
    parser.add_argument("--output", type=Path, default=benchmark.ROOT / "build" / "targeting-sanity")
    run(parser.parse_args())
