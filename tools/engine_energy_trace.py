#!/usr/bin/env python3
"""Record the real pinned engine's bullet/scan contract with two stationary probes."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import benchmark


def validate(records):
    """Derive visibility assertions from observed scans, not callback delivery order."""
    evidence = []
    hits = sorted((row for row in records if row["type"] == "hit" and row["power"] == 0.5), key=lambda row: row["turn"])
    if len(hits) != 2 or {row["label"] for row in hits} != {"Alpha", "Beta"}:
        raise ValueError("Expected exactly one isolated power-0.5 hit from each probe")
    for hit in hits:
        turn, shooter, victim = hit["turn"], hit["self"], hit["enemy"]
        received = [row for row in records if row["type"] == "hitBy" and row["turn"] == turn
                    and row["self"] == victim and row["enemy"] == shooter and row["bullet"] == hit["bullet"]]
        if len(received) != 1 or hit["damage"] != 2 or received[0]["damage"] != 2:
            raise ValueError("Bullet damage must be observed as 2 by both probes")
        for observer, enemy, effect in ((shooter, victim, -2.0), (victim, shooter, 1.5)):
            scans = {row["turn"]: row for row in records if row["type"] == "scan"
                     and row["self"] == observer and row["enemy"] == enemy}
            if not all(t in scans for t in (turn - 1, turn, turn + 1)):
                raise ValueError("Missing consecutive scans around actual hit")
            before, current, after = [scans[t]["energy"] for t in (turn - 1, turn, turn + 1)]
            if not all(math.isfinite(value) for value in (before, current, after)):
                raise ValueError("Nonfinite scan energy")
            if not math.isclose(current, before, abs_tol=1e-9) or not math.isclose(after - current, effect, abs_tol=1e-9):
                raise ValueError("Recorded scan energies do not establish k+1 bullet visibility")
            evidence.append(dict(hitTurn=turn, observer=observer, enemy=enemy, effect=effect,
                                 previousScan=before, sameTurnScan=current, nextTurnScan=after))
    return evidence


def run(args):
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Trace output must be empty")
    output.mkdir(parents=True, exist_ok=True)
    cache = args.cache.resolve()
    runner = benchmark.download(cache, next(name for name in benchmark.ASSETS if name.endswith(".jar")),
                                benchmark.ASSETS["robocode-tankroyale-runner-1.4.0.jar"])
    archive = benchmark.download(cache, "sample-bots-Java-1.4.0.zip", benchmark.ASSETS["sample-bots-Java-1.4.0.zip"])
    benchmark.extract(archive, output / "official-samples")
    apis = list((output / "official-samples").rglob("robocode-tankroyale-bot-api-1.4.0.jar"))
    if len(apis) != 1:
        raise ValueError("Expected one official pinned Bot API")
    source = benchmark.ROOT / "tools" / "EngineEnergyProbe.java"
    bots = []
    for label in ("Alpha", "Beta"):
        bot = output / label
        bot.mkdir()
        shutil.copy2(source, bot / source.name)
        shutil.copy2(apis[0], bot / "bot-api.jar")
        (bot / (label + ".json")).write_text(json.dumps(dict(name=label, version="1.0", authors=["stephanelee"],
            gameTypes=["1v1"], platform="JVM", programmingLang="Java 17")), encoding="utf-8")
        script = bot / (label + ".sh")
        script.write_text(f'#!/bin/sh\nexec java -cp bot-api.jar EngineEnergyProbe.java {label}\n', encoding="utf-8")
        script.chmod(0o755)
        (bot / (label + ".cmd")).write_text(f'@java -cp bot-api.jar EngineEnergyProbe.java {label}\n', encoding="utf-8")
        bots.append(bot)
    runtime = subprocess.run([args.java, "-XshowSettings:properties", "-version"], capture_output=True, text=True, check=True)
    report = dict(schemaVersion=1, status="running", engineVersion=benchmark.VERSION,
                  upstreamSource="https://github.com/robocode-dev/tank-royale/tree/v1.4.0",
                  runnerSHA256=benchmark.digest(runner), sampleArchiveSHA256=benchmark.digest(archive),
                  botApiSHA256=benchmark.digest(apis[0]), probeSourceSHA256=benchmark.digest(source),
                  recorderSHA256=benchmark.digest(Path(__file__)),
                  runnerAdapterSHA256=benchmark.digest(benchmark.ROOT / "tools" / "BenchmarkBattle.java"),
                  sourceCommit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=benchmark.ROOT, text=True).strip(),
                  javaRuntime=runtime.stdout + runtime.stderr,
                  purpose="Actual event recording; no AISurvivor or Retcon gameplay is loaded")
    path = output / "trace.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    env = os.environ.copy()
    env["ENGINE_TRACE_DIR"] = str(output)
    if Path(args.java).is_absolute():
        env["PATH"] = str(Path(args.java).parent) + os.pathsep + env.get("PATH", "")
    try:
        # Embedded official runner allocates its own server; never connects to interactive port 7654.
        benchmark.run_process([args.java, "-cp", str(runner), str(benchmark.ROOT / "tools" / "BenchmarkBattle.java"),
                               "1", "180", str(output / "battle.json"), *map(str, bots)],
                              benchmark.ROOT, env, output / "runner.log", 240)
        records = [json.loads(line) for label in ("Alpha", "Beta")
                   for line in (output / (label + ".jsonl")).read_text(encoding="utf-8").splitlines() if line.strip()]
        report["assertions"] = validate(records)
        report["rawSHA256"] = {label: benchmark.digest(output / (label + ".jsonl")) for label in ("Alpha", "Beta")}
        report["status"] = "verified"
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
    parser.add_argument("--cache", type=Path, default=benchmark.ROOT / "build" / "benchmark-cache")
    parser.add_argument("--output", type=Path, default=benchmark.ROOT / "build" / "engine-trace")
    run(parser.parse_args())
