#!/usr/bin/env python3
"""Pinned, isolated Tank Royale experiments; Python 3.10+, Java 17+, no pip dependencies."""
import argparse
import csv
import hashlib
import html
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import urllib.request
import zipfile
import retcon
import retcon_stats

VERSION = "1.4.0"
ASSETS = {
    "robocode-tankroyale-runner-1.4.0.jar": "02f6d1d8e9346a4aeae1b91e7abb278c98ac1f1d30c605f867758566f5f327cf",
    "sample-bots-Java-1.4.0.zip": "1707c1011a37c6b854fc79d68549ab204007bac6d33d803fa254152ac0765fe2",
}
ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else hashlib.sha256(stream.read()).hexdigest()


def download(cache, name, expected):
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / name
    if not target.exists() or digest(target) != expected:
        temporary = target.with_suffix(target.suffix + ".part")
        request = urllib.request.Request(f"https://github.com/robocode-dev/tank-royale/releases/download/v{VERSION}/{name}", headers={"User-Agent": "AISurvivor-benchmark"})
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as stream:
            shutil.copyfileobj(response, stream)
        if digest(temporary) != expected:
            temporary.unlink()
            raise ValueError(f"SHA256 mismatch: {name}")
        temporary.replace(target)
    return target


def extract(archive, destination):
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zipped:
        for entry in zipped.infolist():
            target = (destination / entry.filename).resolve()
            if not target.is_relative_to(destination.resolve()) or "\\" in entry.filename:
                raise ValueError(f"Unsafe archive path: {entry.filename}")
            if (entry.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Symlinks are not supported in sample archives")
        zipped.extractall(destination)
    for script in destination.rglob("*.sh"):
        script.chmod(script.stat().st_mode | 0o111)


def run_process(command, cwd, env, log, timeout):
    """Use argv, bound execution, and kill only this process group on POSIX timeout."""
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=(os.name != "nt"))
        try:
            code = process.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            if os.name != "nt":
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
            else:
                # Windows TerminateProcess skips JVM shutdown hooks: terminate the owned tree.
                subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                               stdout=stream, stderr=subprocess.STDOUT, timeout=15, check=False)
                if process.poll() is None:
                    process.kill()
            process.wait(timeout=15)
            raise
        if code:
            tail = "\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-40:])
            raise RuntimeError(f"Command exited {code}; see {log}\n{tail}")


def isolate_opponent(source, destination):
    shutil.copytree(source, destination)
    # Official Java samples launch source with ../lib/*; retain their shared runtime.
    shared = source.parent / "lib"
    if shared.is_dir():
        shutil.copytree(shared, destination.parent / "lib", dirs_exist_ok=True)


def summarize_metrics(records):
    keys = ("turns", "shotsFired", "bulletHits", "damageDealt", "damageTaken", "wallCollisions", "botCollisions", "skippedTurns")
    result = {key: sum(row.get(key, 0) for row in records) for key in keys}
    result["hitRate"] = result["bulletHits"] / result["shotsFired"] if result["shotsFired"] else None
    result["roundsRecorded"] = len(records)
    result["checkpointLoaded"] = any(row.get("checkpointLoaded", False) for row in records)
    return result


def validate_metrics(records, rounds, policy, seed, training_rounds):
    """Reject stale, duplicated, wrong-policy, or non-finite experiment evidence."""
    if len(records) != rounds:
        raise ValueError(f"Expected {rounds} telemetry rows, got {len(records)}")
    expected_mode = "train" if policy == "train" else "fixed" if policy.startswith("fixed") else "eval"
    round_ids = []
    counters = ("turns", "totalRounds", "shotsFired", "bulletHits", "wallCollisions", "botCollisions", "skippedTurns")
    for row in records:
        if not isinstance(row, dict) or type(row.get("schemaVersion")) is not int or row["schemaVersion"] != 1:
            raise ValueError("Unsupported telemetry schema")
        if type(row.get("round")) is not int:
            raise ValueError("Round ID must be an integer")
        round_ids.append(row["round"])
        if row.get("mode") != expected_mode or type(row.get("seed")) is not int or row["seed"] != seed:
            raise ValueError("Telemetry mode or seed does not match requested experiment")
        if type(row.get("checkpointLoaded")) is not bool or row["checkpointLoaded"] != (policy == "trained"):
            raise ValueError("Telemetry checkpoint status does not match cold/trained policy")
        expected_action = 0 if expected_mode == "fixed" else None
        if "action" not in row or row["action"] != expected_action or (expected_action is not None and type(row["action"]) is not int):
            raise ValueError("Telemetry action does not match requested policy")
        for key in counters:
            if type(row.get(key)) is not int or row[key] < 0:
                raise ValueError(f"Telemetry {key} must be a non-negative integer")
        for key in ("damageDealt", "damageTaken", "reward", "hitRate"):
            if type(row.get(key)) not in (int, float) or not math.isfinite(row[key]):
                raise ValueError(f"Telemetry {key} must be finite")
        if min(row["damageDealt"], row["damageTaken"]) < 0 or row["bulletHits"] > row["shotsFired"]:
            raise ValueError("Invalid damage or hit counts")
        expected_rate = row["bulletHits"] / row["shotsFired"] if row["shotsFired"] else 0
        if not math.isclose(row["hitRate"], expected_rate, abs_tol=0.000001):
            raise ValueError("Telemetry hit rate disagrees with hit/shot counts")
        expected_total = row["round"] if policy == "train" else training_rounds if policy == "trained" else 0
        if row["totalRounds"] != expected_total:
            raise ValueError("Telemetry training budget does not match requested experiment")
        if any(type(row.get(key)) is not bool for key in ("won", "died")) or (row["won"] and row["died"]):
            raise ValueError("Invalid round outcome flags")
    if set(round_ids) != set(range(1, rounds + 1)):
        raise ValueError("Telemetry round IDs must cover each requested round exactly once")


def write_reports(output, report):
    (output / "results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    fields = ["scenario", "policy", "rounds", "rank", "totalScore", "survival", "firstPlaces", "hitRate", "wallCollisions", "skippedTurns"]
    with (output / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        for battle in report["battles"]:
            writer.writerow({key: battle.get(key, battle.get("metrics", {}).get(key, "")) for key in fields})
    rows = []
    for battle in report["battles"]:
        values = [battle.get(key, battle.get("metrics", {}).get(key, "")) for key in fields]
        values[fields.index("hitRate")] = f"{battle['metrics']['hitRate']:.1%}" if battle.get("metrics", {}).get("hitRate") is not None else "—"
        rows.append("<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in values) + "</tr>")
    labels = {"scenario": "Scenario", "policy": "Policy", "rounds": "Rounds", "rank": "Rank", "totalScore": "Score", "survival": "Survival", "firstPlaces": "Wins", "hitRate": "Hit rate", "wallCollisions": "Walls", "skippedTurns": "Skipped"}
    headers = "".join(f"<th>{labels[field]}</th>" for field in fields)
    notes = html.escape(report["limitations"])
    page = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>AISurvivor battle results</title>
<style>body{{font:16px system-ui;background:#111827;color:#e5e7eb;max-width:1200px;margin:40px auto;padding:0 20px}}h1{{color:#67e8f9}}.scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%;background:#1f2937}}th,td{{padding:12px;text-align:left;border-bottom:1px solid #374151}}th{{color:#a5f3fc}}p{{line-height:1.6}}input{{padding:10px;font:inherit;margin:12px 0;width:260px}}</style>
<h1>AISurvivor battle results</h1><p>Tank Royale {VERSION} · seed {report['seed']} · training budget {report['trainingRounds']} rounds per scenario</p>
<p>{notes}</p><p>Learning comparison: compare <strong>untrained</strong> with <strong>trained</strong> for the same scenario. Training rows show the separate training budget, not evaluation strength.</p><label>Filter scenarios or policies <input id="filter" placeholder="e.g. trained or Walls"></label><div class="scroll"><table><thead><tr>{headers}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<p>Scores are comparable within a scenario and round budget. Downloaded JSON/CSV retain full experiment metadata. Missing telemetry is not a zero measurement.</p>
<script>document.getElementById('filter').addEventListener('input',e=>{{for(const r of document.querySelectorAll('tbody tr'))r.hidden=!r.textContent.toLowerCase().includes(e.target.value.toLowerCase())}});</script></html>"""
    (output / "dashboard.html").write_text(page, encoding="utf-8")


def battle(args, output, runner, samples, scenario, opponents, policy, rounds, model):
    run_dir = output / f"{scenario}-{policy}"
    bot_dir = run_dir / "AISurvivor"
    bot_dir.mkdir(parents=True)
    for name in ("AISurvivor.jar", "AISurvivor.json", "AISurvivor.sh", "AISurvivor.cmd"):
        shutil.copy2(args.bot_dir / name, bot_dir / name)
    (bot_dir / "AISurvivor.sh").chmod(0o755)
    metrics = run_dir / "metrics.jsonl"
    env = os.environ.copy()
    env.update(args.opponent_env)
    if Path(args.java).is_absolute():
        env["PATH"] = str(Path(args.java).parent) + os.pathsep + env.get("PATH", "")
    env.update(AISURVIVOR_MODE="train" if policy == "train" else "fixed" if policy.startswith("fixed") else "eval",
               AISURVIVOR_SEED=str(args.seed), AISURVIVOR_MODEL_PATH=str(model),
               AISURVIVOR_METRICS_PATH=str(metrics), AISURVIVOR_ACTION="0")
    opponent_dirs = []
    for name in opponents:
        configs = [args.retcon_source / "Retcon.json"] if name == "Retcon" and args.retcon_source else list(samples.rglob(name + ".json"))
        if len(configs) != 1:
            raise ValueError(f"Expected one {name} sample config, found {len(configs)}")
        if name == "Retcon" and retcon.source_hash(configs[0].parent) != retcon_stats.SOURCE_SHA256:
            raise ValueError("Observed Retcon source differs from the approved unchanged pin")
        isolated = run_dir / "opponents" / name
        isolate_opponent(configs[0].parent, isolated)
        opponent_dirs.append(str(isolated))
    result_file = run_dir / "battle.json"
    command = [args.java, "-cp", str(runner), str(ROOT / "tools" / "BenchmarkBattle.java"),
               str(rounds), str(args.timeout), str(result_file), str(bot_dir), *opponent_dirs]
    capture = retcon_stats.Capture(retcon_stats.java_temp_directory(args.java)) if "Retcon" in opponents else None
    env.pop("RETCON_STATS_DIR", None)
    if capture:
        env["RETCON_STATS_DIR"] = str(capture.path)
    opponent_stats = None
    opponent_id = None
    before = digest(model) if model.exists() else None
    try:
        run_process(command, ROOT, env, run_dir / "runner.log", args.timeout + 90)
        data = json.loads(result_file.read_text(encoding="utf-8"))
        if capture:
            native_opponents = [row for row in data["results"] if row["name"] == "Retcon"]
            if len(native_opponents) != 1:
                raise ValueError("Expected one Retcon result for native stats matching")
            if native_opponents[0].get("version") != "0.2.0":
                raise ValueError("Observed Retcon version differs from the approved source pin")
            opponent_id = native_opponents[0]["id"]
            opponent_stats = capture.collect(run_dir / "retcon-stats", rounds, opponent_id,
                                             require=getattr(args, "require_retcon_stats", False))
    finally:
        if capture:
            try:
                if opponent_stats is None:
                    capture.collect(run_dir / "retcon-stats", rounds, opponent_id, require=False)
            finally:
                capture.close()
    if data["numberOfRounds"] != rounds:
        raise ValueError("Battle round count does not match requested budget")
    expected_name = json.loads((bot_dir / "AISurvivor.json").read_text(encoding="utf-8"))["name"]
    candidates = [row for row in data["results"] if row["name"] == expected_name]
    if len(candidates) != 1 or len(data["results"]) != len(opponents) + 1:
        raise ValueError("Battle participant identities do not match the requested roster")
    candidate = candidates[0]
    records = [json.loads(line) for line in metrics.read_text(encoding="utf-8").splitlines() if line.strip()] if metrics.exists() else []
    if args.require_metrics or records:
        validate_metrics(records, rounds, policy, args.seed, args.training_rounds)
    if policy != "train" and (digest(model) if model.exists() else None) != before:
        raise ValueError("Evaluation modified its checkpoint")
    return dict(scenario=scenario, policy=policy, rounds=rounds, **candidate,
                metrics=summarize_metrics(records) if records else {}, results=data["results"],
                retconStats=opponent_stats if capture else dict(status="not-applicable"),
                modelSHA256=digest(model) if model.exists() else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="One cold evaluation round against the selected suite's first opponent")
    parser.add_argument("--suite", choices=("samples", "retcon"), default="samples",
                        help="Sample regression roster or pinned Retcon 0.2.0 1v1")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--training-rounds", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument("--timeout", type=int, default=180, help="Battle timeout in seconds")
    parser.add_argument("--output", type=Path, default=ROOT / "build" / "benchmark")
    parser.add_argument("--cache", type=Path, default=ROOT / ".benchmark-cache")
    parser.add_argument("--bot-dir", type=Path, default=ROOT / "AISurvivor")
    parser.add_argument("--java", default="java")
    parser.add_argument("--require-metrics", action="store_true")
    parser.add_argument("--require-retcon-stats", action="store_true", help="Fail if the native Retcon export is missing, partial, or invalid")
    args = parser.parse_args()
    if min(args.rounds, args.training_rounds, args.timeout) <= 0:
        parser.error("round and timeout values must be positive")
    if args.require_retcon_stats and args.suite != "retcon":
        parser.error("--require-retcon-stats requires --suite retcon")
    args.bot_dir = args.bot_dir.resolve()
    args.output = args.output.resolve()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("output directory must be empty to preserve prior experiment evidence")
    args.output.mkdir(parents=True, exist_ok=True)
    args.opponent_env = {}
    args.retcon_source = None
    opponent_provenance = []
    if args.suite == "retcon":
        args.retcon_source, api_cache, provenance = retcon.prepare(args.cache.resolve())
        args.opponent_env["RUMBLE_JAVA_CACHE"] = str(api_cache)
        opponent_provenance.append(provenance)
        runtime = subprocess.run([args.java, "-XshowSettings:properties", "-version"], capture_output=True, text=True, timeout=15)
        import re
        major = re.search(r"java\.specification\.version\s*=\s*(\d+)", runtime.stderr)
        if runtime.returncode or not major or int(major.group(1)) < 22:
            parser.error("Retcon's unchanged source launcher requires --java pointing to JDK 22 or newer (CI uses JDK25)")
    assets = {name: download(args.cache.resolve(), name, expected) for name, expected in ASSETS.items()}
    runner = assets["robocode-tankroyale-runner-1.4.0.jar"]
    samples = args.output / "samples"
    extract(assets["sample-bots-Java-1.4.0.zip"], samples)
    source_commit = os.environ.get("GITHUB_SHA")
    if not source_commit:
        try:
            source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            source_commit = None
    java_version = subprocess.run([args.java, "-version"], capture_output=True, text=True, timeout=15, check=True)
    report = dict(schemaVersion=1, tankRoyaleVersion=VERSION, sourceCommit=source_commit, seed=args.seed,
                  suite=args.suite, runtimeJava=(java_version.stderr + java_version.stdout).strip(),
                  opponentProvenance=opponent_provenance,
                  evaluationScope="in-roster adaptation; training and evaluation use the same opponent identities",
                  trainingRounds=0 if args.smoke else args.training_rounds,
                  botJarSHA256=digest(args.bot_dir / "AISurvivor.jar"), assetSHA256=ASSETS,
                  limitations="Reference experiment, not evidence that learning improves strength. AISurvivor RNG is seeded; opponent randomness, arena starting positions, and engine scheduling are not seeded. Training and evaluation use the same roster, so this is in-roster adaptation rather than held-out generalization. Short samples have high uncertainty; repeat with larger budgets and multiple seeds before drawing strength conclusions.", battles=[])
    scenarios = {"Retcon": ["Retcon"]} if args.suite == "retcon" else {"Target": ["Target"]} if args.smoke else {"Target": ["Target"], "SpinBot": ["SpinBot"], "Walls": ["Walls"], "melee": ["Target", "SpinBot", "Walls"]}
    for name, opponents in scenarios.items():
        policies = ["untrained"] if args.smoke else ["fixed-minrisk", "untrained", "train", "trained"]
        for policy in policies:
            model = args.output / f"{name}-{'trained' if policy in ('train', 'trained') else policy}.dat"
            result = battle(args, args.output, runner, samples, name, opponents, policy,
                            args.training_rounds if policy == "train" else 1 if args.smoke else args.rounds, model)
            if policy == "trained" and not model.exists():
                raise ValueError("Training did not produce a checkpoint")
            report["battles"].append(result)
            write_reports(args.output, report)
            print(f"{name} {policy}: rank={result['rank']} score={result['totalScore']}", flush=True)


if __name__ == "__main__":
    main()
