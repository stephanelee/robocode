#!/usr/bin/env python3
"""Acquire the exact registered 45-round legal-motion pilot; no fitting or bot changes."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import subprocess
import benchmark
import legal_scan_dataset
import retcon
import retcon_stats


def blob(commit, path):
    if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise ValueError("Registration requires full immutable commit")
    return subprocess.check_output(["git", "show", commit + ":" + path], cwd=benchmark.ROOT)


def run(args):
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Fresh output required; never replace a previous attempt")
    output.mkdir(parents=True, exist_ok=True)
    confirmation = getattr(args, "confirmation", False)
    wave_context = getattr(args, "wave_context", False)
    collector_retry = getattr(args, "collector_retry", False)
    if collector_retry and not (confirmation and wave_context):
        raise ValueError("Collector repair is separately registered v3 confirmation only")
    if wave_context and not confirmation:
        raise ValueError("Old-wave context acquisition is fresh confirmation only")
    plan_path = "docs/ai-old-wave-context-v3.md" if wave_context else "docs/ai-containment-proposal.md" if confirmation else "docs/ai-motion-pilot.md"
    layout_path = "tools/fixtures/own-wave-v3/games.json" if wave_context else "tools/fixtures/containment-v2/games.json" if confirmation else "tools/fixtures/motion-pilot-v1/games.json"
    if collector_retry:
        layout_path = "tools/fixtures/own-wave-v3/retry-games.json"
    plan = blob(args.plan_commit, plan_path)
    layout_bytes = blob(args.plan_commit, layout_path)
    layout = json.loads(layout_bytes)
    games = layout["games"]
    expected_games, expected_rounds = (6, 18) if confirmation else (15, 45)
    expected_splits = dict(withheld=3, transfer=3) if confirmation else dict(train=6, dev=3, withheld=3, transfer=3)
    if (layout["roundsPerGame"] != 3 or len(games) != expected_games
            or collections.Counter(g["split"] for g in games) != expected_splits
            or layout["horizons"] != [8, 16, 24, 32, 40] or layout["history"] != 8):
        raise ValueError("Registration differs from reviewed exact pilot budget/split")
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=benchmark.ROOT, text=True).strip()
    jar_hash = benchmark.digest(args.bot_dir / "AISurvivor.jar")
    dataset_id = "ais-own-wave-confirm-v3" if wave_context else "ais-containment-confirm-v2" if confirmation else "ais-motion-pilot-v1"
    manifest = dict(schemaVersion=1, datasetId=dataset_id, provenance=dict(sourceCommit=source_commit,
        botJarSha256=jar_hash, engineVersion=benchmark.VERSION, collectorVersion="legal-scan-v1.1-closed-round"), games=games)
    legal_scan_dataset.validate_manifest(manifest)
    if source_commit == "0" * 40 or jar_hash == "0" * 64:
        raise ValueError("Synthetic provenance is forbidden for real acquisition")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output / "registered-plan.md").write_bytes(plan)
    (output / "registered-games.json").write_bytes(layout_bytes)
    report = dict(schemaVersion=1, status="preparing", sourceCommit=source_commit, botJarSha256=jar_hash,
        registrationCommit=args.plan_commit, planSHA256=hashlib.sha256(plan).hexdigest(),
        layoutSHA256=hashlib.sha256(layout_bytes).hexdigest(), plannedGames=expected_games, plannedRounds=expected_rounds,
        validatedGames=0, validatedRounds=0, games=[], runtimeOS=retcon_stats.os_provenance())
    path = output / "acquisition.json"
    def save():
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    save()
    all_records = []
    try:
        runtime = subprocess.run([args.java, "-XshowSettings:properties", "-version"], capture_output=True, text=True, timeout=15, check=True)
        report["runtimeJava"] = runtime.stdout + runtime.stderr
        cache = args.cache.resolve()
        runner = benchmark.download(cache, "robocode-tankroyale-runner-1.4.0.jar", benchmark.ASSETS["robocode-tankroyale-runner-1.4.0.jar"])
        archive = benchmark.download(cache, "sample-bots-Java-1.4.0.zip", benchmark.ASSETS["sample-bots-Java-1.4.0.zip"])
        samples = output / "official-samples"; benchmark.extract(archive, samples)
        opponent, api_cache, provenance = retcon.prepare(cache)
        report.update(runnerSHA256=benchmark.digest(runner), sampleArchiveSHA256=benchmark.digest(archive),
                      opponentProvenance=provenance, status="running")
        for game in games:
            if benchmark.digest(args.bot_dir / "AISurvivor.jar") != jar_hash:
                raise ValueError("Frozen collector JAR changed")
            name, game_id = game["opponent"], game["gameId"]
            game_output = output / "games" / game_id; game_output.mkdir(parents=True)
            events = game_output / "events.jsonl"
            options = argparse.Namespace(java=args.java, bot_dir=args.bot_dir, seed=game["aisSeed"], timeout=600,
                training_rounds=20, require_metrics=True, require_retcon_stats=name == "Retcon", retcon_source=opponent,
                opponent_env=dict(RUMBLE_JAVA_CACHE=str(api_cache), AISURVIVOR_DATASET_PATH=str(events), AISURVIVOR_GAME_ID=game_id))
            report["inProgressGame"] = game_id; save()
            measured = benchmark.battle(options, game_output, runner, samples, name, [name], "fixed-minrisk", 3,
                                        game_output / "absent-model.dat")
            if measured["metrics"]["skippedTurns"] or measured["modelSHA256"] is not None:
                raise ValueError("Data acquisition requires zero AIS skipped turns and no model writes")
            status = json.loads(Path(str(events) + ".status.json").read_text())
            records = list(legal_scan_dataset.load_events(events))
            if (status["status"] != "complete" or status["gameId"] != game_id or status["error"] is not None
                    or status["collectorVersion"] != manifest["provenance"]["collectorVersion"]
                    or {r["round"] for r in status["rounds"]} != {1, 2, 3}
                    or len(status["rounds"]) != 3 or any(not r["complete"] for r in status["rounds"])
                    or {r["round"] for r in records} != {1, 2, 3}
                    or any(r["gameId"] != game_id for r in records)):
                raise ValueError("Incomplete/invalid real producer capture")
            legal_scan_dataset.validated_groups(records, manifest)
            for round_status in status["rounds"]:
                if sum(row["round"] == round_status["round"] for row in records) != round_status["records"]:
                    raise ValueError("Producer row counts disagree")
            all_records.extend(records)
            if len(all_records) > legal_scan_dataset.MAX_RECORDS:
                raise ValueError("Total record budget exceeded")
            with (output / "events.jsonl").open("ab") as joined:
                joined.write(events.read_bytes())
            report["games"].append(dict(gameId=game_id, split=game["split"], records=len(records),
                eventsSHA256=benchmark.digest(events), statusSHA256=benchmark.digest(Path(str(events) + ".status.json")), battle=measured))
            report["validatedGames"] += 1; report["validatedRounds"] += 3
            del report["inProgressGame"]; save()
            print(f"CAPTURED {game_id}: 3 rounds, {len(records)} legal rows", flush=True)
        if report["validatedRounds"] != expected_rounds or report["validatedGames"] != expected_games:
            raise ValueError("Incomplete declared pilot budget")
        legal_scan_dataset.validated_groups(all_records, manifest)
        report.update(status="complete", eventsSHA256=benchmark.digest(output / "events.jsonl"), inputRecords=len(all_records))
    except Exception as error:
        report.update(status="failed", error=str(error)); raise
    finally:
        save(); print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-commit", required=True)
    parser.add_argument("--confirmation", action="store_true", help="Only the separately registered six-game/18-round fresh containment confirmation")
    parser.add_argument("--wave-context", action="store_true", help="Use only the separately registered v3 fresh IDs/plan")
    parser.add_argument("--collector-retry", action="store_true", help="Only separately reviewed post-failure v3 IDs with the repaired collector")
    parser.add_argument("--java", default="java")
    parser.add_argument("--bot-dir", type=Path, default=benchmark.ROOT / "AISurvivor")
    parser.add_argument("--cache", type=Path, default=benchmark.ROOT / ".benchmark-cache")
    parser.add_argument("--output", type=Path, default=benchmark.ROOT / "build" / "ai-pilot")
    run(parser.parse_args())
