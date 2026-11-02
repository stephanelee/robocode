import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import retcon_stats
import retcon_diagnostics
import benchmark


def fixture(rounds=5):
    return dict(bot="Retcon", botId=2, roundsRecorded=rounds, turnTimeoutMicros=100000,
                computeBudgetMicros=50000, turns=1000, computeP50Micros=10, computeP99Micros=80,
                computeMaxMicros=75, overBudgetTurns=0, skippedTurns=0, skippedPerRound=[0] * (rounds + 1),
                skippedTurnList="", computeHistogram10us="0:900 7:100",
                timing=dict(firesChecked=100, firesNotCreated=0, createdTurnMatches=100, originMatches=0,
                            originMatchesAfterFirstMove=100, headingMatches=100, stillHitsChecked=0,
                            stillHitMatches=0, hitsOnMovingTargets=12, mismatches=""),
                waves=dict(firesDetected=80, unanchoredShots=5, hitsTaken=20, hitsOnWave=18,
                           wavePowerCorrected=1, waveHitsExact=18, waveHitsOffByOneStep=0,
                           waveHitsOtherResidual=0, waveMismatches=""),
                movement=dict(moveTurnsChecked=900, moveMatches=899, moveWallHitsChecked=2,
                              moveNotChecked=100, moveMismatches="turn 9 mismatch"),
                gun=dict(bulletsFired=100, bulletHits=12, virtualGuns="LinearAimModel:20/70 HeadOnAimModel:10/70"))


class RetconStatsTests(unittest.TestCase):
    def test_retained_cohorts_preserve_original_native_hashes_and_ubuntu_provenance(self):
        root = Path(__file__).resolve().parents[1] / "benchmarks/retcon-observability-2026-11-02"
        for number in (1, 2):
            cohort = root / f"cohort-{number}"
            report = json.loads((cohort / "results.json").read_text(encoding="utf-8"))
            self.assertEqual(report["runtimeOS"]["osRelease"]["VERSION_ID"], "26.04")
            self.assertEqual(report["botJarSHA256"], "0d120a0461a7bf32b1c6d6277da34f9480309f255caa44e168e6bcdf8636bff2")
            self.assertEqual(len(report["blocks"]), 3)
            for block in report["blocks"]:
                capture = block["retconStats"]
                self.assertEqual(capture["status"], "complete")
                native_root = cohort / f"block-{block['block']}" / "Retcon-fixed-minrisk/retcon-stats"
                for artifact in capture["rawArtifacts"]:
                    raw = (native_root / artifact["path"]).read_bytes()
                    self.assertEqual(hashlib.sha256(raw).hexdigest(), artifact["sha256"])
                    self.assertEqual(json.loads(raw), capture["data"])
            self.assertEqual(retcon_diagnostics.aggregate(report["blocks"]), report["aggregateRetconStats"])

    def test_existing_namespace_caller_captures_without_new_cli_attribute(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = root / "candidate"
            candidate.mkdir()
            for name in ("AISurvivor.jar", "AISurvivor.sh", "AISurvivor.cmd"):
                (candidate / name).write_text("fixture")
            (candidate / "AISurvivor.json").write_text('{"name":"AISurvivor"}')
            source = root / "Retcon"
            source.mkdir()
            (source / "Retcon.json").write_text('{"name":"Retcon"}')
            options = SimpleNamespace(bot_dir=candidate, java="selected-java", opponent_env={},
                retcon_source=source, require_metrics=False, training_rounds=0, seed=42, timeout=1)
            temporary_paths = []
            def run(command, cwd, env, log, timeout):
                native = Path(env["RETCON_STATS_DIR"])
                temporary_paths.append(native)
                (native / "Retcon-2-100.json").write_text(json.dumps(fixture(1)))
                Path(command[6]).write_text(json.dumps(dict(numberOfRounds=1, results=[
                    dict(id=1, name="AISurvivor", version="1", rank=2, totalScore=1, firstPlaces=0),
                    dict(id=2, name="Retcon", version="0.2.0", rank=1, totalScore=2, firstPlaces=1)])))
            with patch.object(benchmark.retcon_stats, "java_temp_directory", return_value=root), \
                    patch.object(benchmark.retcon, "source_hash", return_value=retcon_stats.SOURCE_SHA256), \
                    patch.object(benchmark, "run_process", side_effect=run):
                result = benchmark.battle(options, root / "output", root / "runner.jar", None,
                    "Retcon", ["Retcon"], "fixed-minrisk", 1, root / "never-loaded.dat")
            self.assertEqual(result["retconStats"]["status"], "complete")
            self.assertTrue((root / "output/Retcon-fixed-minrisk/retcon-stats/capture.json").exists())
            self.assertFalse(temporary_paths[0].exists())

    def test_aggregation_counts_each_game_final_snapshot_once(self):
        data = fixture()
        capture = dict(status="complete", data=data, **retcon_stats.validate(data, 5, 2))
        blocks = [dict(retconStats=copy.deepcopy(capture)) for _ in range(3)]
        total = retcon_diagnostics.aggregate(blocks)
        self.assertEqual(total["roundsRecorded"], 15)
        self.assertEqual(total["actualGun"], dict(shots=300, hits=36, hitRate=0.12))
        self.assertEqual(total["virtualGuns"]["LinearAimModel"]["shots"], 210)
        blocks[0]["retconStats"]["status"] = "partial"
        with self.assertRaisesRegex(ValueError, "missing or partial"):
            retcon_diagnostics.aggregate(blocks)

    def test_unique_direct_java_temp_children_and_owned_cleanup(self):
        with tempfile.TemporaryDirectory(prefix="java-temp-space ") as directory:
            first, second = retcon_stats.Capture(directory), retcon_stats.Capture(directory)
            self.assertNotEqual(first.path, second.path)
            self.assertEqual(first.path.parent, Path(directory))
            unrelated = Path(directory) / "unrelated.txt"
            unrelated.write_text("preserve")
            first.close()
            self.assertTrue(second.path.exists())
            second.close()
            self.assertEqual(unrelated.read_text(), "preserve")

    def test_java_temp_root_comes_from_selected_jvm(self):
        with tempfile.TemporaryDirectory() as directory:
            result = type("Result", (), {"stderr": "Property settings:\n    java.io.tmpdir = " + directory + "\n"})()
            with patch.object(retcon_stats.subprocess, "run", return_value=result) as run:
                self.assertEqual(retcon_stats.java_temp_directory("selected-java"), Path(directory))
                self.assertEqual(run.call_args.args[0][0], "selected-java")

    def test_complete_export_preserves_raw_and_normalizes_actual_virtual_separately(self):
        with tempfile.TemporaryDirectory() as directory:
            capture = retcon_stats.Capture(directory)
            raw = json.dumps(fixture())
            (capture.path / "Retcon-2-100.json").write_text(raw)
            destination = Path(directory) / "artifacts"
            summary = capture.collect(destination, 5, 2, require=True)
            self.assertEqual(summary["status"], "complete")
            self.assertEqual(summary["actualGun"], dict(shots=100, hits=12, hitRate=0.12))
            self.assertEqual(summary["virtualGuns"][0]["shots"], 70)
            self.assertEqual(summary["data"]["waves"]["hitsOnWave"], 18)
            self.assertEqual((destination / "Retcon-2-100.json").read_text(), raw)
            capture.close()
            self.assertTrue((destination / "capture.json").exists())

    def test_missing_and_partial_are_not_silent_zero_measurements(self):
        with tempfile.TemporaryDirectory() as directory:
            capture = retcon_stats.Capture(directory)
            target = Path(directory) / "artifacts"
            missing = capture.collect(target, 5, 2)
            self.assertEqual(missing["status"], "missing")
            self.assertNotIn("actualGun", missing)
            data = fixture()
            data["roundsRecorded"] = 4
            (capture.path / "Retcon-2-100.json").write_text(json.dumps(data))
            self.assertEqual(capture.collect(target, 5, 2)["status"], "partial")
            with self.assertRaisesRegex(ValueError, "partial"):
                capture.collect(target, 5, 2, require=True)
            capture.close()

    def test_malformed_identity_and_counter_data_fail_strict_capture(self):
        mutations = [lambda d: d.update(bot="Other"), lambda d: d.update(botId=3),
                     lambda d: d.update(roundsRecorded=6), lambda d: d.update(turns=-1),
                     lambda d: d.update(turns=True), lambda d: d["gun"].update(bulletHits=101),
                     lambda d: d["gun"].update(virtualGuns="Linear:2/1"),
                     lambda d: d["waves"].update(hitsOnWave=21), lambda d: d.update(skippedTurns=1)]
        with tempfile.TemporaryDirectory() as directory:
            capture = retcon_stats.Capture(directory)
            for mutate in mutations:
                data = fixture()
                mutate(data)
                (capture.path / "Retcon-2-100.json").write_text(json.dumps(data))
                with self.subTest(data=data), self.assertRaisesRegex(ValueError, "invalid"):
                    capture.collect(Path(directory) / "artifacts", 5, 2, require=True)
            (capture.path / "Retcon-2-100.json").write_text('{"bot":"Retcon","turns":NaN}')
            with self.assertRaisesRegex(ValueError, "Non-finite"):
                capture.collect(Path(directory) / "artifacts", 5, 2, require=True)
            capture.close()

    def test_multiple_game_files_and_wrong_filename_identity_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            capture = retcon_stats.Capture(directory)
            (capture.path / "Retcon-3-100.json").write_text(json.dumps(fixture()))
            with self.assertRaisesRegex(ValueError, "filename botId"):
                capture.collect(Path(directory) / "artifacts", 5, 2, require=True)
            (capture.path / "Retcon-2-200.json").write_text(json.dumps(fixture()))
            with self.assertRaisesRegex(ValueError, "Multiple"):
                capture.collect(Path(directory) / "artifacts", 5, 2, require=True)
            capture.close()

    def test_cleanup_refuses_replaced_owner_and_unexpected_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            capture = retcon_stats.Capture(directory)
            capture.marker.write_text("another owner")
            with self.assertRaisesRegex(ValueError, "ownership"):
                capture.close()
            capture.marker.write_text(capture.token)
            child = capture.path / "unexpected"
            child.mkdir()
            with self.assertRaisesRegex(ValueError, "subdirectories"):
                capture.close()
            child.rmdir()
            capture.close()


if __name__ == "__main__":
    unittest.main()
