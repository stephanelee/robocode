import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from benchmark import extract, isolate_opponent, summarize_metrics, validate_metrics, write_reports


class BenchmarkTests(unittest.TestCase):
    def test_isolated_java_sample_retains_sibling_runtime_and_independent_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "samples" / "Target"
            source.mkdir(parents=True)
            (source / "Target.sh").write_text("java -cp ../lib/* Target.java")
            (source / "Target.java").write_text("original")
            shared = source.parent / "lib"
            shared.mkdir()
            (shared / "bot-api.jar").write_bytes(b"shared runtime")
            destination = root / "battle" / "opponents" / "Target"
            isolate_opponent(source, destination)
            self.assertEqual((destination.parent / "lib" / "bot-api.jar").read_bytes(), b"shared runtime")
            (destination / "Target.java").write_text("round-local state")
            self.assertEqual((source / "Target.java").read_text(), "original")

    def metric(self, round_id=1, **overrides):
        return dict(dict(schemaVersion=1, round=round_id, turns=100, mode="eval", seed=42,
                         action=None, totalRounds=0, checkpointLoaded=False, shotsFired=10,
                         bulletHits=2, hitRate=0.2, damageDealt=8.0, damageTaken=4.0,
                         wallCollisions=0, botCollisions=0, skippedTurns=0,
                         won=True, died=False, reward=15.0), **overrides)

    def test_valid_cold_fixed_train_and_trained_evidence(self):
        validate_metrics([self.metric()], 1, "untrained", 42, 10)
        validate_metrics([self.metric(mode="fixed", action=0)], 1, "fixed-minrisk", 42, 10)
        validate_metrics([self.metric(mode="train", totalRounds=1)], 1, "train", 42, 10)
        validate_metrics([self.metric(totalRounds=10, checkpointLoaded=True)], 1, "trained", 42, 10)

    def test_duplicate_rounds_do_not_pass_count_check(self):
        with self.assertRaisesRegex(ValueError, "round IDs"):
            validate_metrics([self.metric(), self.metric()], 2, "untrained", 42, 10)

    def test_wrong_mode_seed_checkpoint_action_or_schema_is_rejected(self):
        for changes in ({"mode": "train"}, {"seed": 43}, {"checkpointLoaded": True},
                        {"action": 1}, {"schemaVersion": 2}, {"totalRounds": 10}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_metrics([self.metric(**changes)], 1, "untrained", 42, 10)
        with self.assertRaisesRegex(ValueError, "checkpoint status"):
            validate_metrics([self.metric(totalRounds=10)], 1, "trained", 42, 10)

    def test_nonfinite_negative_and_inconsistent_metrics_are_rejected(self):
        for changes in ({"reward": float("nan")}, {"damageTaken": float("inf")},
                        {"wallCollisions": -1}, {"shotsFired": True}, {"hitRate": 0.5},
                        {"bulletHits": 11}, {"won": True, "died": True}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_metrics([self.metric(**changes)], 1, "untrained", 42, 10)

    def test_missing_telemetry_fails_required_gate(self):
        with self.assertRaisesRegex(ValueError, "Expected 1 telemetry rows"):
            validate_metrics([], 1, "untrained", 42, 10)

    def test_rejects_archive_traversal_before_extracting_anything(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "bad.zip"
            with zipfile.ZipFile(archive, "w") as zipped:
                zipped.writestr("safe.txt", "safe")
                zipped.writestr("../escape.txt", "bad")
            with self.assertRaises(ValueError):
                extract(archive, root / "samples")
            self.assertFalse((root / "escape.txt").exists())
            self.assertFalse((root / "samples" / "safe.txt").exists())

    def test_metrics_aggregate_shots_instead_of_averaging_round_rates(self):
        metrics = summarize_metrics([{"shotsFired": 10, "bulletHits": 1, "wallCollisions": 2},
                                     {"shotsFired": 2, "bulletHits": 2, "skippedTurns": 1}])
        self.assertEqual(metrics["hitRate"], 0.25)
        self.assertEqual(metrics["wallCollisions"], 2)
        self.assertEqual(metrics["skippedTurns"], 1)
        self.assertIsNone(summarize_metrics([])["hitRate"])

    def test_html_escapes_untrusted_names_and_json_preserves_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            report = dict(seed=5, trainingRounds=2, limitations="small sample", battles=[
                dict(scenario="<script>alert(1)</script>", policy="untrained", rounds=1,
                     rank=2, totalScore=10, firstPlaces=0, metrics={})])
            write_reports(output, report)
            self.assertNotIn("<script>alert(1)</script>", (output / "dashboard.html").read_text())
            self.assertIn("&lt;script&gt;", (output / "dashboard.html").read_text())
            self.assertEqual(json.loads((output / "results.json").read_text()), report)


if __name__ == "__main__":
    unittest.main()
