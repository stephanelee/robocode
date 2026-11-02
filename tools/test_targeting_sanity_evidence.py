import json
import unittest
import benchmark
import retcon_stats


class TargetingSanityEvidenceTests(unittest.TestCase):
    def test_registered_five_round_compatibility_reproduces_raw_evidence(self):
        root = benchmark.ROOT / "benchmarks" / "targeting-refactor"
        report = json.loads((root / "sanity.json").read_text())
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["validatedRounds"], 5)
        self.assertEqual(report["sourceCommit"], "438e16de5cef95e10fb8a8d2d6b8254a0e47a68f")
        self.assertEqual(report["planCommit"], "c7f64d3be25220d4574f1b5b31bfe9d8bc6c6db7")
        self.assertEqual(benchmark.digest(root / "registered-plan.md"), report["planSHA256"])
        self.assertEqual(report["runtimeOS"]["osRelease"]["VERSION_ID"], "26.04")
        self.assertEqual([(game["scenario"], game["rounds"]) for game in report["battles"]],
                         [("Retcon", 2), ("Target", 1), ("SpinBot", 1), ("Walls", 1)])
        for game in report["battles"]:
            directory = root / (game["scenario"] + "-fixed-minrisk")
            raw = json.loads((directory / "battle.json").read_text())
            rows = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines()]
            self.assertEqual(raw["numberOfRounds"], game["rounds"])
            self.assertEqual(raw["results"], game["results"])
            benchmark.validate_metrics(rows, game["rounds"], "fixed-minrisk", 20261102, 20)
            self.assertEqual(benchmark.summarize_metrics(rows), game["metrics"])
            self.assertEqual(game["metrics"]["skippedTurns"], 0)
            self.assertIsNone(game["modelSHA256"])
            if game["scenario"] == "Retcon":
                native = json.loads((directory / "retcon-stats" / "capture.json").read_text())
                self.assertEqual(native, game["retconStats"])
                self.assertEqual(native["status"], "complete")
                enemy = next(row for row in raw["results"] if row["name"] == "Retcon")
                self.assertEqual(retcon_stats.validate(native["data"], 2, enemy["id"])["actualGun"], native["actualGun"])
                for artifact in native["rawArtifacts"]:
                    self.assertEqual(benchmark.digest(directory / "retcon-stats" / artifact["path"]), artifact["sha256"])
