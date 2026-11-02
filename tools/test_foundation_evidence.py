"""Recompute the rejected foundation experiment from durable raw evidence."""
import json
from pathlib import Path
import unittest
import benchmark
import compare
import engine_energy_trace
import retcon_stats


class FoundationEvidenceTests(unittest.TestCase):
    def test_complete_fixed_rounds_native_exports_and_registered_decision(self):
        root = benchmark.ROOT / "benchmarks" / "foundation-scan-lag"
        report = json.loads((root / "comparison.json").read_text())
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["validatedScoredRounds"], 120)
        self.assertEqual(report["validatedOnlineTrainingRounds"], 0)
        self.assertEqual(report["category"], "correctness")
        self.assertEqual(report["purpose"], "primary")
        self.assertEqual(report["policy"], "fixed-minrisk")
        self.assertFalse(report["earlyStoppingEnabled"])
        self.assertEqual(report["seeds"], [20261102, 20261103, 20261104])
        self.assertEqual(report["preregistrationCommit"], "9b5a5313f13719920b8abf04094dd9438a74b7df")
        self.assertEqual(report["preregistrationAddendumCommit"], "677f622c8b90b89783b224e59d90e0da9942eca2")
        self.assertEqual(benchmark.digest(root / "preregistered-plan.md"), report["preregistrationSHA256"])
        self.assertEqual(benchmark.digest(root / "preregistered-addendum.md"), report["preregistrationAddendumSHA256"])
        for block in report["blocks"]:
            for arm in ("previous", "candidate"):
                game = block[arm]
                directory = root / f"seed-{block['seed']}" / arm / "Retcon-fixed-minrisk"
                official = json.loads((directory / "battle.json").read_text())
                rows = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines()]
                self.assertEqual(official["numberOfRounds"], 20)
                self.assertEqual(official["results"], game["results"])
                benchmark.validate_metrics(rows, 20, "fixed-minrisk", block["seed"], 20)
                self.assertEqual(benchmark.summarize_metrics(rows), game["metrics"])
                self.assertIsNone(game["modelSHA256"])
                native = json.loads((directory / "retcon-stats" / "capture.json").read_text())
                self.assertEqual(native, game["retconStats"])
                self.assertEqual(native["status"], "complete")
                enemy = next(row for row in official["results"] if row["name"] == "Retcon")
                derived = retcon_stats.validate(native["data"], 20, enemy["id"])
                self.assertEqual(derived["actualGun"], native["actualGun"])
                for artifact in native["rawArtifacts"]:
                    path = directory / "retcon-stats" / artifact["path"]
                    self.assertEqual(benchmark.digest(path), artifact["sha256"])
                    if path.suffix == ".json":
                        self.assertEqual(json.loads(path.read_text()), native["data"])
        self.assertEqual(compare.decision(report["blocks"], "correctness"), report["decision"])
        self.assertEqual(report["decision"]["status"], "reject")
        self.assertEqual({key for key, passed in report["decision"]["checks"].items() if not passed},
                         {"meanShareGain", "wallRate"})

    def test_match_runtime_probe_raw_hashes_reproduce_actual_contract(self):
        root = benchmark.ROOT / "benchmarks" / "foundation-scan-lag"
        directory = root / "match-runtime-trace"
        report = json.loads((directory / "trace.json").read_text())
        self.assertEqual(report["sourceCommit"], "78315b77eb4e763bb438cf0cb7df5e7535aa5638")
        self.assertEqual(report["os"]["osRelease"]["VERSION_ID"], "26.04")
        records = []
        for label, expected in report["rawSHA256"].items():
            path = directory / (label + ".jsonl")
            self.assertEqual(benchmark.digest(path), expected)
            records.extend(json.loads(line) for line in path.read_text().splitlines())
        self.assertEqual(engine_energy_trace.validate(records), report["assertions"])
        for field, name in (("probeSourceSHA256", "EngineEnergyProbe.java"),
                            ("recorderSHA256", "engine_energy_trace.py"),
                            ("runnerAdapterSHA256", "BenchmarkBattle.java")):
            self.assertEqual(benchmark.digest(root / "engine-trace" / "source" / name), report[field])
