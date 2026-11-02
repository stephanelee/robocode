import hashlib
import json
from pathlib import Path
import unittest

import benchmark
import compare


class CycleThreeEvidenceTests(unittest.TestCase):
    def test_all_training_and_evaluation_rows_models_and_diagnostics_reproduce_report(self):
        root = Path(__file__).resolve().parents[1] / "benchmarks" / "cycle-3"
        report = json.loads((root / "comparison.json").read_text(encoding="utf-8"))
        self.assertEqual(report["plannedTrainingRounds"], 60)
        self.assertEqual(report["validatedTrainingRounds"], 60)
        self.assertEqual(report["plannedEvaluationRounds"], 120)
        self.assertEqual(report["validatedEvaluationRounds"], 120)
        self.assertEqual(hashlib.sha256((root / "preregistered-plan.md").read_bytes()).hexdigest(), report["preregistrationSHA256"])
        self.assertEqual(hashlib.sha256((root / "preregistered-addendum.md").read_bytes()).hexdigest(), report["preregistrationAddendumSHA256"])
        training = {battle["seed"]: battle for battle in report["trainingBattles"]}
        self.assertEqual(set(training), set(report["seeds"]))
        for block in report["blocks"]:
            seed = block["seed"]
            model = root / "checkpoints" / f"seed-{seed}.dat"
            expected_hash = report["checkpointSHA256BySeed"][str(seed)]
            self.assertEqual(benchmark.digest(model), expected_hash)
            self.assertEqual(compare.checkpoint_rounds(model), 20)
            diagnostics = compare.checkpoint_diagnostics(model)
            self.assertEqual(diagnostics, report["checkpointDiagnosticsBySeed"][str(seed)])
            self.assertEqual(diagnostics, training[seed]["checkpointDiagnostics"])
            self.assertTrue(diagnostics["targetingEvidenceActive"])
            self.assertEqual(training[seed]["modelSHA256"], expected_hash)
            self.assertEqual(training[seed]["trainerSource"], report["previousSource"])
            self.assertEqual(block["checkpointSHA256"], expected_hash)
            games = [("training-previous", "train", training[seed]),
                     ("previous", "trained", block["previous"]), ("candidate", "trained", block["candidate"])]
            for arm, policy, game in games:
                directory = root / f"seed-{seed}" / arm / f"Retcon-{policy}"
                raw = json.loads((directory / "battle.json").read_text(encoding="utf-8"))
                rows = [json.loads(line) for line in (directory / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
                self.assertEqual(raw["numberOfRounds"], 20)
                self.assertEqual(raw["results"], game["results"])
                benchmark.validate_metrics(rows, 20, policy, seed, 20)
                self.assertEqual(benchmark.summarize_metrics(rows), game["metrics"])
                self.assertEqual(game["modelSHA256"], expected_hash)
        self.assertEqual(compare.decision(report["blocks"]), report["decision"])
        self.assertEqual(report["decision"]["status"], "reject")
        self.assertEqual({key for key, passed in report["decision"]["checks"].items() if not passed},
                         {"meanShareGain", "maximumBlockLoss"})


if __name__ == "__main__":
    unittest.main()
