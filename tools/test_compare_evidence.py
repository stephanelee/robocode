import hashlib
import json
from pathlib import Path
import unittest

import benchmark
import compare


class CycleOneEvidenceTests(unittest.TestCase):
    def test_durable_raw_evidence_reproduces_rejection_and_reported_metrics(self):
        root = Path(__file__).resolve().parents[1] / "benchmarks" / "cycle-1"
        report = json.loads((root / "comparison.json").read_text(encoding="utf-8"))
        self.assertEqual(report["validatedEvaluationRounds"], 120)
        self.assertEqual(report["plannedEvaluationRounds"], 120)
        # Git may check text out as CRLF on Windows; the artifact/Git blob uses canonical LF.
        canonical_plan = (root / "preregistered-plan.md").read_text(encoding="utf-8").encode("utf-8")
        self.assertEqual(hashlib.sha256(canonical_plan).hexdigest(), report["preregistrationSHA256"])
        for block in report["blocks"]:
            for arm in ("previous", "candidate"):
                directory = root / f"seed-{block['seed']}" / arm / "Retcon-untrained"
                raw = json.loads((directory / "battle.json").read_text(encoding="utf-8"))
                records = [json.loads(line) for line in (directory / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
                self.assertEqual(raw["numberOfRounds"], 20)
                self.assertEqual(raw["results"], block[arm]["results"])
                benchmark.validate_metrics(records, 20, "untrained", block["seed"], 0)
                self.assertEqual(benchmark.summarize_metrics(records), block[arm]["metrics"])
        self.assertEqual(compare.decision(report["blocks"]), report["decision"])
        self.assertEqual(report["decision"]["status"], "reject")
        failed = {key for key, passed in report["decision"]["checks"].items() if not passed}
        self.assertEqual(failed, {"meanShareGain", "maximumBlockLoss"})


class CycleTwoEvidenceTests(unittest.TestCase):
    def test_raw_cycle2_evidence_reproduces_rejection_and_exact_plan(self):
        root = Path(__file__).resolve().parents[1] / "benchmarks" / "cycle-2"
        report = json.loads((root / "comparison.json").read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["validatedEvaluationRounds"], 120)
        self.assertEqual(report["plannedEvaluationRounds"], 120)
        self.assertEqual(report["previousSource"], "18f2659f6f55f10a7a30b1ec54460eb7159ce101")
        self.assertEqual(report["candidateSource"], "4ead24ad85b6851ab5d2f336b1fe32be15f6ace8")
        self.assertEqual(report["preregistrationCommit"], "f34cffbf5bbe8c79fb55d1b4acba5b581ac802d6")
        expected_plan = "3e635c893281c938652506214ec708b2a464e0eedddcc870390eaf5d9c59cfc3"
        self.assertEqual(hashlib.sha256((root / "preregistered-plan.md").read_bytes()).hexdigest(), expected_plan)
        self.assertEqual(report["preregistrationSHA256"], expected_plan)
        for block in report["blocks"]:
            for arm in ("previous", "candidate"):
                directory = root / f"seed-{block['seed']}" / arm / "Retcon-untrained"
                raw = json.loads((directory / "battle.json").read_text(encoding="utf-8"))
                records = [json.loads(line) for line in (directory / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
                self.assertEqual(raw["numberOfRounds"], 20)
                self.assertEqual(raw["results"], block[arm]["results"])
                benchmark.validate_metrics(records, 20, "untrained", block["seed"], 0)
                self.assertEqual(benchmark.summarize_metrics(records), block[arm]["metrics"])
        self.assertEqual(compare.decision(report["blocks"]), report["decision"])
        self.assertEqual(report["decision"]["status"], "reject")
        self.assertEqual({key for key, passed in report["decision"]["checks"].items() if not passed}, {"meanShareGain"})


if __name__ == "__main__":
    unittest.main()
