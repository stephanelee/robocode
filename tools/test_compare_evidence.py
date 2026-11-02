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
        self.assertEqual(hashlib.sha256((root / "preregistered-plan.md").read_bytes()).hexdigest(), report["preregistrationSHA256"])
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


if __name__ == "__main__":
    unittest.main()
