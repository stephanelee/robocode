import json
import unittest
import benchmark
import containment_forecast


class ContainmentScreenEvidenceTests(unittest.TestCase):
    def test_frozen_development_rejection_and_input_dependencies(self):
        root = benchmark.ROOT / "benchmarks" / "ai-containment-v2"
        old = benchmark.ROOT / "benchmarks" / "ai-motion-v1"
        model = json.loads((root / "model.json").read_text())
        fit = json.loads((root / "fit-report.json").read_text())
        screen = json.loads((root / "screen-report.json").read_text())
        digest = benchmark.digest(root / "model.json")
        self.assertEqual(digest, "eacae24357e378e857ef2a37ad7e7533dac847773921556494e3c36a079e48dc")
        self.assertEqual(fit["modelSha256"], digest)
        self.assertEqual(screen["modelSha256"], digest)
        for report in (model, fit, screen):
            self.assertEqual(report["manifestSha256"], benchmark.digest(old / "manifest.json"))
            self.assertEqual(report["eventsSha256"], "d2cb8e055659485d89648eec6cd3aebe8cbfce8dfb4dd19f83e7ba3efa1d11e2")
            self.assertEqual(report["toolSha256"], benchmark.digest(root / "source" / "containment_forecast.py"))
            self.assertEqual(report["dependencySha256"], benchmark.digest(old / "source" / "motion_forecast.py"))
            self.assertEqual(report["datasetToolSha256"], benchmark.digest(old / "source" / "legal_scan_dataset.py"))
        self.assertEqual(set(screen["cells"]), {"r-dev-01", "r-dev-02", "r-dev-03"})
        self.assertEqual(containment_forecast.decision(screen["cells"]), screen["decision"])
        self.assertEqual(screen["decision"]["status"], "reject")
        self.assertEqual({key for key, valid in screen["decision"]["checks"].items() if not valid}, {"primaryGain"})
        self.assertLess(screen["decision"]["primaryContainmentDelta"], .02)
        self.assertTrue(all(delta > 0 for delta in screen["decision"]["gameDeltas"].values()))
