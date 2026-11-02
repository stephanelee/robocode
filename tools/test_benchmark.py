import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from benchmark import extract, summarize_metrics, write_reports


class BenchmarkTests(unittest.TestCase):
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
