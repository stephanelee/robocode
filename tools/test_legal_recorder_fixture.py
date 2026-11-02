"""Synthetic Java-producer output, not a real game or training dataset."""
import json
import unittest
import benchmark
import legal_scan_dataset


class JavaProducerSchemaTests(unittest.TestCase):
    def test_actual_java_serialization_obeys_the_accepted_label_schema(self):
        folder = benchmark.ROOT / "tools" / "fixtures" / "legal-recorder"
        manifest = json.loads((folder / "synthetic-manifest.json").read_text())
        records = list(legal_scan_dataset.load_events(folder / "synthetic-events.jsonl"))
        samples = list(legal_scan_dataset.build_samples(records, manifest, horizons=(1,), history=2))
        self.assertEqual(len(records), 4)
        self.assertEqual(len(samples), 4)
        self.assertEqual(sum(row["label"]["observed"] for row in samples), 1)
        self.assertTrue(all(row["split"] == "train" for row in samples))
        for row in samples:
            frame = row["features"]["history"][-1]
            self.assertLess(frame["previousIntent"]["submittedTurn"], row["anchorTurn"])
            if row["anchorTurn"] == 1:
                self.assertEqual(frame["confirmedShots"], [])
            else:
                self.assertEqual(len(frame["confirmedShots"]), 1)
                self.assertEqual(frame["confirmedShots"][0]["turn"], 2)
