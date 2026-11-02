import json
import unittest
import benchmark


class TargetingRecordingTests(unittest.TestCase):
    def test_accepted_recording_provenance_and_known_firing_quirk_remain_frozen(self):
        root = benchmark.ROOT / "benchmarks" / "targeting-refactor"
        provenance = json.loads((root / "capture-provenance.json").read_text())
        self.assertEqual(benchmark.digest(root / "accepted-AISurvivor.java.snapshot"), provenance["acceptedAISurvivorSHA256"])
        self.assertEqual(benchmark.digest(root / "accepted-goldens.json"), provenance["goldenSHA256"])
        outcomes = json.loads((root / "accepted-goldens.json").read_text())["outcomes"]
        self.assertEqual(len(outcomes), 288)
        cases = {row["name"]: row for row in outcomes}
        # This refactor intentionally preserves the old fixed 10-degree gate.
        self.assertEqual(len(cases["range-500.0-gun-9.0"]["fire"]), 1)
        self.assertEqual(len(cases["range-500.0-gun-10.0"]["fire"]), 0)
        self.assertFalse(cases["rejected-fire"]["waveChanged"])
        self.assertFalse(cases["null-stats"]["waveChanged"])
        self.assertTrue(cases["no-target"]["wavePresent"])
        self.assertEqual(cases["no-target"]["gun"], [])
