"""Synthetic fixtures only: no acquisition, observer truth, model fit or matches."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from legal_scan_dataset import build_samples, parse_json, validate_manifest, validate_record


def manifest():
    return dict(schemaVersion=1, datasetId="synthetic-test", provenance=dict(
        sourceCommit="0" * 40, botJarSha256="0" * 64, engineVersion="1.4.0", collectorVersion="proposal-v1"),
        games=[dict(gameId=game, acquisitionId="capture-" + game, split=split,
                    opponent="Synthetic", opponentPin="fixture-v1", arenaWidth=1600, arenaHeight=1200)
               for game, split in (("A", "train"), ("B", "dev"), ("C", "withheld"), ("D", "transfer"))])


def event(turn, game="A", round_number=1, target=2, observed=None, x=None):
    return dict(schemaVersion=1, gameId=game, round=round_number, turn=turn, targetId=target,
        own=dict(x=400, y=300, direction=90, speed=8, gunDirection=0),
        scan=dict(observedTurn=turn if observed is None else observed,
                  x=100 + turn if x is None else x, y=100, direction=0, speed=1),
        previousIntent=None, confirmedShots=[])


def samples(rows, horizons=(1,), history=4):
    return list(build_samples(rows, manifest(), horizons, history))


class LabelFixtures(unittest.TestCase):
    def test_exact_fresh_future_only_no_interpolation(self):
        rows = [event(1), event(2, observed=1, x=101), event(3)]
        result = samples(rows, (1, 2))
        anchor = [s for s in result if s["anchorTurn"] == 1]
        self.assertFalse(anchor[0]["label"]["observed"])
        self.assertIsNone(anchor[0]["label"]["x"])
        self.assertTrue(anchor[1]["label"]["observed"])
        self.assertEqual(anchor[1]["label"]["x"], 103)

    def test_no_labels_across_game_round_or_target(self):
        rows = [event(1), event(2, game="B"), event(2, round_number=2), event(2, target=3)]
        self.assertTrue(all(not s["label"]["observed"] for s in samples(rows)))

    def test_future_scan_values_do_not_enter_history(self):
        before = samples([event(1)])[0]["features"]
        after = samples([event(1), event(2, x=999)])[0]
        self.assertEqual(before, after["features"])
        self.assertEqual(after["label"]["x"], 999)
        self.assertNotIn("label", after["features"])

    def test_missing_turn_and_scan_masks_are_explicit(self):
        missing_scan = event(2)
        missing_scan["scan"] = None
        result = samples([event(1), missing_scan, event(4, observed=1, x=101)])
        self.assertEqual(len(result), 2)  # unknown target state at2 is not anchor
        history = result[-1]["features"]["history"]
        self.assertFalse(history[1]["scanFresh"])
        self.assertIsNone(history[1]["scanAge"])
        self.assertFalse(history[2]["recordPresent"])
        self.assertIsNone(history[2]["own"])
        self.assertEqual(history[3]["scanAge"], 3)

    def test_future_and_same_turn_intents_rejected(self):
        for submitted in (2, 3):
            row = event(2)
            row["previousIntent"] = dict(submittedTurn=submitted, targetSpeed=8)
            with self.assertRaises(ValueError):
                samples([row])
        row["previousIntent"]["submittedTurn"] = 1
        self.assertEqual(samples([row])[0]["features"]["history"][-1]["previousIntent"]["submittedTurn"], 1)

    def test_requested_fire_is_not_confirmed_fire(self):
        row = event(2)
        row["previousIntent"] = dict(submittedTurn=1, firepower=3)
        features = samples([row])[0]["features"]["history"][-1]
        self.assertEqual(features["previousIntent"]["firepower"], 3)
        self.assertEqual(features["confirmedShots"], [])

    def test_future_own_shot_and_conflicting_identity_rejected(self):
        row = event(2)
        row["confirmedShots"] = [dict(turn=3, bulletId=1, power=2, direction=0, x=400, y=300)]
        with self.assertRaises(ValueError):
            samples([row])
        row["confirmedShots"][0]["turn"] = 2
        other = copy.deepcopy(row)
        other["targetId"] = 3
        other["confirmedShots"][0]["power"] = 1
        with self.assertRaises(ValueError):
            samples([row, other])

    def test_conflicting_cached_scan_and_duplicate_record_rejected(self):
        for rows in ([event(1), event(1)], [event(1), event(2, observed=1, x=900)]):
            with self.assertRaises(ValueError):
                samples(rows)

    def test_optional_state_fields_preserve_unknown_not_zero(self):
        frame = samples([event(1)])[0]["features"]["history"][0]
        self.assertIsNone(frame["own"]["radarDirection"])
        self.assertIsNone(frame["own"]["energy"])
        self.assertIsNone(frame["scan"]["energy"])

    def test_order_invariance_and_independent_feature_objects(self):
        rows = [event(1), event(3), event(2, game="B")]
        self.assertEqual(samples(rows), samples(list(reversed(rows))))
        result = samples([event(1), event(2)], (1, 2))
        result[0]["features"]["history"][0]["scan"]["x"] = 999
        self.assertEqual(result[1]["features"]["history"][0]["scan"]["x"], 101)


class SplitAndValidationFixtures(unittest.TestCase):
    def test_entire_game_assigned_before_windows(self):
        rows = [event(t, game=g, x=1000 + t if g == "B" else 100 + t)
                for g in ("A", "B", "C", "D") for t in (1, 2, 3)]
        result = samples(rows)
        for sample in result:
            expected = dict(A="train", B="dev", C="withheld", D="transfer")[sample["gameId"]]
            self.assertEqual(sample["split"], expected)
            xs = [f["scan"]["x"] for f in sample["features"]["history"] if f["scan"]]
            self.assertTrue(all(x > 1000 for x in xs) if sample["gameId"] == "B" else all(x < 1000 for x in xs))

    def test_duplicate_game_or_acquisition_cannot_cross_splits(self):
        for key in ("gameId", "acquisitionId"):
            value = manifest()
            value["games"][1][key] = value["games"][0][key]
            with self.assertRaises(ValueError):
                validate_manifest(value)

    def test_undeclared_game_and_injected_split_rejected(self):
        for row in (event(1, game="unassigned"), {**event(1), "split": "train"}):
            with self.assertRaises(ValueError):
                samples([row])

    def test_same_tick_own_snapshot_consistency_and_arena_bounds(self):
        row, other = event(1), event(1, target=3)
        other["own"]["speed"] = 7
        with self.assertRaises(ValueError):
            samples([row, other])
        with self.assertRaises(ValueError):
            samples([event(1, x=1700)])

    def test_cached_scan_time_cannot_regress(self):
        with self.assertRaises(ValueError):
            samples([event(2), event(3, observed=1, x=101)])

    def test_private_unknown_fields_are_not_silently_features(self):
        for parent in ("own", "scan"):
            row = event(1)
            row[parent]["retconHistogram"] = [1, 2, 3]
            with self.assertRaises(ValueError):
                validate_record(row)

    def test_future_scan_nonfinite_bool_duplicate_json_rejected(self):
        for row in (event(1, observed=2), {**event(1), "round": True}):
            with self.assertRaises(ValueError):
                samples([row])
        for value in (float("nan"), float("inf"), True):
            row = event(1)
            row["own"]["x"] = value
            with self.assertRaises(ValueError):
                samples([row])
        for text in ('{"round":1,"round":2}', '{"x":NaN}'):
            with self.assertRaises(ValueError):
                parse_json(text)

    def test_invalid_window_and_horizon_rejected(self):
        for horizons, history in (((0,), 4), ((1, 1), 4), ((151,), 4), ((1,), True), ((1,), 0)):
            with self.assertRaises(ValueError):
                samples([event(1)], horizons, history)

    def test_cli_partitions_before_any_model_fit_and_preserves_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text(json.dumps(manifest()), encoding="utf-8")
            rows = [event(1), event(2), event(1, game="B"), event(2, game="B")]
            (root / "events.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
            command = [sys.executable, str(Path(__file__).with_name("legal_scan_dataset.py")),
                       "--manifest", str(root / "manifest.json"), "--events", str(root / "events.jsonl"),
                       "--output", str(root / "out"), "--horizons", "1"]
            completed = subprocess.run(command, capture_output=True, text=True, timeout=10)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            metadata = json.loads((root / "out/dataset.json").read_text())
            self.assertEqual(metadata["counts"]["train"], dict(examples=2, observedLabels=1))
            self.assertEqual(metadata["counts"]["dev"], dict(examples=2, observedLabels=1))
            self.assertEqual(metadata["counts"]["withheld"], dict(examples=0, observedLabels=0))
            self.assertEqual(len(metadata["toolSha256"]), 64)
            before = (root / "out/train.jsonl").read_bytes()
            self.assertNotEqual(subprocess.run(command, capture_output=True, timeout=10).returncode, 0)
            self.assertEqual(before, (root / "out/train.jsonl").read_bytes())


if __name__ == "__main__":
    unittest.main()
