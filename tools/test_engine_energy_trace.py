"""Validator regressions from recorded engine traces and explicit synthetic edge cases."""
import copy
import unittest
import hashlib
import json
import engine_energy_trace


def fixture():
    rows = []
    for shooter, victim, label, turn in ((1, 2, "Alpha", 70), (2, 1, "Beta", 220)):
        rows += [dict(type="fired", label=label, turn=turn - 5, self=shooter, bullet=shooter, power=0.5),
                 dict(type="hit", label=label, turn=turn, self=shooter, enemy=victim,
                      bullet=shooter, power=0.5, damage=2),
                 dict(type="hitBy", turn=turn, self=victim, enemy=shooter,
                      bullet=shooter, power=0.5, damage=2)]
        for observer, enemy, delta in ((shooter, victim, -2), (victim, shooter, 1.5)):
            for offset in (-1, 0, 1):
                rows.append(dict(type="scan", turn=turn + offset, self=observer, enemy=enemy,
                                 energy=100 + (delta if offset == 1 else 0)))
    return rows


class TraceValidatorTests(unittest.TestCase):
    def close_start(self):
        directory = engine_energy_trace.benchmark.ROOT / "tools" / "fixtures" / "engine-energy-trace"
        raw = (directory / "close-start.jsonl").read_bytes()
        metadata = json.loads((directory / "close-start.provenance.json").read_bytes())
        self.assertEqual(hashlib.sha256(raw).hexdigest(), metadata["fixtureSHA256"])
        return [json.loads(line) for line in raw.splitlines()]

    def test_recorded_close_start_cost_does_not_hide_delayed_effects(self):
        rows = self.close_start()
        assertions = engine_energy_trace.validate(rows)
        self.assertEqual(assertions, engine_energy_trace.validate(list(reversed(rows))))
        self.assertEqual(sorted(row["effect"] for row in assertions), [-2.0, -2.0, 1.5, 1.5])
        adjusted = [row for row in assertions if row.get("sameTurnFiringCost")]
        self.assertEqual(len(adjusted), 2)
        self.assertTrue(all(row["sameTurnFiringCost"] == .5 and row["nextTurnFiringCost"] == 0
                            and row["nextTurnScan"] - row["sameTurnScan"] == 1.5 for row in adjusted))

    def test_close_start_still_rejects_early_and_late_damage_or_bonus(self):
        original = self.close_start()
        for hit in (row for row in original if row["type"] == "hit"):
            for observer, enemy, effect in ((hit["self"], hit["enemy"], -2),
                                            (hit["enemy"], hit["self"], 1.5)):
                for timing in ("early", "late"):
                    rows = copy.deepcopy(original)
                    scan = next(row for row in rows if row["type"] == "scan" and row["self"] == observer
                                and row["enemy"] == enemy and row["turn"] == hit["turn"] + (timing == "late"))
                    scan["energy"] += effect if timing == "early" else -effect
                    with self.assertRaisesRegex(ValueError, "k\\+1"):
                        engine_energy_trace.validate(rows)

    def test_only_observed_enemy_costs_at_exact_scan_turns_are_subtracted(self):
        rows = fixture()
        rows += [dict(type="fired", self=2, bullet=99, turn=70, power=.75),
                 dict(type="fired", self=2, bullet=100, turn=71, power=1.25)]
        for row in rows:
            if row["type"] == "scan" and row["self"] == 1 and row["enemy"] == 2:
                if row["turn"] == 70:
                    row["energy"] -= .75
                if row["turn"] == 71:
                    row["energy"] -= 2
        self.assertEqual(len(engine_energy_trace.validate(rows)), 4)
        for alteration in ("unobserved", "wrong-owner", "wrong-turn"):
            changed = copy.deepcopy(rows)
            event = next(row for row in changed if row["type"] == "fired" and row["bullet"] == 100)
            if alteration == "unobserved":
                changed.remove(event)
            elif alteration == "wrong-owner":
                event["self"] = 1
            else:
                event["turn"] = 72
            with self.assertRaisesRegex(ValueError, "k\\+1"):
                engine_energy_trace.validate(changed)

    def test_invalid_duplicate_or_future_firing_evidence_is_rejected(self):
        for power in (0, -.5, 3.1, float("nan"), float("inf")):
            rows = fixture() + [dict(type="fired", self=2, bullet=99, turn=70, power=power)]
            with self.assertRaisesRegex(ValueError, "firing event"):
                engine_energy_trace.validate(rows)
        rows = fixture()
        rows.append(copy.deepcopy(rows[0]))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            engine_energy_trace.validate(rows)
        rows = fixture()
        rows[0]["turn"] = 71
        with self.assertRaisesRegex(ValueError, "Matching observed"):
            engine_energy_trace.validate(rows)

    def test_recorded_official_engine_fixture_hashes_and_assertions(self):
        root = engine_energy_trace.benchmark.ROOT
        directory = root / "benchmarks" / "foundation-scan-lag" / "engine-trace"
        report = json.loads((directory / "trace.json").read_text())
        records = []
        for label, expected in report["rawSHA256"].items():
            raw = (directory / (label + ".jsonl")).read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(), expected)
            records.extend(json.loads(line) for line in raw.splitlines())
        self.assertEqual(engine_energy_trace.validate(records), report["assertions"])
        self.assertEqual(report["os"]["osRelease"]["VERSION_ID"], "26.04")
        self.assertEqual(report["runnerSHA256"], engine_energy_trace.benchmark.ASSETS["robocode-tankroyale-runner-1.4.0.jar"])
        for field, name in (("probeSourceSHA256", "EngineEnergyProbe.java"),
                            ("recorderSHA256", "engine_energy_trace.py"),
                            ("runnerAdapterSHA256", "BenchmarkBattle.java")):
            original = (directory / "source" / name).read_bytes()
            self.assertEqual(hashlib.sha256(original).hexdigest(), report[field])

    def test_accepts_both_actual_effects_without_assuming_callback_delivery_order(self):
        rows = fixture()
        self.assertEqual(len(engine_energy_trace.validate(rows)), 4)
        self.assertEqual(engine_energy_trace.validate(rows), engine_energy_trace.validate(list(reversed(rows))))

    def test_rejects_same_turn_visibility_or_missing_consecutive_scan(self):
        for alteration in ("same-turn", "missing"):
            rows = copy.deepcopy(fixture())
            scan = next(row for row in rows if row["type"] == "scan" and row["turn"] == 70)
            if alteration == "same-turn":
                scan["energy"] = 98
            else:
                rows.remove(scan)
            with self.assertRaises(ValueError):
                engine_energy_trace.validate(rows)

    def test_rejects_wrong_bonus_and_unconfirmed_damage(self):
        rows = fixture()
        next(row for row in rows if row["type"] == "scan" and row["turn"] == 71 and row["self"] == 2)["energy"] = 101
        with self.assertRaisesRegex(ValueError, "k\\+1"):
            engine_energy_trace.validate(rows)
        rows = [row for row in fixture() if row["type"] != "hitBy"]
        with self.assertRaisesRegex(ValueError, "both probes"):
            engine_energy_trace.validate(rows)
