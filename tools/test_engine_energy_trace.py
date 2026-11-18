"""Validator regressions from recorded engine traces and explicit synthetic edge cases."""
import copy
import unittest
import hashlib
import json
import contextlib
import io
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
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


def battle_fixture():
    return dict(numberOfRounds=1, results=[dict(id=1, name="Alpha", version="1.0"),
                                         dict(id=2, name="Beta", version="1.0")])


class ProbeRosterTests(unittest.TestCase):
    def test_exact_roster_accepts_reordered_ids_and_rejects_incomplete_or_wrong_results(self):
        good = battle_fixture()
        engine_energy_trace.validate_battle(good)
        engine_energy_trace.validate_battle(dict(numberOfRounds=1, results=list(reversed(good["results"]))))
        bad = [None, {}, dict(good, numberOfRounds=2), dict(good, numberOfRounds=True),
               dict(good, results=good["results"][:1]), dict(good, results=good["results"] * 2)]
        for key, value in (("id", 1), ("id", True), ("name", "Alpha"), ("version", "2.0"), ("name", [])):
            changed = copy.deepcopy(good)
            changed["results"][1][key] = value
            bad.append(changed)
        for data in bad:
            with self.subTest(data=data), self.assertRaisesRegex(ValueError, "startup/result"):
                engine_energy_trace.validate_battle(data)

    def run_fixture(self, battle, records=None):
        """Exercise the real run() staging/validation/report path, never a server."""
        with tempfile.TemporaryDirectory() as directory, contextlib.ExitStack() as stack:
            root = Path(directory)
            output = root / "output"
            def download(cache, name, expected):
                cache.mkdir(parents=True, exist_ok=True)
                path = cache / name
                path.write_bytes(b"offline fixture")
                return path
            def extract(archive, target):
                target.mkdir()
                (target / "robocode-tankroyale-bot-api-1.4.0.jar").write_bytes(b"offline API fixture")
            def execute(*args):
                (output / "battle.json").write_text(json.dumps(battle), encoding="utf-8")
                if records is not None:
                    for identity, label in ((1, "Alpha"), (2, "Beta")):
                        selected = [row for row in records if row["self"] == identity]
                        (output / (label + ".jsonl")).write_text(
                            "".join(json.dumps(row) + "\n" for row in selected), encoding="utf-8")
            stack.enter_context(patch.object(engine_energy_trace.benchmark, "download", side_effect=download))
            stack.enter_context(patch.object(engine_energy_trace.benchmark, "extract", side_effect=extract))
            process = stack.enter_context(patch.object(engine_energy_trace.benchmark, "run_process", side_effect=execute))
            stack.enter_context(patch.object(engine_energy_trace.subprocess, "run", return_value=
                                           subprocess.CompletedProcess([], 0, "offline Java", "")))
            stack.enter_context(patch.object(engine_energy_trace.subprocess, "check_output", return_value="fixture-head\n"))
            stack.enter_context(patch.object(engine_energy_trace.retcon_stats, "os_provenance", return_value={}))
            timing = stack.enter_context(patch.object(engine_energy_trace, "validate", wraps=engine_energy_trace.validate))
            with contextlib.redirect_stdout(io.StringIO()):
                try:
                    engine_energy_trace.run(SimpleNamespace(output=output, cache=root / "cache", java="java"))
                    error = None
                except ValueError as failure:
                    error = str(failure)
            report = json.loads((output / "trace.json").read_text(encoding="utf-8"))
            self.assertEqual(1, process.call_count, "No automatic retry")
            self.assertEqual(battle, json.loads((output / "battle.json").read_text(encoding="utf-8")))
            return report, error, timing.call_count

    def test_real_entrypoint_reports_solo_startup_before_reading_absent_traces(self):
        # Same one-Alpha roster retained in issue102; trace files deliberately absent.
        solo = battle_fixture()
        solo["results"].pop()
        report, error, calls = self.run_fixture(solo)
        self.assertIn("startup/result roster", error)
        self.assertEqual(("failed", "roster"), (report["status"], report["failureStage"]))
        self.assertIn("battleSHA256", report)
        self.assertNotIn("assertions", report)
        self.assertEqual(0, calls)

    def test_real_entrypoint_keeps_timing_assertions_binding_after_valid_roster(self):
        report, error, calls = self.run_fixture(battle_fixture(), fixture())
        self.assertIsNone(error)
        self.assertEqual(("verified", 4, 1), (report["status"], len(report["assertions"]), calls))
        bad = fixture()
        next(row for row in bad if row["type"] == "hitBy")["damage"] = 0
        report, error, calls = self.run_fixture(battle_fixture(), bad)
        self.assertIn("both probes", error)
        self.assertEqual(("failed", "timing", 1), (report["status"], report["failureStage"], calls))


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
