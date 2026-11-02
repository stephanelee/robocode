"""Synthetic validator tests; actual-engine evidence is recorded by the probe workflow."""
import copy
import unittest
import engine_energy_trace


def fixture():
    rows = []
    for shooter, victim, label, turn in ((1, 2, "Alpha", 70), (2, 1, "Beta", 220)):
        rows += [dict(type="hit", label=label, turn=turn, self=shooter, enemy=victim,
                      bullet=shooter, power=0.5, damage=2),
                 dict(type="hitBy", turn=turn, self=victim, enemy=shooter,
                      bullet=shooter, power=0.5, damage=2)]
        for observer, enemy, delta in ((shooter, victim, -2), (victim, shooter, 1.5)):
            for offset in (-1, 0, 1):
                rows.append(dict(type="scan", turn=turn + offset, self=observer, enemy=enemy,
                                 energy=100 + (delta if offset == 1 else 0)))
    return rows


class TraceValidatorTests(unittest.TestCase):
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
