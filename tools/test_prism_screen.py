"""Offline tests of the registered Prism screen plan and decision gates. No games are run."""
import copy
import unittest

import prism_screen as ps


def row(opp, share, **kw):
    r = dict(opponent=opp, status="valid", share=share, failures=0, skipped=0, turns=10000, collisions=10)
    r.update(kw)
    return r


def passing():
    rows = []
    for opp, n in (("Diamond Port", 3), ("Retcon", 3), ("Adaptive Prime", 3), ("SpinBot", 2), ("Walls", 2)):
        rows += [row(opp, 0.6) for _ in range(n)]
    return rows


class PlanTest(unittest.TestCase):
    def test_plan_is_the_registered_budget(self):
        plan = ps.load_plan()
        self.assertEqual(200, sum(g["rounds"] for g in plan["games"]))
        seats = [g["seat"] for g in plan["games"] if g["opponent"] == "Retcon"]
        self.assertIn("prism-first", seats)
        self.assertIn("opponent-first", seats)
        self.assertEqual(len({g["seed"] for g in plan["games"]}), len(plan["games"]))

    def test_all_gates_pass(self):
        self.assertEqual("passed screen", ps.gates(passing())["decision"])

    def test_any_invalid_game_rejects(self):
        rows = passing()
        rows[4] = dict(rows[4], status="invalid")
        self.assertEqual("reject", ps.gates(rows)["decision"])

    def test_share_thresholds(self):
        rows = copy.deepcopy(passing())
        for r in rows:
            if r["opponent"] == "Diamond Port":
                r["share"] = 0.40          # must be strictly above 40%
        self.assertEqual("reject", ps.gates(rows)["decision"])
        rows = copy.deepcopy(passing())
        for r in rows:
            if r["opponent"] == "Walls":
                r["share"] = 0.50          # at least 50% passes
        self.assertEqual("passed screen", ps.gates(rows)["decision"])

    def test_skips_collisions_and_failures(self):
        rows = passing()
        rows[0] = dict(rows[0], skipped=200)   # 200 of 30000 Diamond turns > 0.5%
        self.assertEqual("reject", ps.gates(rows)["decision"])
        rows = passing()
        for r in rows:
            r["collisions"] = 100
        self.assertEqual("reject", ps.gates(rows)["decision"])
        rows = passing()
        rows[7] = dict(rows[7], failures=1)
        self.assertEqual("reject", ps.gates(rows)["decision"])

    def test_share(self):
        self.assertEqual(0.5, ps.share(0, 0))
        self.assertAlmostEqual(0.25, ps.share(1, 3))


if __name__ == "__main__":
    unittest.main()
