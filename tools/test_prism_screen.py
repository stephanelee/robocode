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



class FakeGit:
    """Fake command runner: a remote that holds refs, and a diff between heads."""
    def __init__(self, changed=("docs/prism/screen-plan.json",), ancestor=True):
        self.refs, self.changed, self.ancestor, self.calls = {}, list(changed), ancestor, []

    def __call__(self, cmd):
        self.calls.append(cmd)
        if cmd[:2] == ["gh", "api"]:
            ref = [c for c in cmd if c.startswith("ref=")][0][4:]
            sha = [c for c in cmd if c.startswith("sha=")][0][4:]
            if ref in self.refs:
                return 1, '{"message":"Reference already exists"}'
            self.refs[ref] = sha
            return 0, "{}"
        if cmd[:2] == ["git", "ls-remote"]:
            ref = cmd[-1]
            return (0, self.refs[ref] + "\t" + ref) if ref in self.refs else (0, "")
        if cmd[:3] == ["git", "merge-base", "--is-ancestor"]:
            return (0 if self.ancestor else 1), ""
        if cmd[:3] == ["git", "diff", "--name-only"]:
            return 0, "\n".join(self.changed)
        raise AssertionError(cmd)


REVIEWED = "a" * 40
HEAD = "b" * 40


def approved_plan():
    plan = ps.load_plan()
    plan["approvedReview"] = "https://example.test/review"
    plan["approvedHead"] = REVIEWED
    return plan


class ReceiptTest(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.tmp = tempfile.TemporaryDirectory()
        from pathlib import Path
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_unrecorded_or_mismatched_review_is_refused(self):
        plan = ps.load_plan()
        with self.assertRaises(SystemExit):
            ps.verify_clearance(plan, "https://example.test/review", HEAD, FakeGit())
        with self.assertRaises(SystemExit):
            ps.verify_clearance(approved_plan(), "https://other.test", HEAD, FakeGit())

    def test_exact_head_may_only_add_the_approval_record(self):
        ps.verify_clearance(approved_plan(), "https://example.test/review", HEAD, FakeGit())
        with self.assertRaises(SystemExit):
            ps.verify_clearance(approved_plan(), "https://example.test/review", HEAD,
                                FakeGit(changed=["docs/prism/screen-plan.json", "experiments/Prism/src/prism/Brain.java"]))
        with self.assertRaises(SystemExit):
            ps.verify_clearance(approved_plan(), "https://example.test/review", HEAD, FakeGit(ancestor=False))

    def test_second_dispatch_is_refused_before_any_game(self):
        git = FakeGit()
        plan = approved_plan()
        r1 = self.dir / "run1.json"
        ps.consume_receipt(plan, "owner/repo", HEAD, 111, git, r1)
        ps.verify_receipt(plan, HEAD, 111, git, r1)
        # a later dispatch: fresh workspace, same remote; consumption fails and no receipt file exists
        with self.assertRaises(SystemExit):
            ps.consume_receipt(plan, "owner/repo", HEAD, 222, git, self.dir / "run2.json")
        self.assertFalse((self.dir / "run2.json").exists())
        with self.assertRaises(SystemExit):
            ps.verify_receipt(plan, HEAD, 222, git, self.dir / "run2.json")
        # the receipt stays spent: it is never deleted
        self.assertIn(ps.receipt_ref(plan), git.refs)

    def test_receipt_of_another_run_or_head_is_rejected(self):
        git = FakeGit()
        plan = approved_plan()
        r = self.dir / "r.json"
        ps.consume_receipt(plan, "owner/repo", HEAD, 111, git, r)
        with self.assertRaises(SystemExit):
            ps.verify_receipt(plan, HEAD, 112, git, r)          # a re-run attempt with another id
        with self.assertRaises(SystemExit):
            ps.verify_receipt(plan, "c" * 40, 111, git, r)      # another head

    def test_receipt_is_tied_to_the_frozen_candidate(self):
        plan = approved_plan()
        self.assertTrue(ps.receipt_ref(plan).startswith("refs/tags/prism-screen-v1-"))
        self.assertIn(plan["freeze"]["prismJarSHA256"][:16], ps.receipt_ref(plan))

if __name__ == "__main__":
    unittest.main()
