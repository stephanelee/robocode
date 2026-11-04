"""Offline tests of the registered Prism screen plan and decision gates. No games are run."""
import copy
import json
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
    def __init__(self, changed=("docs/prism/screen-plan.json",), ancestor=True, reviewed_plan=None):
        self.refs, self.changed, self.ancestor, self.calls = {}, list(changed), ancestor, []
        # the plan as it existed at the reviewed head (default: the committed, unapproved plan)
        self.reviewed_plan = ps.load_plan() if reviewed_plan is None else reviewed_plan

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
        if cmd[:2] == ["git", "show"]:
            if self.reviewed_plan == "unreadable":
                return 128, ""
            return 0, json.dumps(self.reviewed_plan)
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

    def test_approval_only_edit_of_the_plan_is_accepted(self):
        # reviewed plan: approval pending; execution plan: same content plus the approval record
        ps.verify_clearance(approved_plan(), "https://example.test/review", HEAD, FakeGit())

    def test_reviewers_repro_changed_seed_and_id_is_rejected(self):
        plan = approved_plan()
        plan["games"][0]["seed"] = 999999
        plan["games"][0]["id"] = "unreviewed-population"
        with self.assertRaises(SystemExit) as e:
            ps.verify_clearance(plan, "https://example.test/review", HEAD, FakeGit())
        self.assertIn("games", str(e.exception))

    def test_changed_seat_opponent_rounds_freeze_or_receipt_is_rejected(self):
        edits = [
            lambda p: p["games"][1].update(seat="prism-first" if p["games"][1]["seat"] == "opponent-first" else "opponent-first"),
            lambda p: p["games"][2].update(opponent="Walls"),
            lambda p: p["games"][3].update(rounds=19),
            lambda p: p["games"].reverse(),
            lambda p: p["freeze"].update(prismJarSHA256="0" * 64),
            lambda p: p["freeze"].update(profile="other-profile"),
            lambda p: p.update(receipt="refs/tags/other"),
            lambda p: p.update(extra="unreviewed field"),
            lambda p: p.pop("status"),
        ]
        for i, edit in enumerate(edits):
            plan = approved_plan()
            edit(plan)
            with self.assertRaises(SystemExit, msg="edit %d accepted" % i):
                ps.verify_clearance(plan, "https://example.test/review", HEAD, FakeGit())

    def test_unreadable_reviewed_plan_is_refused(self):
        with self.assertRaises(SystemExit):
            ps.verify_clearance(approved_plan(), "https://example.test/review", HEAD, FakeGit(reviewed_plan="unreadable"))

    def test_approval_fields_are_the_only_exemption(self):
        self.assertEqual(("approvedReview", "approvedHead"), ps.APPROVAL_FIELDS)
        plan = approved_plan()
        self.assertNotIn("approvedReview", ps.frozen_content(plan))
        self.assertIn("games", ps.frozen_content(plan))
        self.assertIn("freeze", ps.frozen_content(plan))

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

class NativeStagingTest(unittest.TestCase):
    def test_both_python_opponents_use_their_locked_source_paths(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from types import SimpleNamespace
        import sys
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cache = root / "cache"
            source = cache / "strong-python" / "fixture" / "source"
            opponents = [
                dict(descriptor="diamond-port.json", sourcePath="bots/ports/diamond-port"),
                dict(descriptor="adaptive-prime.json", sourcePath="bots/adaptive-prime"),
            ]
            for bot in opponents:
                parent = source / bot["sourcePath"]
                parent.mkdir(parents=True)
                (parent / bot["descriptor"]).write_text(bot["sourcePath"])
            checked = []
            def verify(stage, names):
                for key in names:
                    bot = next(b for b in opponents if Path(b["descriptor"]).stem == key)
                    self.assertEqual(bot["sourcePath"], (stage / "opponents" / key / bot["descriptor"]).read_text())
                    checked.append(key)
            fake = SimpleNamespace(manifest=lambda: dict(commit="fixture", opponents=opponents),
                                   prepare=lambda *_: None, stage=lambda *_: {}, verify_staged=verify)
            with patch.dict(sys.modules, {"strong_melee_opponents": fake}), patch.object(ps, "CACHE", cache):
                for name, key in [("Diamond Port", "diamond-port"), ("Adaptive Prime", "adaptive-prime")]:
                    stage = root / key
                    stage.mkdir()
                    result = ps.stage_opponent(name, stage, {})
                    self.assertEqual(stage / "opponents" / key, result)
            self.assertEqual(["diamond-port", "adaptive-prime"], checked)


if __name__ == "__main__":
    unittest.main()
