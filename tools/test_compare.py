import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import compare


def result(share, wins=2, walls=2, skipped=0):
    return dict(rounds=20, firstPlaces=wins, totalScore=share * 100,
                results=[dict(name="AISurvivor", totalScore=share * 100),
                         dict(name="Retcon", totalScore=(1 - share) * 100)],
                metrics=dict(turns=1000, shotsFired=10, bulletHits=2, damageDealt=10,
                             damageTaken=20, wallCollisions=walls, skippedTurns=skipped))


class CompareTests(unittest.TestCase):
    def block(self, previous=0.2, candidate=0.3, **candidate_kwargs):
        return dict(previous=result(previous), candidate=result(candidate, **candidate_kwargs))

    def test_retention_requires_all_preregistered_checks(self):
        blocks = [self.block() for _ in range(3)]
        measured = compare.decision(blocks)
        self.assertEqual(measured["status"], "retain")
        self.assertAlmostEqual(measured["candidate"]["meanScoreShare"], 0.3)
        self.assertEqual(compare.decision(blocks[:2])["status"], "incomplete")
        for changes, failed in ((dict(candidate=0.24), "meanShareGain"),
                                (dict(skipped=1), "noSkippedTurns"),
                                (dict(walls=3), "wallRate")):
            decision = compare.decision([self.block(**changes) for _ in range(3)])
            self.assertEqual(decision["status"], "reject")
            self.assertFalse(decision["checks"][failed])
        regressed = [self.block(candidate=0.5), self.block(candidate=0.5), self.block(candidate=0.14)]
        self.assertFalse(compare.decision(regressed)["checks"]["maximumBlockLoss"])

    def test_early_stop_requires_both_score_loss_and_strong_previous_wins(self):
        block = dict(previous=result(0.5, wins=4), candidate=result(0.2, wins=0))
        self.assertTrue(compare.catastrophic_first_block(block))
        block["candidate"]["firstPlaces"] = 1
        self.assertFalse(compare.catastrophic_first_block(block))
        block["candidate"] = result(0.4, wins=0)
        self.assertFalse(compare.catastrophic_first_block(block))

    def test_wilson_intervals_remain_bounded_for_all_losses_or_wins(self):
        losses = compare.wilson(0, 60)
        wins = compare.wilson(60, 60)
        self.assertAlmostEqual(losses["lower"], 0)
        self.assertGreater(losses["upper"], 0)
        self.assertLess(wins["lower"], 1)
        self.assertAlmostEqual(wins["upper"], 1)
        with self.assertRaises(ValueError):
            compare.wilson(1, 0)

    def bundles(self, root):
        for arm in ("previous", "candidate"):
            directory = root / arm
            directory.mkdir()
            for name in compare.BOT_FILES:
                (directory / name).write_bytes((arm if name.endswith(".jar") else name).encode())
        return root / "previous", root / "candidate"

    def test_version_snapshots_detect_mutation_and_reject_changed_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            previous, candidate = self.bundles(root)
            snapshots, hashes = compare.snapshot_bots(previous, candidate, root / "output")
            compare.verify_snapshots(snapshots, hashes)
            (snapshots["previous"] / "AISurvivor.jar").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "snapshot changed"):
                compare.verify_snapshots(snapshots, hashes)
            (candidate / "AISurvivor.json").write_bytes(b"different descriptor")
            with self.assertRaisesRegex(ValueError, "configuration/launcher differs"):
                compare.snapshot_bots(previous, candidate, root / "other")

    def test_checkpoint_copies_are_separate_and_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.dat"
            source.write_bytes(b"shared immutable model")
            expected = compare.benchmark.digest(source)
            compare.copy_checkpoint(source, root / "previous.dat", expected)
            compare.copy_checkpoint(source, root / "candidate.dat", expected)
            (root / "candidate.dat").write_bytes(b"changed")
            self.assertEqual(compare.benchmark.digest(source), expected)
            self.assertEqual(compare.benchmark.digest(root / "previous.dat"), expected)
            with self.assertRaisesRegex(ValueError, "does not match"):
                compare.copy_checkpoint(source, root / "bad.dat", "wrong")

    def test_run_isolates_seed_arms_alternates_order_and_preserves_source_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            previous, candidate = self.bundles(root)
            checkpoint = root / "checkpoint.dat"
            checkpoint.write_bytes(b"original trained checkpoint")
            plan = root / "plan.md"
            plan.write_text("preregistered plan", encoding="utf-8")
            args = argparse.Namespace(output=root / "output", previous_bot_dir=previous,
                candidate_bot_dir=candidate, previous_source="old", candidate_source="new",
                plan=plan, plan_commit="registered", policy="trained", purpose="primary",
                checkpoint=checkpoint, training_rounds=20, seeds=[1, 2, 3], rounds=20,
                timeout=60, java="java", cache=root / "cache")
            calls = []
            def battle(options, output, runner, samples, scenario, opponents, policy, rounds, model):
                calls.append((options.seed, options.bot_dir.name, model))
                self.assertEqual(model.read_bytes(), checkpoint.read_bytes())
                return result(0.3 if options.bot_dir.name == "candidate" else 0.2)
            runtime = argparse.Namespace(stderr="java.specification.version = 25", stdout="", returncode=0)
            runner = root / "runner.jar"
            runner.write_bytes(b"runner")
            with patch.object(compare.subprocess, "run", return_value=runtime), \
                 patch.object(compare.retcon, "prepare", return_value=(root / "Retcon", root / "api", {"commit": "pinned"})), \
                 patch.object(compare.benchmark, "download", return_value=runner), \
                 patch.object(compare.benchmark, "battle", side_effect=battle):
                report = compare.run(args)
            self.assertEqual([(seed, arm) for seed, arm, _ in calls],
                             [(1, "previous"), (1, "candidate"), (2, "candidate"),
                              (2, "previous"), (3, "previous"), (3, "candidate")])
            self.assertEqual(len({model for _, _, model in calls}), 6)
            self.assertEqual(checkpoint.read_bytes(), b"original trained checkpoint")
            self.assertEqual(report["decision"]["status"], "retain")
            self.assertEqual(report["plannedEvaluationRounds"], 120)
            self.assertEqual(report["validatedEvaluationRounds"], 120)
            self.assertEqual(json.loads((args.output / "comparison.json").read_text())["status"], "complete")
            with self.assertRaisesRegex(ValueError, "empty"):
                compare.run(args)

    def test_seed_list_requires_three_unique_valid_java_longs(self):
        self.assertEqual(compare.seeds_value("1,2,3"), [1, 2, 3])
        for value in ("1,1,2", "1,2", "1,2,9223372036854775808"):
            with self.assertRaises(argparse.ArgumentTypeError):
                compare.seeds_value(value)


if __name__ == "__main__":
    unittest.main()
