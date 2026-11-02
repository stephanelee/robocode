import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import compare
import benchmark
from test_compare import result
from test_compare_training import model_bytes
import test_benchmark


class OnlineComparisonTests(unittest.TestCase):
    def test_correctness_category_allows_ties_without_rewriting_tactical_strength_gate(self):
        blocks = [dict(previous=result(0.2), candidate=result(0.2)) for _ in range(3)]
        self.assertEqual(compare.decision(blocks)["status"], "reject")
        corrected = compare.decision(blocks, "correctness")
        self.assertEqual(corrected["status"], "retain")
        self.assertEqual(corrected["classification"], "correctness nonregression")
        self.assertFalse(corrected["tacticalStrengthGatePassed"])
        self.assertEqual(compare.THRESHOLDS["meanShareGain"], 0.05)
        self.assertEqual(compare.CORRECTNESS_THRESHOLDS["meanShareGain"], 0)

    def test_correctness_requires_mean_nonregression_and_two_nonnegative_blocks(self):
        rows = [dict(previous=result(0.2), candidate=result(value)) for value in (0.25, 0.25, 0.15)]
        self.assertEqual(compare.decision(rows, "correctness")["status"], "retain")
        rows = [dict(previous=result(0.2), candidate=result(value)) for value in (0.2, 0.2, 0.19)]
        self.assertFalse(compare.decision(rows, "correctness")["checks"]["meanShareGain"])
        rows = [dict(previous=result(0.2), candidate=result(value)) for value in (0.3, 0.19, 0.19)]
        self.assertFalse(compare.decision(rows, "correctness")["checks"]["nonnegativeBlocks"])

    def prepare(self, root):
        for arm in ("previous", "candidate"):
            bundle = root / arm
            bundle.mkdir()
            for name in compare.BOT_FILES:
                (bundle / name).write_bytes((arm if name.endswith(".jar") else name).encode())
        runner = root / "runner.jar"
        runner.write_bytes(b"runner")
        args = argparse.Namespace(output=root / "output", previous_bot_dir=root / "previous",
            candidate_bot_dir=root / "candidate", previous_source="old-source", candidate_source="new-source",
            plan=root / "plan.md", plan_commit="registered", policy="online-train", category="correctness",
            purpose="primary", checkpoint=None, train_previous=False, training_rounds=20,
            seeds=[1, 2, 3], rounds=20, timeout=60, java="java", cache=root / "cache", require_retcon_stats=True)
        return args, runner

    def run_mock(self, args, runner, battle):
        runtime = argparse.Namespace(stderr="java.specification.version = 25", stdout="", returncode=0)
        with patch.object(compare.subprocess, "run", return_value=runtime), \
             patch.object(compare, "registered_plan", return_value=b"immutable correctness plan"), \
             patch.object(compare.retcon, "prepare", return_value=(args.output.parent / "Retcon", args.output.parent / "api", {"commit": "pinned"})), \
             patch.object(compare.benchmark, "download", return_value=runner), \
             patch.object(compare.benchmark, "battle", side_effect=battle):
            return compare.run(args)

    def test_both_arms_train_from_fresh_models_and_all_game_scores_enter_the_veto(self):
        with tempfile.TemporaryDirectory() as directory:
            args, runner = self.prepare(Path(directory))
            calls = []
            def battle(options, output, runner, samples, scenario, opponents, policy, rounds, model):
                self.assertEqual(policy, "train")
                self.assertTrue(options.require_metrics)
                self.assertTrue(options.require_retcon_stats)
                self.assertFalse(model.exists())
                calls.append((options.seed, options.bot_dir.name, model))
                model.write_bytes(model_bytes(options.seed + (100 if options.bot_dir.name == "candidate" else 0)))
                measured = result(0.21 if options.bot_dir.name == "candidate" else 0.2)
                measured["modelSHA256"] = compare.benchmark.digest(model)
                return measured
            report = self.run_mock(args, runner, battle)
            self.assertEqual(len(calls), 6)
            self.assertEqual(len({model for _, _, model in calls}), 6)
            self.assertEqual([(seed, arm) for seed, arm, _ in calls],
                             [(1, "previous"), (1, "candidate"), (2, "candidate"), (2, "previous"),
                              (3, "previous"), (3, "candidate")])
            self.assertEqual(report["plannedOnlineTrainingRounds"], 120)
            self.assertEqual(report["validatedOnlineTrainingRounds"], 120)
            self.assertEqual(report["validatedScoredRounds"], 120)
            self.assertEqual(report["plannedEvaluationRounds"], 0)
            self.assertEqual(report["validatedEvaluationRounds"], 0)
            self.assertEqual(report["validatedTrainingRounds"], 0)
            self.assertEqual(report["trainingBattles"], [])
            self.assertEqual(report["decision"]["status"], "retain")
            self.assertAlmostEqual(report["decision"]["candidate"]["meanScoreShare"], 0.21)
            self.assertFalse(report["earlyStoppingEnabled"])
            self.assertFalse(report["decision"]["tacticalStrengthGatePassed"])
            for seed in args.seeds:
                for arm in ("previous", "candidate"):
                    saved = args.output / "artifacts" / f"online-{arm}-seed-{seed}.dat"
                    self.assertEqual(compare.checkpoint_rounds(saved), 20)
                    self.assertEqual(compare.benchmark.digest(saved), report["onlineCheckpointSHA256BySeedArm"][str(seed)][arm])
                self.assertNotEqual(report["onlineCheckpointSHA256BySeedArm"][str(seed)]["previous"],
                                    report["onlineCheckpointSHA256BySeedArm"][str(seed)]["candidate"])

    def test_incorrect_online_model_counter_stops_and_retains_failure_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            args, runner = self.prepare(Path(directory))
            def battle(options, output, runner, samples, scenario, opponents, policy, rounds, model):
                model.write_bytes(model_bytes(options.seed, rounds=19))
                measured = result(0.2)
                measured["modelSHA256"] = compare.benchmark.digest(model)
                return measured
            with self.assertRaisesRegex(ValueError, "exact-round checkpoint"):
                self.run_mock(args, runner, battle)
            self.assertEqual(json.loads((args.output / "comparison.json").read_text())["status"], "failed")

    def test_online_protocol_forbids_imported_or_shared_pretraining_models(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _ = self.prepare(Path(directory))
            args.checkpoint = Path("forbidden.dat")
            with self.assertRaisesRegex(ValueError, "fresh independent"):
                compare.validate_design(args)
            args.checkpoint = None
            args.train_previous = True
            with self.assertRaisesRegex(ValueError, "fresh independent"):
                compare.validate_design(args)

    def test_online_veto_cannot_be_relabelled_a_short_diagnostic(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _ = self.prepare(Path(directory))
            args.purpose = "diagnostic"
            with self.assertRaisesRegex(ValueError, "purpose=primary"):
                compare.validate_design(args)

    def test_online_telemetry_requires_fresh_loaded_flags_and_each_round_counter(self):
        factory = test_benchmark.BenchmarkTests()
        records = [factory.metric(round_id=index, mode="train", totalRounds=index) for index in range(1, 21)]
        benchmark.validate_metrics(records, 20, "train", 42, 20)
        records[9]["checkpointLoaded"] = True
        with self.assertRaisesRegex(ValueError, "checkpoint status"):
            benchmark.validate_metrics(records, 20, "train", 42, 20)
        records[9]["checkpointLoaded"] = False
        records[9]["totalRounds"] = 9
        with self.assertRaisesRegex(ValueError, "training budget"):
            benchmark.validate_metrics(records, 20, "train", 42, 20)

    def test_correctness_protocol_never_performance_stops_before_all_declared_blocks(self):
        with tempfile.TemporaryDirectory() as directory:
            args, runner = self.prepare(Path(directory))
            def battle(options, output, runner, samples, scenario, opponents, policy, rounds, model):
                model.write_bytes(model_bytes(options.seed))
                measured = result(0.1 if options.bot_dir.name == "candidate" else 0.5,
                                  wins=0 if options.bot_dir.name == "candidate" else 4)
                measured["modelSHA256"] = compare.benchmark.digest(model)
                return measured
            report = self.run_mock(args, runner, battle)
            self.assertEqual(len(report["blocks"]), 3)
            self.assertEqual(report["validatedOnlineTrainingRounds"], 120)
            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["decision"]["status"], "reject")

    def test_future_tactical_protocol_also_completes_every_declared_block(self):
        with tempfile.TemporaryDirectory() as directory:
            args, runner = self.prepare(Path(directory))
            args.category = "tactical"
            def battle(options, output, runner, samples, scenario, opponents, policy, rounds, model):
                model.write_bytes(model_bytes(options.seed))
                measured = result(0.1 if options.bot_dir.name == "candidate" else 0.5,
                                  wins=0 if options.bot_dir.name == "candidate" else 4)
                measured["modelSHA256"] = compare.benchmark.digest(model)
                return measured
            report = self.run_mock(args, runner, battle)
            self.assertEqual(len(report["blocks"]), 3)
            self.assertEqual(report["validatedScoredRounds"], 120)
            self.assertFalse(report["earlyStoppingEnabled"])


if __name__ == "__main__":
    unittest.main()
