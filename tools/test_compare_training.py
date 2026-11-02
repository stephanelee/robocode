import argparse
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import compare
from test_compare import result


def model_bytes(seed, rounds=20, evidence=None):
    data = struct.pack(">7id", 0x41495356, 1, 14, 32, 6, rounds, 1000 + seed, 1.0)
    data += b"\0" * (2 * (14 * 32 + 32 + 32 * 6 + 6)) * 8
    stats = [0.0] * (5 * 9 * 47)
    stats[0] = float(seed if evidence is None else evidence)
    return data + struct.pack(f">{len(stats)}d", *stats)


class SharedTrainingTests(unittest.TestCase):
    def prepare(self, root):
        for arm in ("previous", "candidate"):
            bundle = root / arm
            bundle.mkdir()
            for name in compare.BOT_FILES:
                (bundle / name).write_bytes((arm if name.endswith(".jar") else name).encode())
        runner = root / "runner.jar"
        runner.write_bytes(b"runner")
        args = argparse.Namespace(output=root / "output", previous_bot_dir=root / "previous",
            candidate_bot_dir=root / "candidate", previous_source="incumbent", candidate_source="candidate",
            plan=root / "plan.md", plan_commit="registered", policy="trained", purpose="primary",
            checkpoint=None, train_previous=True, training_rounds=20, seeds=[1, 2, 3], rounds=20,
            timeout=60, java="java", cache=root / "cache")
        return args, runner

    def run_mock(self, args, runner, battle):
        runtime = argparse.Namespace(stderr="java.specification.version = 25", stdout="", returncode=0)
        with patch.object(compare.subprocess, "run", return_value=runtime), \
             patch.object(compare, "registered_plan", side_effect=lambda path, commit: b"prospective diagnostics addendum" if commit == getattr(args, "plan_addendum_commit", None) else b"immutable training plan"), \
             patch.object(compare.retcon, "prepare", return_value=(args.output.parent / "Retcon", args.output.parent / "api", {"commit": "pinned"})), \
             patch.object(compare.benchmark, "download", return_value=runner), \
             patch.object(compare.benchmark, "battle", side_effect=battle):
            return compare.run(args)

    def test_only_incumbent_trains_once_per_seed_and_both_arms_share_that_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args, runner = self.prepare(root)
            args.plan_addendum = args.plan
            args.plan_addendum_commit = "a" * 40
            calls = []
            evaluation_models = {}
            def battle(options, output, runner, samples, scenario, opponents, policy, rounds, model):
                calls.append((options.seed, options.bot_dir.name, policy, rounds))
                if policy == "train":
                    self.assertEqual(options.bot_dir.name, "previous")
                    self.assertFalse(model.exists())
                    model.write_bytes(model_bytes(options.seed))
                    measured = result(0.99, wins=20) # This score must never enter the primary comparison.
                else:
                    self.assertEqual(policy, "trained")
                    self.assertEqual(model.read_bytes(), model_bytes(options.seed))
                    evaluation_models[(options.seed, options.bot_dir.name)] = model
                    measured = result(0.3 if options.bot_dir.name == "candidate" else 0.2)
                measured["modelSHA256"] = compare.benchmark.digest(model)
                return measured
            report = self.run_mock(args, runner, battle)
            self.assertEqual([call for call in calls if call[2] == "train"],
                             [(1, "previous", "train", 20), (2, "previous", "train", 20), (3, "previous", "train", 20)])
            self.assertEqual(report["plannedTrainingRounds"], 60)
            self.assertEqual(report["validatedTrainingRounds"], 60)
            self.assertEqual(report["validatedEvaluationRounds"], 120)
            self.assertEqual(len(report["trainingBattles"]), 3)
            self.assertEqual(len(set(report["checkpointSHA256BySeed"].values())), 3)
            self.assertAlmostEqual(report["decision"]["previous"]["meanScoreShare"], 0.2)
            self.assertAlmostEqual(report["decision"]["candidate"]["meanScoreShare"], 0.3)
            self.assertEqual(report["decision"]["previous"]["rounds"], 60)
            self.assertEqual(report["decision"]["status"], "retain")
            self.assertEqual(report["preregistrationAddendumCommit"], "a" * 40)
            self.assertEqual((args.output / "preregistered-plan.md").read_bytes(), b"immutable training plan")
            self.assertEqual((args.output / "preregistered-addendum.md").read_bytes(), b"prospective diagnostics addendum")
            for seed in args.seeds:
                previous = evaluation_models[(seed, "previous")]
                candidate = evaluation_models[(seed, "candidate")]
                self.assertNotEqual(previous, candidate)
                self.assertEqual(previous.read_bytes(), candidate.read_bytes())
                self.assertEqual(compare.benchmark.digest(previous), report["checkpointSHA256BySeed"][str(seed)])
                self.assertEqual((args.output / "artifacts" / f"checkpoint-seed-{seed}.dat").read_bytes(), model_bytes(seed))

    def test_wrong_persisted_training_counter_fails_before_any_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            args, runner = self.prepare(Path(directory))
            calls = []
            def battle(options, output, runner, samples, scenario, opponents, policy, rounds, model):
                calls.append(policy)
                model.write_bytes(model_bytes(options.seed, rounds=19))
                measured = result(0.5)
                measured["modelSHA256"] = compare.benchmark.digest(model)
                return measured
            with self.assertRaisesRegex(ValueError, "exact 20-round checkpoint"):
                self.run_mock(args, runner, battle)
            self.assertEqual(calls, ["train"])
            self.assertEqual(json.loads((args.output / "comparison.json").read_text())["status"], "failed")

    def test_evaluation_model_mutation_stops_experiment_and_preserves_training_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            args, runner = self.prepare(Path(directory))
            def battle(options, output, runner, samples, scenario, opponents, policy, rounds, model):
                model.write_bytes(model_bytes(options.seed) if policy == "train" else b"mutated during evaluation")
                measured = result(0.5)
                measured["modelSHA256"] = compare.benchmark.digest(model)
                return measured
            with self.assertRaisesRegex(ValueError, "changed or loaded a different"):
                self.run_mock(args, runner, battle)
            self.assertEqual((args.output / "artifacts" / "checkpoint-seed-1.dat").read_bytes(), model_bytes(1))

    def test_train_previous_rejects_changed_training_budget_policy_or_imported_model(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _ = self.prepare(Path(directory))
            for key, value in (("training_rounds", 19), ("policy", "untrained"), ("checkpoint", Path("another.dat"))):
                original = getattr(args, key)
                setattr(args, key, value)
                with self.assertRaisesRegex(ValueError, "Train-previous"):
                    compare.validate_design(args)
                setattr(args, key, original)

    def test_checkpoint_header_rejects_legacy_truncation_and_wrong_dimensions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.dat"
            path.write_bytes(model_bytes(1))
            self.assertEqual(compare.checkpoint_rounds(path), 20)
            for data in (b"legacy", struct.pack(">7i", 0x41495356, 1, 99, 32, 6, 20, 1000)):
                path.write_bytes(data)
                with self.assertRaises(ValueError):
                    compare.checkpoint_rounds(path)

    def test_targeting_diagnostics_record_zero_or_positive_evidence_without_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.dat"
            for mass in (0, 12.5):
                original = model_bytes(1, evidence=mass)
                path.write_bytes(original)
                diagnostic = compare.checkpoint_diagnostics(path)
                self.assertEqual(diagnostic["classGFMass"], mass)
                self.assertEqual(diagnostic["classGFNonzeroBins"], int(mass > 0))
                self.assertEqual(diagnostic["targetingEvidenceActive"], mass > 0)
                self.assertEqual(diagnostic["classGFByClass"][0]["mass"], mass)
                self.assertEqual(path.read_bytes(), original)
            for mass in (-1, float("nan"), float("inf")):
                path.write_bytes(model_bytes(1, evidence=mass))
                with self.assertRaisesRegex(ValueError, "finite and nonnegative"):
                    compare.checkpoint_diagnostics(path)

    def test_zero_saved_targeting_evidence_does_not_exclude_a_seed_or_retrain(self):
        with tempfile.TemporaryDirectory() as directory:
            args, runner = self.prepare(Path(directory))
            calls = []
            def battle(options, output, runner, samples, scenario, opponents, policy, rounds, model):
                calls.append((options.seed, policy))
                if policy == "train":
                    model.write_bytes(model_bytes(options.seed, evidence=0 if options.seed == 1 else options.seed))
                measured = result(0.3 if options.bot_dir.name == "candidate" else 0.2)
                measured["modelSHA256"] = compare.benchmark.digest(model)
                return measured
            report = self.run_mock(args, runner, battle)
            self.assertEqual(sum(policy == "train" for _, policy in calls), 3)
            self.assertEqual(len(report["blocks"]), 3)
            self.assertFalse(report["checkpointDiagnosticsBySeed"]["1"]["targetingEvidenceActive"])
            self.assertEqual(report["validatedTrainingRounds"], 60)
            self.assertEqual(report["validatedEvaluationRounds"], 120)
            self.assertNotIn("matching cold networks", report["limitations"])


if __name__ == "__main__":
    unittest.main()
