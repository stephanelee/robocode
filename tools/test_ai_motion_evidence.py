"""Verify preserved real pilot bytes, provenance and the frozen rejection; no new fitting."""
import gzip
import hashlib
import json
import math
import unittest
import benchmark
import legal_scan_dataset
import motion_forecast
import retcon_stats


class RealMotionPilotEvidenceTests(unittest.TestCase):
    def test_raw_games_model_reports_and_decision_remain_reproducible(self):
        root = benchmark.ROOT / "benchmarks" / "ai-motion-v1"
        manifest = json.loads((root / "manifest.json").read_text())
        acquisition = json.loads((root / "acquisition.json").read_text())
        index = json.loads((root / "event-index.json").read_text())
        raw = gzip.decompress((root / "events.jsonl.gz").read_bytes())
        self.assertEqual(hashlib.sha256(raw).hexdigest(), acquisition["eventsSHA256"])
        self.assertEqual(len(raw), index["uncompressedBytes"])
        self.assertEqual(index["uncompressedSHA256"], acquisition["eventsSHA256"])
        records = [legal_scan_dataset.parse_json(line) for line in raw.splitlines()]
        self.assertEqual(len(records), 26962)
        legal_scan_dataset.validated_groups(records, manifest)
        self.assertEqual(acquisition["validatedRounds"], 45)
        self.assertEqual(acquisition["validatedGames"], 15)
        self.assertEqual(acquisition["status"], "complete")
        self.assertEqual(benchmark.digest(root / "registered-plan.md"), acquisition["planSHA256"])
        self.assertEqual(benchmark.digest(root / "registered-games.json"), acquisition["layoutSHA256"])
        self.assertEqual(manifest["provenance"]["sourceCommit"], "eeff2c053f77e1e4dc4e8b814239c5af13ff1aed")
        by_id = {game["gameId"]: game for game in acquisition["games"]}
        for segment in index["games"]:
            data = raw[segment["offset"]:segment["offset"] + segment["length"]]
            self.assertEqual(hashlib.sha256(data).hexdigest(), segment["sha256"])
            game = by_id[segment["gameId"]]; measured = game["battle"]
            self.assertEqual(segment["sha256"], game["eventsSHA256"])
            self.assertEqual(len(data.splitlines()), game["records"])
            folder = root / "games" / game["gameId"]
            self.assertEqual(benchmark.digest(folder / "events.jsonl.status.json"), game["statusSHA256"])
            status = json.loads((folder / "events.jsonl.status.json").read_text())
            self.assertEqual(status["status"], "complete")
            self.assertEqual({row["round"] for row in status["rounds"]}, {1, 2, 3})
            battle_folder = folder / (measured["scenario"] + "-fixed-minrisk")
            official = json.loads((battle_folder / "battle.json").read_text())
            metrics = [json.loads(line) for line in (battle_folder / "metrics.jsonl").read_text().splitlines()]
            self.assertEqual(official["numberOfRounds"], 3)
            self.assertEqual(official["results"], measured["results"])
            seed = next(g["aisSeed"] for g in manifest["games"] if g["gameId"] == game["gameId"])
            benchmark.validate_metrics(metrics, 3, "fixed-minrisk", seed, 20)
            self.assertEqual(benchmark.summarize_metrics(metrics), measured["metrics"])
            self.assertEqual(measured["metrics"]["skippedTurns"], 0)
            self.assertIsNone(measured["modelSHA256"])
            if measured["scenario"] == "Retcon":
                native = json.loads((battle_folder / "retcon-stats" / "capture.json").read_text())
                self.assertEqual(native, measured["retconStats"])
                self.assertEqual(native["status"], "complete")
                enemy = next(row for row in official["results"] if row["name"] == "Retcon")
                retcon_stats.validate(native["data"], 3, enemy["id"])
                for artifact in native["rawArtifacts"]:
                    self.assertEqual(benchmark.digest(battle_folder / "retcon-stats" / artifact["path"]), artifact["sha256"])
        model = json.loads((root / "model.json").read_text())
        fit = json.loads((root / "fit-report.json").read_text())
        evaluation = json.loads((root / "evaluation-report.json").read_text())
        self.assertEqual(benchmark.digest(root / "model.json"), "2b9650591220b17847ada24c19bc142d16360a9d40f263065452bdbcc665b8d5")
        self.assertEqual(benchmark.digest(root / "source" / "motion_forecast.py"), model["toolSha256"])
        for report in (model, fit, evaluation):
            self.assertEqual(report["manifestSha256"], benchmark.digest(root / "manifest.json"))
            self.assertEqual(report["eventsSha256"], acquisition["eventsSHA256"])
        self.assertEqual(fit["selectedBaseline"], "headon")
        self.assertEqual(evaluation["modelSha256"], benchmark.digest(root / "model.json"))
        self.assertEqual(motion_forecast.decision(evaluation["cells"], manifest, "headon"), evaluation["decision"])
        self.assertEqual(evaluation["decision"]["status"], "reject")
        self.assertEqual({key for key, valid in evaluation["decision"]["checks"].items() if not valid},
                         {"primaryError", "improvedGames", "worstGame", "containment", "transfer"})
        for weights in model["models"].values():
            self.assertEqual(len(weights["mean"]), 24)
            self.assertEqual(len(weights["scale"]), 24)
            self.assertTrue(all(math.isfinite(value) and value > 0 for value in weights["scale"]))
