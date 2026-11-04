"""Offline preservation/reproduction checks for the spent Prism screen; runs no bots."""
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest
import sys

ROOT = Path(__file__).resolve().parents[1]
B = ROOT / "benchmarks/prism-screen-v2"
DATA = B / "raw/build/prism-screen"
HEAD = "cc6886d86b43f4b955870dd6a6d05877f9cad226"
JAR = "f9c75a86e21aafa3dd889b626a2d9c4e321d69d4d3917e0483ea324360e27263"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def reproduce():
    plan = read(B / "source/docs/prism/screen-v2-plan.json")
    report = read(DATA / "decision.json")
    assert report["head"] == HEAD
    assert report["unplayedGameIds"] == []
    assert len(report["games"]) == len(plan["games"]) == 13
    assert report["planSHA256"] == hashlib.sha256((B / "source/docs/prism/screen-v2-plan.json").read_bytes()).hexdigest()
    receipt = read(B / "raw/build/prism-screen-receipt.json")
    assert receipt["head"] == HEAD and receipt["runId"] == "41892189167"
    assert receipt["ref"] == "refs/tags/prism-screen-v2-f9c75a86e21aafa3"
    rows = []
    missing = {}
    for game, row in zip(plan["games"], report["games"]):
        assert all(row[k] == v for k, v in game.items())
        assert row["status"] == "valid"
        p = DATA / game["id"]
        attempt = read(p / "attempt.json")
        assert attempt["game"] == game and attempt["execution"] == HEAD
        assert attempt["prismJarSHA256"] == JAR
        own_slot = -2 if game["seat"] == "prism-first" else -1
        assert attempt["command"][own_slot].endswith("/" + game["id"] + "/Prism")
        result = read(p / "results.json")
        assert result["numberOfRounds"] == game["rounds"]
        own, = [x for x in result["results"] if x["name"] == "Prism"]
        other, = [x for x in result["results"] if x["name"] != "Prism"]
        expected_names = {"SpinBot": "Spin Bot"}
        expected_versions = {"Diamond Port": "1.8.28", "Retcon": "0.2.0", "Adaptive Prime": "1.0", "SpinBot": "1.0", "Walls": "1.0"}
        assert own["version"] == "0.2.0" and other["name"] == expected_names.get(game["opponent"], game["opponent"])
        assert other["version"] == expected_versions[game["opponent"]]
        assert own["id"] != other["id"]
        # Official first places may include ties; do not impose complementary win counts.
        assert 0 <= own["firstPlaces"] <= game["rounds"] and 0 <= other["firstPlaces"] <= game["rounds"]
        telemetry = [json.loads(x) for x in (p / "prism-telemetry.jsonl").read_text().splitlines()]
        assert all(x["seed"] == game["seed"] and x["profile"] == "prism-profile-2-surfer-gun" for x in telemetry)
        round_rows = [x["round"] for x in telemetry if x["event"] == "round"]
        assert round_rows == sorted(set(round_rows))
        absent = sorted(set(range(1, game["rounds"] + 1)) - set(round_rows))
        if absent:
            missing[game["id"]] = absent
        endpoints = [x for x in telemetry if x["event"] == "game"]
        assert len(endpoints) == 1 and telemetry[-1] == endpoints[0]
        c = endpoints[0]["counters"]
        assert c["rounds"] == game["rounds"] and c == row["counters"]
        assert all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for x in telemetry for v in x["counters"].values())
        expected = dict(prismScore=own["totalScore"], opponentScore=other["totalScore"],
                        share=own["totalScore"] / (own["totalScore"] + other["totalScore"]),
                        prismFirstPlaces=own["firstPlaces"], opponentFirstPlaces=other["firstPlaces"],
                        prismBulletDamage=own["bulletDamage"], opponentBulletDamage=other["bulletDamage"],
                        shotsRequested=c.get("fire.requested", 0), shotsConfirmed=c.get("fire.confirmed", 0),
                        shotsHit=c.get("fire.hit", 0), collisions=c.get("collision.wall", 0) + c.get("collision.bot", 0),
                        skipped=c.get("turns.skipped", 0),
                        turns=sum(v for k, v in c.items() if k.startswith("move.action.")) + c.get("move.reason.min-risk", 0),
                        failures=sum(v for k, v in c.items() if k.startswith("failure.")))
        assert all(row[k] == v for k, v in expected.items())
        rows.append(dict(game, **expected, telemetryRoundRows=len(round_rows), opponentVersion=other["version"]))
    assert sum(x["rounds"] for x in rows) == 200
    spec = importlib.util.spec_from_file_location("frozen_prism_screen", B / "source/tools/prism_screen.py")
    module = importlib.util.module_from_spec(spec)
    old_path, old_bytecode = list(sys.path), sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = old_path
        sys.dont_write_bytecode = old_bytecode
    assert module.gates(report["games"]) == report["gates"]
    assert report["decision"] == report["gates"]["decision"] == "reject"
    assert [k for k, v in report["gates"].items() if v is False] == ["diamondAbove40"]
    return dict(executionHead=HEAD, runId=41892189167, artifactId=13353639321,
                originalZipSHA256="f58a4f6d000c9f87a6276df6b05b8235a43285af8910296cc355b1874ee88b7d",
                games=rows, physicalRounds=200, telemetryRoundRows=sum(x["telemetryRoundRows"] for x in rows),
                missingRoundTelemetry=missing, gates=report["gates"], decision="reject")


class PrismV2EvidenceTest(unittest.TestCase):
    def test_every_preserved_file_hash(self):
        for name, entry in read(B / "manifest.json")["files"].items():
            data = (B / name).read_bytes()
            self.assertEqual(len(data), entry["bytes"], name)
            self.assertEqual(hashlib.sha256(data).hexdigest(), entry["sha256"], name)

    def test_original_official_and_cumulative_evidence_reproduces_rejection(self):
        self.assertEqual(reproduce(), read(B / "audit-summary.json"))


if __name__ == "__main__":
    unittest.main()
