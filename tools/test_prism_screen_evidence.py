"""Offline preservation/reproduction checks for the spent Prism screen; runs no bots."""
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
B = ROOT / "benchmarks/prism-screen-v1"
DATA = B / "raw/build/prism-screen"
HEAD = "0ac58d50ddaf041d0103a125b745fd4f2709a90c"
JAR = "8c045f80d37a3076ced7b337b020d14bf3942c6c2a89152f11021f8ff9e9e7ca"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def reproduce():
    plan = read(B / "source/docs/prism/screen-plan.json")
    report = read(DATA / "decision.json")
    assert report["head"] == HEAD
    assert len(report["games"]) == len(plan["games"]) == 13
    assert report["planSHA256"] == hashlib.sha256((B / "source/docs/prism/screen-plan.json").read_bytes()).hexdigest()
    receipt = read(B / "raw/build/prism-screen-receipt.json")
    assert receipt["head"] == HEAD and receipt["runId"] == "41863990617"
    assert receipt["ref"] == "refs/tags/prism-screen-v1-8c045f80d37a3076"
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
        assert own["version"] == "0.1.0" and other["name"] == expected_names.get(game["opponent"], game["opponent"])
        assert other["version"] == expected_versions[game["opponent"]]
        assert own["id"] != other["id"]
        # Official first places may include ties; do not impose complementary win counts.
        assert 0 <= own["firstPlaces"] <= game["rounds"] and 0 <= other["firstPlaces"] <= game["rounds"]
        telemetry = [json.loads(x) for x in (p / "prism-telemetry.jsonl").read_text().splitlines()]
        assert all(x["seed"] == game["seed"] and x["profile"] == "prism-initial-profile-1" for x in telemetry)
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
    assert missing == {"diamond-3": [15]}
    assert sum(x["rounds"] for x in rows) == 200
    assert sum(x["telemetryRoundRows"] for x in rows) == 199
    spec = importlib.util.spec_from_file_location("frozen_prism_screen", B / "source/tools/prism_screen.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.gates(report["games"]) == report["gates"]
    assert report["decision"] == report["gates"]["decision"] == "reject"
    assert [k for k, v in report["gates"].items() if v is False] == ["diamondAbove40"]
    return dict(executionHead=HEAD, runId=41863990617, artifactId=13343025344,
                originalZipSHA256="26489fff821c0237536c0442f6c674e845dfe2756ac208e37cceb7203ed46a8f",
                games=rows, physicalRounds=200, telemetryRoundRows=199,
                missingRoundTelemetry=missing, gates=report["gates"], decision="reject")


class PrismEvidenceTest(unittest.TestCase):
    def test_every_preserved_file_hash(self):
        for name, entry in read(B / "manifest.json")["files"].items():
            data = (B / name).read_bytes()
            self.assertEqual(len(data), entry["bytes"], name)
            self.assertEqual(hashlib.sha256(data).hexdigest(), entry["sha256"], name)

    def test_original_official_and_cumulative_evidence_reproduces_rejection(self):
        self.assertEqual(reproduce(), read(B / "audit-summary.json"))


if __name__ == "__main__":
    unittest.main()
