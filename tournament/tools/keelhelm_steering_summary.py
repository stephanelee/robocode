"""Aggregate the registered #149 paired-window replay and expose each decision criterion."""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def summarize(control, steer):
    primary = "Diamond Port / 500+"
    shooters = defaultdict(Counter)
    games = {}
    counts = Counter()
    dispositions = Counter()
    if set(control) != set(steer) or len(steer) != 12:
        raise ValueError("incomplete game list")
    for game, windows in steer.items():
        total = Counter()
        for base, w in zip(control[game], windows, strict=True):
            for key in ("round", "start", "turnSign"):
                if base[key] != w[key]:
                    raise ValueError((game, key))
            if base["end"] != w["end"] and w["stop"] not in {"wall", "ram"}:
                raise ValueError("unexplained window difference")
            counts["windows"] += 1
            counts[w["stop"]] += 1
            for key in ("pulseTicks", "followTicks", "walls", "rams", "wallPenalty", "ramPenalty"):
                counts[key] += w[key]
            penalty = w["wallPenalty"] + w["ramPenalty"]
            total["charged"] += penalty
            for shooter, d in w["damage"].items():
                if base["end"] == w["end"] and abs(base["damage"].get(shooter, {}).get("original", 0) - d["original"]) > 1e-9:
                    raise ValueError("control damage mismatch")
                charged = sum(d[key] for key in ("candidate", "unresolved", "latent", "truncation"))
                shooters[shooter].update(d)
                shooters[shooter]["charged"] += charged
                total["charged"] += charged
                total["original"] += d["original"]
            for trace in w["diamondLong"]:
                dispositions[trace["disposition"]] += 1
        games[game] = dict(total)
    diamond = shooters[primary]
    original = sum(g["original"] for g in games.values())
    charged = sum(g["charged"] for g in games.values())
    improved = sum(g["original"] > g["charged"] + 1e-8 for g in games.values())
    checks = {
        "minimumDiamondHits": diamond["originalHits"] >= 20,
        "diamondDamageReduced5Percent": diamond["charged"] <= .95 * diamond["original"],
        "allDamageReduced2Percent": charged <= .98 * original,
        "sevenGamesImproved": improved >= 7,
        "zeroCollisions": counts["rams"] + counts["walls"] == 0,
    }
    return dict(checks=checks, passed=all(checks.values()), gamesImproved=improved, counts=counts,
                original=original, charged=charged, shooters=shooters, games=games, diamondTraces=dispositions)


if __name__ == "__main__":
    control, steer = [json.loads(Path(path).read_text()) for path in sys.argv[1:]]
    print(json.dumps(summarize(control, steer), indent=2, sort_keys=True))
