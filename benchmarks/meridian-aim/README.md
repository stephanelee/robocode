# Reproduce the Meridian targeting screen

[Registration and results](../../docs/improvements/meridian-aim.md) · [game schedule](games.json) · [runtime pins](profile.json) · [no-game preflight](https://github.com/stephanelee/robocode/actions/runs/41737978269).

To verify the retained games from the repository root, reduce the official recordings and private event markers again:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=tools python3 - <<'PY'
import json
from pathlib import Path
from meridian_aim import passes_screen
from meridian_duels import metrics
root = Path("benchmarks/meridian-aim")
raw = root / "raw"
assert (raw / "profile.json").read_bytes() in [(root / name).read_bytes() for name in ("profile.json", "profile-20261026.json")]
plan = json.loads((root / "games.json").read_text())
report = json.loads((raw / "decision.json").read_text())
assert len(report["games"]) in (4, 13)
for game, row in zip(plan, report["games"]):
    assert all(row[k] == v for k, v in game.items())
    if row["status"] == "valid":
        computed = metrics(raw / row["id"], game)
        assert all(row[k] == v for k, v in computed.items())
        print(game["id"], computed["wins"], computed["share"], computed["shots"])
assert passes_screen(report["games"][:4]) == (len(report["games"]) == 13)
PY
```

For a physical replication, follow the [registered workflow](../../.github/workflows/meridian-aim.yml) on a fresh Ubuntu 26.04 checkout with Git history available. Build with Java 17 and verify JAR entry parity. Switch to Java 25, use Python 3.13.18 and run `python tools/meridian_profile.py`. Compare `build/meridian/profile.json` to the two registered fingerprints, then run `python tools/meridian_aim.py`. The runner reconstructs the exact v2 control from commit `85cccdeb5f9bde8af42f971510e13f0e089f28ec` and hashes all four bot assets. Starts vary across physical games.
