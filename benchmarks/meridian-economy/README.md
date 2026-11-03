# Reproduce the Meridian firing-power comparison

[Protocol and results](../../docs/improvements/meridian-economy.md) · [manifest](games.json) · [hosted run](https://github.com/stephanelee/robocode/actions/runs/41742323605)

From the repository root, check the recorded runtime, recompute each valid row from the official recording and bot markers, and verify that confirmation was run only if the screen passed:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=tools python3 - <<'PY'
import json
from pathlib import Path
from meridian_economy import passes_screen
from meridian_duels import metrics
root = Path("benchmarks/meridian-economy")
raw = root / "raw"
assert (raw / "profile.json").read_bytes() in [(root / n).read_bytes() for n in ("profile.json", "profile-20261026.json")]
plan = json.loads((root / "games.json").read_text())
report = json.loads((raw / "decision.json").read_text())
rows = report["games"]
assert len(rows) == (len(plan) if passes_screen(rows[:4]) else 4)
for game, row in zip(plan, rows):
    assert all(row[k] == v for k, v in game.items())
    if row["status"] == "valid":
        actual = metrics(raw / game["id"], game)
        assert all(row[k] == v for k, v in actual.items())
        print(game["id"], row["wins"], row["share"], row["shotEnergy"] / row["shots"])
PY
```

The [workflow](../../.github/workflows/meridian-economy.yml) rebuilds on Java 17 and checks JAR parity, then uses Java 25 to launch matches. Both arms use the same Ubuntu 26.04 host. The runner extracts the v3 control from 809b880cdcee98d7946d05e6bcd30f623f604348 and checks its runnable files against the v3 manifest. Physical replications have uncontrolled start positions.
