# Reproduce the Meridian frozen-prior test

[Protocol and results](../../docs/improvements/meridian-prior.md) · [manifest](games.json)

From the repository root, check the recorded runtime, recompute each valid row from the official recording and bot markers, and verify that confirmation was run only if the screen passed:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=tools python3 - <<'PY'
import json
from pathlib import Path
from meridian_prior import passes_screen
from meridian_duels import metrics
root = Path("benchmarks/meridian-prior")
raw = root / "raw"
assert (raw / "profile.json").read_bytes() in [(root / n).read_bytes() for n in ("profile.json", "profile-20261026.json")]
plan = json.loads((root / "games.json").read_text())
report = json.loads((raw / "decision.json").read_text())
rows = report["games"]
assert len(rows) == (len(plan) if passes_screen(rows[:5]) else 5)
for game, row in zip(plan, rows):
    assert all(row[k] == v for k, v in game.items())
    if row["status"] == "valid":
        actual = metrics(raw / game["id"], game)
        assert all(row[k] == v for k, v in actual.items())
        print(game["id"], row["wins"], row["share"], row["damageDealt"] / row["shots"])
PY
```

The [workflow](../../.github/workflows/meridian-prior.yml) rebuilds on Java 17 and checks JAR parity, then uses Java 25 to launch matches. Both arms use the same Ubuntu 26.04 host. The runner extracts the v3 control from 809b880cdcee98d7946d05e6bcd30f623f604348 and checks its runnable files against the v3 manifest. Physical replications have uncontrolled start positions.

Recreate the bank with `PYTHONDONTWRITEBYTECODE=1 python3 tools/meridian_prior_bank.py` on the original generation host. On other hosts use `PYTHONDONTWRITEBYTECODE=1 python3 tools/meridian_prior_bank.py --verify` to compare all reconstructed values while retaining the frozen resource bytes.

The `parity/queries.txt` rows encode bearing (degrees), orientation, range, power, the four unscaled context features, and the Python reference aim (degrees). Compile `parity/PriorParity.java` against the committed Meridian JAR and run it with that query file. All 282 reference predictions must agree within 1e-8 degrees.
