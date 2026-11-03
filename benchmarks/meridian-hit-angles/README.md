# Meridian hit-angle screen

The execution commit is `c1fd38adab41bad51d707c84edbaba4d22c74d45`. [Protocol and results](../../docs/improvements/meridian-hit-angles.md) · [game IDs](games.json) · [runtime manifest](profile.json) · [independent pre-execution review](https://github.com/stephanelee/robocode/pull/79#pullrequestreview-5760714347) · [registered execution](https://github.com/stephanelee/robocode/actions/runs/41735907101).

The fresh control was built from the already-verified v1 bundle on PR #77. All four runnable assets are extracted from its immutable Git commit. The candidate uses the committed 0.2.0 JAR after a clean Java 17 parity check. The [workflow](../../.github/workflows/meridian-hit-angles.yml) contains the runner commands, private staging paths and exact profile check. The OS build and Python interpreter come from the [measured preflight](https://github.com/stephanelee/robocode/actions/runs/41735518958). The runner and both arms use the same verified environment after the switch to Java 25.

From the repository root, verify the retained screen without playing another game:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=tools python3 - <<'PY'
import json
from pathlib import Path
from meridian_duels import metrics
from meridian_hit_angles import passes_screen
root = Path("benchmarks/meridian-hit-angles")
raw = root / "raw"
assert (raw / "profile.json").read_bytes() == (root / "profile.json").read_bytes()
plan = json.loads((root / "games.json").read_text())
decision = json.loads((raw / "decision.json").read_text())
screen = []
for game, row in zip(plan, decision["games"], strict=False):
    assert all(row[k] == v for k, v in game.items())
    if row["status"] == "valid":
        values = metrics(raw / row["id"], game)
        assert all(row[k] == v for k, v in values.items())
        print(game["id"], values["wins"], values["share"], values["damagePer100EnemyShots"])
    if game["stage"] == "screen":
        screen.append(row)
assert len(screen) == 4
assert (decision["decision"] != "reject at screen") == passes_screen(screen)
PY
```

For a new physical replication, follow the workflow on the registered commit with a clean checkout including Git history, Java 17 for the Gradle build, then Java 25 and the pinned Python interpreter. Reacquire the exact profile with `python tools/meridian_profile.py`, compare it with the registered file, then use `python tools/meridian_hit_angles.py`. Starts are uncontrolled by the runner. A replication produces separate evidence.
