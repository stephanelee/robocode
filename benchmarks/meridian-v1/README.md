# Reproducing the Meridian screen

[Protocol](../../docs/improvements/meridian-v1.md) · [game schedule](games.json) · [runtime manifest](profile.json) · [independent pre-execution review](https://github.com/stephanelee/robocode/pull/77#pullrequestreview-5760583065) · [registered run](https://github.com/stephanelee/robocode/actions/runs/41733889701).

The immutable registered commit is `d859bce650cf8881492e75c66ac69a1ab5daed01`. Its workflow first built on Java 17, then launched the runner and Retcon using JDK 25. The hosted [profile-only preflight](https://github.com/stephanelee/robocode/actions/runs/41733510872) spent no games. Ports and bot RNG seeds are fixed, but engine starting positions are not seeded through this runner; replaying conditions is not bit-identical replay.

**Offline verification.** From the repository root, inspect each preserved recording and recalculate its game metrics without starting bots:

```sh
PYTHONPATH=tools python3 - <<'PY'
import json
from pathlib import Path
from meridian_duels import metrics
root = Path("benchmarks/meridian-v1")
for game in json.loads((root / "games.json").read_text()):
    folder = root / "raw" / game["id"]
    if folder.is_dir() and list(folder.glob("*.battle.gz")):
        print(game["id"], metrics(folder, game))
PY
```

**Reproduce the setup.** In a separate checkout with the exact Linux image and toolchains in the manifest, inspect the [archived registered workflow](source/meridian-registered-duel.yml). Its exact executed bytes are retained here; the completed acquisition launcher has been retired from active Actions, while ordinary build/test/JAR-parity CI remains enabled. This is historical setup documentation, not authorization for another registered run. The equivalent commands, including private staging, are:

```sh
# With Temurin 17 active and the repository's wrapper:
mkdir -p build/meridian
java -version 2>build/meridian/build-java.txt
sha256sum "$JAVA_HOME/bin/javac" "$JAVA_HOME/lib/modules" | sed 's@  .*/@  @' >build/meridian/build-java.sha256
./gradlew --no-daemon -p experiments/Meridian clean build
python3 tools/verify_jar.py experiments/Meridian/Meridian.jar experiments/Meridian/build/bot/Meridian/Meridian.jar
# Switch JAVA_HOME and PATH to Temurin 25. With the pinned Python 3.13:
python3 tools/meridian_profile.py
cmp benchmarks/meridian-v1/profile.json build/meridian/profile.json
python3 tools/meridian_duels.py
```

The last command consumes a new set of physical games; never use it to replace registered evidence. The official scores and full tick stream are in each `.battle.gz`. Shot energy counts actual bullet creation, not requests. Damage totals are uncapped event damage, not the official capped bullet score. Collision counts use the single Meridian-directed contact record per collision. Skip counts are Meridian's private callbacks, which the public recorder does not receive.
