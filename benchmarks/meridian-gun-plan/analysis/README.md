# Independent outcome audit

This audit imports no repository benchmark reducer. It independently counts official events and reconciles every game and round with the original machine decision, verifies exact staged source/JAR files and opponent locks, checks cold AIS checkpoints, and verifies runtime/package/host/receipt bindings. It also checks equal-game means, sample standard deviations and differences. It starts no games.

Run from any directory, with Python 3.10+ and the repository available:

```sh
python benchmarks/meridian-gun-plan/analysis/independent_audit.py --output /tmp/meridian-independent-audit.json
```

The default artifact directory is `<repository>/build/acquisition-41766660321`. It must contain the original `artifact.zip` (SHA256 `0e63d31532e2cd03657d14352f69c80b111a148ae3543457b4542a66e070facd`) and its complete extracted tree in `extracted/`. Obtain the full artifact from [run 41766660321](https://github.com/stephanelee/robocode/actions/runs/41766660321) before its 90-day hosted retention expires, or use the locally preserved original ZIP. The curated committed raw folder omits duplicate staged source trees and is insufficient by itself for this full audit.

Use `--repository /path/to/robocode --artifact-dir /path/to/full-artifact` for other locations. `--output` is optional; without it the script only validates and prints the game summary. Assertions failing cause a nonzero exit.

`original-audit.json` preserves the independent initial recomputation. `provenance.json` records script/output/artifact hashes and the separately checked public GitHub run/receipt identity. Those live remote checks are documented evidence; the portable script does not query GitHub. The script cannot prove that no external, unrecorded battles occurred. It verifies the twelve-game preserved execution and its fixed receipt.
