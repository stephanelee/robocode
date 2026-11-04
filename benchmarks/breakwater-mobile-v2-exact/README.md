# Authorized Breakwater 0.2 exact-Python run

[Decision and measurements](../../docs/improvements/breakwater-mobile-v2-exact-results.md) · [Immutable launcher and acquisition command](https://github.com/stephanelee/robocode/blob/cb1a2b752b9933b463565f6ddca59889e8249b0a/.github/workflows/breakwater-mobile-v2-exact.yml) · [Original Actions artifact](https://github.com/stephanelee/robocode/actions/runs/41715478408/artifacts/13270650047)

The two game directories contain the original official scores, round JSONL, private endpoints, launched descriptors, stage hashes, runner logs and command lines. The original frozen candidate and opponent pins reproduce the staged binaries and source. `artifact-files.json` hashes all 133 files in the original experiment directory. `retry-provenance/` preserves the actual runtime and both the launcher and payload heads. The Actions artifact SHA-256 is `ae615a9ae66b1c74b78ccd77a53c70aed8a11e292d59cb926835eedc600cd22b`.

Replay the validation offline from the repository root:

```sh
PYTHONPATH=tools python3 -B - <<'PY'
import json
from pathlib import Path
import breakwater_mobile_experiment as runner
base = Path('benchmarks/breakwater-mobile-v2-exact')
report = json.loads((base / 'experiment.json').read_bytes())
for game in report['games']:
    result = runner.validate(base / game['gameId'], game['gameId'], game['seed'], 'aisSeed' in game)
    assert result == game
assert runner.decision(report['games'][:1], 'pilot') == report['pilotDecision']
print(report['pilotDecision'])
PY
```
