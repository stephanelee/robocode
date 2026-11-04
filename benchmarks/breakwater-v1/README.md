# Breakwater v1 evidence

The registered pilot completed once. The decision was **reject** after 20 rounds. [Report](../../docs/improvements/breakwater-v1-results.md) · [Protocol](registered-plan.md) · [PR #66](https://github.com/stephanelee/robocode/pull/66)

- [Raw game directory](breakwater-v1-pilot/) contains all 20 original round records, official result, endpoint, executed command, runner log, staged source hashes, descriptors and license notices.
- [Original experiment output](experiment.json) records the decision, per-game measurements, runtime, package versions and source pins. [Freeze](freeze.json) records the source, JAR and premeasurement runtime hashes.
- [File manifest](artifact-files.json) covers all 117 files in the original output directory, including staged source and launched JAR. Identical staging copies can be recovered from the registered JAR and 110-file upstream lock. The original Actions artifact ID is `13268035643`, digest `sha256:c1de10907a664312764ab868df5743e50ec8c571297d4ef4a082dd6f6ac28507`. [Run](https://github.com/stephanelee/robocode/actions/runs/41708366019) · [Artifact](https://github.com/stephanelee/robocode/actions/runs/41708366019/artifacts/13268035643)

From the repository root, replay the decision offline using the following commands. They verify the retained files against the acquisition manifest and read only the original pilot.

```sh
PYTHONPATH=tools python3 -B - <<'PY'
import hashlib, json
from pathlib import Path
from breakwater_experiment import decision, validate

root = Path('benchmarks/breakwater-v1')
manifest = json.loads((root / 'artifact-files.json').read_bytes())
for path in root.rglob('*'):
    name = path.relative_to(root).as_posix()
    if path.is_file() and name in manifest:
        assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest[name], name
game = validate(root / 'breakwater-v1-pilot', 'breakwater-v1-pilot', 531801)
result = decision([game], 'pilot')
assert result == json.loads((root / 'experiment.json').read_bytes())['pilotDecision']
print(result)
PY
```

The exact acquisition environment, build/test commands, committed-JAR checks, one-shot activation and executed Java invocation remain available at the immutable head. The registered competitive budget is spent. [Workflow](https://github.com/stephanelee/robocode/blob/fced7e405b64de4d059de2cfa36be36be34742d5/.github/workflows/breakwater-v1.yml) · [Java command](breakwater-v1-pilot/command.json)
