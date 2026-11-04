# Breakwater 0.3 paired diagnostic and competitive evidence

[Report](../../docs/improvements/breakwater-envelope-v3-exact-results.md) · [Executed workflow and acquisition commands](https://github.com/stephanelee/robocode/blob/e5780d81ebf6ae78bead3f719242c20380dbf84d/.github/workflows/breakwater-envelope-v3-exact.yml) · [Original Actions artifact](https://github.com/stephanelee/robocode/actions/runs/41721148640/artifacts/13272557069)

Artifact SHA-256 `395b0f2d23abb9550799e9841d79f6cd2703c459dec97cdbc8e4f33105cbb62e`. The per-game directories retain every official scoreboard, command, runner log, round stream, private endpoint, staged hash list, descriptor and license needed to audit the run. The two large shadow JSONL streams are losslessly gzip-compressed. The manifest hashes all 249 original experiment output files, including uncompressed traces and reproducible staging copies. The launcher/checkout heads and actual measured runtime are in `retry-provenance/`.

From repository root, replay both proposed rays and all masks, verify retained bytes and recalculate the decision offline:

```sh
PYTHONPATH=tools python3 -B - <<'PY'
import gzip, hashlib, json, shutil, tempfile
from pathlib import Path
import breakwater_envelope_experiment as runner

src = Path('benchmarks/breakwater-envelope-v3-exact')
with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    shutil.copytree(src, root, dirs_exist_ok=True)
    for packed in root.rglob('*.jsonl.gz'):
        packed.with_suffix('').write_bytes(gzip.decompress(packed.read_bytes()))
    for name, digest in json.loads((root/'artifact-files.json').read_bytes()).items():
        path = root/name
        if path.is_file():
            assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, name
    report = json.loads((root/'experiment.json').read_bytes())
    for game in report['games']:
        gid, seed = game['gameId'], game['seed']
        stage = next(s for s in ('shadow','pilot','confirmation','transfer') if (gid,seed) in runner.layout(s))
        assert runner.validate(root/gid, gid, seed, stage) == game
    assert runner.decision(report['games'][1:2], 'pilot') == report['diamondDecision']
    print(report['diamondDecision'])
PY
```
