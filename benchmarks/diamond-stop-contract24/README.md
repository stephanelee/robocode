# Diamond STOP24: prospective source-contract diagnostic

Status: [PR #100](https://github.com/stephanelee/robocode/pull/100) implements the
[issue #99](https://github.com/stephanelee/robocode/issues/99) registration.
Hosted zero-case preparation [41818421304](https://github.com/stephanelee/robocode/actions/runs/41818421304)
passed independent review. Exact-head review then cleared the sole numeric attempt,
but acquisition workflow [41843951756](https://github.com/stephanelee/robocode/actions/runs/41843951756)
failed the immutable runtime-profile gate before activation. **0/24 numeric cases
and 0 games executed; no STOP prediction result was obtained.** See the
[detailed result](../../docs/improvements/diamond-stop-contract24-results.md),
[complete failure evidence](failed-preparation-41843951756/manifest.json) and
[independent failure audit](failed-preparation-41843951756/independent-acquisition-review.md).
[Protocol](../../docs/improvements/diamond-stop-contract24.md) fixes24 one-step
cases, zero games and one acquisition attempt. [cases.json](cases.json) contains
inputs only, including exact float representations; no outputs were calculated.
The original predictor, actual SDK controller/clamps and pinned engine pure
helpers plus source-derived translation/wall composition are now bound in the
write-once [freeze](freeze.json), which remains unchanged after the failure. There is no proposed correction,
new bot, live budget or strength gate. Keep all mismatches and failures.

The durable acquisition ref is
`refs/tags/experiment-receipts/diamond-stop-contract24-v1`; it must be absent before
one atomic create, and never reset/delete/reuse afterward. Zero-case profile
preparation does not consume it. No extra numeric smoke calls are permitted.

The [evidence manifest](profile-evidence-41818421304/manifest.json) retains all29
original artifact files, the complete original ZIP in verbatim parts, full job
log, static audit and independent reviews. The original server JAR is also split
verbatim so every retained file is at most2MiB. Reassemble/check all original
bytes, the complete embedded profile and unchanged30-tool closure with:

```sh
python3 -I -B -S benchmarks/diamond-stop-contract24/profile-evidence-41818421304/verify.py
```

The original profile SHA256 is
`dfbd5df2660bf5207144aa5990c2c019ce3662d1fd6efa93b2f9d65567a7ca1d`;
freeze SHA256 is `be6e1bc1271f5e5e0f1d2d1f76b008ed47125c6a3797876788adeb949e242c92`.
The selected runtime is Ubuntu26.04 image20261101.186.1, Temurin17.0.21+9,
CPython3.13.18 and package manifest52b7…. Preparation compiled the bridge and
collected/verified metadata only; tested SDK/port imports, bridge main and engine
numeric helpers remain unexecuted. The fresh acquisition must reproduce the
entire frozen profile before consuming the sole receipt. This checkpoint makes
no STOP-option, gameplay or strength claim.

The fresh job received image20261026.180.1 instead of exact frozen20261101.186.1;
181 profile leaves differed. The receipt remains unused. Do not rerun or relabel
until a favorable image appears. Any changed runtime plan/freeze/payload requires
a prospective amendment and independent review. Verify the complete original
failed-run ZIP and all72 original files without evaluating cases:

```sh
python3 -I -B -S benchmarks/diamond-stop-contract24/failed-preparation-41843951756/verify.py
```
