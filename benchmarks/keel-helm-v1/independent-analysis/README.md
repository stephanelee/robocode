# Independent KeelHelm outcome audit

This is a standalone standard-library reducer written before any KeelHelm game.
It does not import or execute the bot, benchmark harness, harness reducer, server,
opponent, SDK motion helpers or candidate implementation. It reads existing files
only. Its ownership is separate from candidate and harness implementation.

The registered protocol and `../games.json` remain authoritative. This directory
opens no games, replacement rounds, retry, tolerance/criterion change or stage
clearance. Source/JAR review and the actively assigned same-worker runtime review
are separate gates; this outcome auditor does not supply them.

## Commands

From the repository root, after downloading and preserving a stage's original ZIP:

```sh
python3 -I -B -S benchmarks/keel-helm-v1/independent-analysis/audit.py verify-archive \
  --zip /absolute/original-artifact.zip --sha256 API_ORIGINAL_ZIP_SHA256 \
  --extracted-root /absolute/original-extraction --output /absolute/archive-audit.json

python3 -I -B -S benchmarks/keel-helm-v1/independent-analysis/audit.py audit-stage \
  --stage screen --head FULL_EXECUTION_HEAD --run RUN_ID \
  --evidence /absolute/extraction/build/keel-helm/screen \
  --source-root /absolute/exact-reviewed-source --output /absolute/screen-independent-audit.json
```

Use `--stage confirmation` for its own fresh population. `--evidence` names the
stage directory containing `games/`, `profile.json`, accepted approval/receipt
files and the `provenance/` subdirectory. `--source-root` supplies the exact
registered game registry and fallback source freeze, not a changed population.
Retain complete command, interpreter identity, stdout/exit code and output hashes.
A nonzero exit means reject, incomplete or missing provenance; do not repair or
repeat a battle in response. The output distinguishes raw-complete reduction,
registered gate results, missing/partial games and offline provenance checks.

## Independently reduced checks

- Exact official duel setup, frozen participant names/versions and IDs; contiguous
  round ticks, unique ordered endpoints, final/round cumulative score and first-
  place reconciliation. Fractions preserve exact equal-game means and percentage-
  point boundary comparisons rather than pooling rounds or using floating gates.
- Per-round/game score share, first places, own/enemy shots/hits/uncapped bullet
  damage/shot energy, power mix, hit rates, walls, bot collision events, bullet
  collisions, turn count and firing rate. Bot contacts count official events with
  `botId` equal to own ID, matching the published Keel definition.
- Exactly one start/stats/end per completed round, nonnegative integer `maxMs`,
  all skip markers and the `<75`/zero-skip guard for both arms. Candidate-only
  exposure counters are independently summed; baseline exposure is N/A.
- Registered stage order/populations and every engineering progress/confirmation
  gate. Missing/invalid games never receive replacement outcomes. Known skip or
  timing observations survive even an incomplete recording. Round IDs observed
  in partial recordings are a **lower bound**, never proof that unrecorded rounds
  did not run. Data after a first invalid/timing failure is flagged.
- Actual AIS cold TRAIN seed/mode/checkpoint fields and stored model completion
  when AIS is an opponent. No persistent Keel model file is invented.
- Exported frozen source/JAR bytes, profile/source-freeze equality, prepared command
  population/input manifests, clean environment identities, strict approval
  bindings and fixed annotated receipt identity/admission fields. The original ZIP
  verifier checks its independently obtained API digest and every extracted regular
  member byte while rejecting path escapes, symlinks and duplicate ZIP members.

## Remaining manual/API evidence review

An offline file reducer cannot prove current remote API state or hash live files
that were not exported. Independently read actual run/job/attempt and permanent
ref/tag API metadata, full original job/runner logs, artifact API digest, current
approval/dispatch/report API records and their archived revalidations. Establish
one execution/receipt, before-game review, actual deadline/admission/timeout,
fresh process/cold writable directory state, teardown and physical budget from
all evidence. Source/JAR parity and exact native/JDK/Python manifests also need
review of their original build/provenance records. These limits are explicit in
the JSON output; an offline `PASS` must not be called a complete competitive
strength or merge clearance by itself. The future stage files remain unexercised
until actual evidence exists; interface adaptation may correct parsing only,
never reinterpret registered rules or replace raw outcomes.

## Verification already performed (not KeelHelm results)

`verification/historical-keel-confirmation.json` independently replays the six
**already published immutable Keel0.1.0** confirmation recordings, 120 rounds.
All score/win/per-round share/turn/shot/hit/damage/energy/contact/hash fields
reconcile with the old published decision with zero mismatches. This verifies
recording parsing only and is not a new contemporary control or KeelHelm result.
No candidate/harness reducer was read or imported for this replay.

`test_audit.py` supplies pure synthetic valid/corrupt observer, marker and ZIP
fixtures, including missing ticks/endpoints, unknown identities, duplicate stats,
negative/high timing, skips, partial recordings, exact gate boundaries and unsafe
archive members. They run no game, process, server or motion helper. Logs under
`verification/` preserve their results; these tests do not establish strength.
