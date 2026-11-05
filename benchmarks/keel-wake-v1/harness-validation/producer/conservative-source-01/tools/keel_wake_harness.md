# KeelWake registered producer

The prospective authority is [issue118 / PR119](https://github.com/stephanelee/robocode/issues/118) and [the published protocol](../docs/improvements/keel-wake-v1.md). This producer introduces no additional games. Screen is exactly14games/120rounds; conditional confirmation is18freshgames/300rounds. Fixed IDs, seeds, ports, opponents, paired baseline and score gates come from the published registry. Independent source review, actual same-worker provenance approval and a create-only stage receipt are separate admissions. No fixture result supplies any admission.

## Source and preparation

Every job uses ubuntu-26.04. The registered workflow builds and tests AISurvivor, original Keel and KeelWake with Temurin17.0.21+9 and checks committed-JAR/source parity before switching to Temurin25.0.5+8 for Retcon and execution. CPython3.13.17 and the registered native/wheel/source graphs are mandatory. Checkout does not retain credentials; game children receive an explicit environment without GitHub credentials or Java/Python injection variables.

`freeze` is write-once and remains unavailable until final source/DATA review and actual JAR publication. It covers registered inputs, sources/JARs, executable tools, workflows, independent decoder and DATA. The canonical exported files are `provenance/profile.json` and `provenance/freeze.json`. Optional stage-root aliases must be whole-byte identical; they never replace missing canonical files. Preparation owns fresh outputs and freezes actual installed bytes; later checks verify read-only and never repair a changed environment.

## Authority and complete revalidation

A root-assigned independent task must be active before dispatch. The exact10-key `KEEL_WAKE_DISPATCH_ASSIGNMENT_V1` binds that task, implementation tasks, actual tool record and storage observation. The executing worker exports zero-game provenance, whose finalized original artifact identity/digest must precede the independent four-key reasoned report and20-key approval. Initial approval must be created within1800seconds of artifact finalization; this does not expire an immutable timely approval during later legitimate stage checks. Re-fetches reject edits, deletion, revocation, changed source/profile/input bytes, wrong current PR head and receipt changes.

Each before-receipt and before-child check owns a fresh Python process group. A parent real-time alarm bounds the complete check to180seconds, including local reads/hashes, subprocesses, HTTP and evidence writes; the child has a slightly shorter process-group timeout and is killed/reaped on failure. Each I/O uses the minimum of its existing cap, remaining check budget and original job time. END `checkedAtUTC` is recorded only after the worker exits. END must retain at least2100seconds of the original340minute job. Start/end/monotonic duration, source-input revalidation and every raw HTTP observation are retained.

Raw FETCH bodies live at `<phase>-fetches/NNN.json`; the strict10-key observation binds endpoint, method, start/end UTC, status, original response path/size/SHA, transport completion and error. The sole permitted accepted404 is complete expected absence of the new fixed receipt before consumption. Post-receipt404 and other failed/partial responses cannot authorize. The current authority objects and original job deadline bind to these raw bodies.

Whole-second provider completion T is an interval[T,T+1). Without separately reviewed causal proof, captures must end at or before T. No implicit one-second grace is used. Timestamps finer than microseconds are rejected at ordering boundaries instead of truncated. A precise event-order alternative remains a separate reviewed contract; this producer currently supplies no such fallback.

## Image and final external identity

The exact policy `keel-wake-qualified-native-observations-v1` permits only its two complete literal joint cases. Both native files and both separate environment observations must match one case. Before-tools/prepared/live snapshots bind the same case and frozen actual source/runtime graph. This is qualified native evidence, not canonical VM attestation. An actual early job-log200 with matched full header or complete current404 with independent API access must be retained in the pregame report. Final independent full provider run/jobs/log reconciliation remains mandatory; raw decisions retain `PROVENANCE-PENDING` until then.

Final independent reviewer identity is an explicit external `provider/reviewer-identity.json` sidecar, referenced by exact path/bytes/SHA. It binds the original10-key predispatch comment, assignment body/tool hashes, actual review tool evidence and the same non-implementation task. Producer-generated facts cannot supply independent PASS or manufacture a final reviewer. No decoder-author name is hardcoded as the reviewer.

## Counters, results and custody

Candidate round stats contain exactly `matchedDangerLabels,refreshEligiblePlans,refreshedWaves,densityChangedWaves` in addition to the ordinary stats. Each is a JSON integer0..1000000. Require changed<=refreshed, plans<=refreshed<=2*plans and zero refresh counters when matched labels are zero. Baseline diagnostics are N/A. `ROUND diagnosticOverflow` or a private skip immediately stops the current owned acquisition; observer-only skips are checked at completed-game boundary before another child. Missing/malformed counters invalidate; no default zero is substituted.

All scheduled valid games finish regardless of scores. Exact rational score arithmetic implements the registered Diamond+2pp and Retcon-2pp guards, stage-specific wins/consistency and transfer checks. Stage-local Diamond densityChangedWaves must be positive; zero exposure is inconclusive and preserves score failures. These counters do not establish strength or causation. First invalid stops; no retry, replacement, resume or unspent-budget conversion is available.

The final original ZIP, raw decision and partial failures remain immutable. A separate finalized-custody artifact binds the original artifact API identity/digest/size and decision SHA. Independent raw replay and final provider reconciliation remain required. Missing custody is incomplete, never automatic PASS.

Confirmation additionally verifies a nine-key passing-screen gate and lossless retained original ZIP parts/member/source manifest. Every gate, custody manifest, retained evidence file and ZIP part is included in each confirmation prepared game's `inputFiles` and explicit `upstreamInputBindings`, bound by approval.commandsSHA256. Original bytes are exported to `provenance/upstream-inputs/` with an exact manifest. Screen replay uses its original exported source/profile/freeze, not the fresh confirmation graph. No execution freeze is rewritten after seeing a screen result.

## Pure checks

`python -I -B -S -m unittest discover -s tools -p 'test_keel_wake*.py' -v` runs synthetic producer fixtures only. Actual commands and outputs, including failures, are retained under `benchmarks/keel-wake-v1/harness-validation/producer/`. The independent decoder has separate tests and ownership; this producer does not import it. These checks invoke no engine, games, installed runtime preparation or actual provider API requests. Actual pinned hosted source CI, independent review, final freeze and same-worker runtime approval remain outstanding.
