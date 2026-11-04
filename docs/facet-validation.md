# Facet validation and evidence

2026-11-04. Work in progress; this is a buildable duel milestone, not the full
specification or a promotion approval. Tracking: [issue 101](https://github.com/stephanelee/robocode/issues/101).

The bot is a new independent implementation from the supplied source-informed
specification. See [input separation](facet-source-separation.md),
[decisions](../experiments/Facet/decisions.md),
[semantic matrix](../experiments/Facet/semantic-matrix.md) and
[prospective protocol](facet-protocol.md).

## Revision 4: integrated correction artifact

[Draft PR 103](https://github.com/stephanelee/robocode/pull/103) contains the reviewable milestone. Gradle Java 17 `build verifyJarParity` passed: **18 groups / 10,034 assertions**, all 866 recorded transitions, package build and 832 matching JAR entries. [Raw Gradle evidence](../experiments/Facet/evidence/gradle-revision4-build.txt). The independent second review required eight further semantic corrections; [its findings](../experiments/Facet/evidence/independent-review-revision2.md) and the [correction mapping](../experiments/Facet/revision2-corrections.md) are retained. The latest artifact has not yet received independent premeasurement approval.

Current JAR SHA-256: `73be04d7386a60474743a78fcebf3fd0c143e37120b410b18aa668bd5ed50c16`.
Current source/fixture/config/build-input SHA-256: `d50a1c8c3a7af6822c81cab79efbe2c5e67314451f9921fedaea3b1f4ae7c73c`.
Configuration SHA-256: `01b33da9bfd40de080214d111578277207dabf45e4a8a1ac9b3f61de4332dc69` (model schema 2).
[Canonical manifest](../experiments/Facet/evidence/artifact-manifest.json) defines the hash algorithm. Earlier filtered-workspace hashes identify their own artifacts; formatting changes debug line tables.

Terminal batch reconciliation, controlled two-wave action change and bounded confirmed-bullet shadow calculations now have passing fixtures. The shadow service grants zero credit to continuous incoming uncertainty. Actual live API terminal dispatch, skipped-turn reconciliation, rejected fire and bot-bot/multi-contact acquisition remain gaps. Melee modules and displacement recency views are implemented but activation remains gated. No competitive strength follows from these fixtures.

User-authorized subsequent benchmark: **five cold 20-round games per AISurvivor, Retcon and Diamond Port, 300 rounds total**. This replaces the unexecuted proposed 200-round roster; it is recorded before observing any Facet result. Runtime pins and independent review must be frozen before collection. No battles have run, and no automatic confirmation or tuning is authorized.

## Revision 2: Java 17 build and fixture milestone

The self-contained Maven/Gradle build passed with Gradle 8.14.3 and Homebrew
OpenJDK 17.0.21.0.3 on macOS 27.2 arm64, using a 256 MiB test JVM. The suite passed
17 groups and 7,119 assertions, including 866 actual recorded engine transitions,
the recorded request/creation/first-motion clock, four isolated energy-visibility
windows, exact-neighbor/brute-force agreement, compatible fresh-scan firing,
cohort chronology, bounded learning/export, lifecycle and deterministic work
quotas. These tests do not establish competitive strength or full adapter
acceptance.

After formatting, the first Gradle invocation correctly rejected the previous
JAR's changed debug-line payload. The source-built JAR was refreshed, Gradle
`build` then passed, and both parity checks passed: 832 matching file entries
with the repository checker; 831 with the bot helper, which excludes the manifest.
The official API payload was checked against its frozen SHA-256.

Exact identities are in the [artifact manifest](../experiments/Facet/evidence/artifact-manifest.json).
Revision 2 JAR SHA-256:
`378ebc23a589c8107eb381924b53736e19bf0a4247969257a301ad2a0434bec5`.
Source/fixture/config/build-input manifest SHA-256:
`64060f9f54a8e0f1994fbfc18dbc6f31623a08063e413de2c6c21e73946ca5f8`.
The earlier filtered-workspace manual-build reports identify their own source
and JAR separately; they are retained development evidence, not the formatted
Gradle artifact's identity.

Raw checks: [Gradle build](../experiments/Facet/evidence/gradle-revision2-build.txt),
[Gradle parity](../experiments/Facet/evidence/gradle-revision2-parity.txt),
[expected bootstrap parity failure](../experiments/Facet/evidence/gradle-revision2-bootstrap-parity-failure.txt),
[filtered implementation tests](../experiments/Facet/evidence/correction-final.txt).
All ten correction attempt outputs are retained beside the final output.

## Independent engine and geometry diagnostics

The independent reviewer acquired one deliberately stopped diagnostic round
with two newly authored own-state probes. The first startup attempt failed
before a game began because child Java was absent from PATH; it is preserved.
The successful acquisition produced 434 decisions per probe, exceeding the
400-turn stop marker because shutdown is asynchronous. All 866 consecutive
transitions passed; maximum position error was 1.37e-12, speed error zero and
heading error below 3.13e-13 degrees. Request turn 55, creation turn 56 and first
bullet motion turn 56 were distinguished. This was not a Facet battle or an
opponent-strength experiment.

[Diagnostic protocol](../experiments/Facet/evidence/engine-contracts/protocol.md),
[validation](../experiments/Facet/evidence/engine-contracts/validation.json),
[raw Alpha](../experiments/Facet/evidence/engine-contracts/Alpha.jsonl),
[raw Beta](../experiments/Facet/evidence/engine-contracts/Beta.jsonl), and
[failed startup](../experiments/Facet/evidence/engine-contracts/attempt-1/invalid.json)
retain the acquisition and its limitations. Raw Infinity tokens are original
API output; normalized fixtures encode them explicitly as strings.

A separate mathematical shadow oracle compared 720,917 angular/edge samples
over 11 independently constructed cases with the actual official server
intersection/broad-phase predicates and found zero false credit. This is bounded
oracle evidence, not a full proof of Facet's now-active bounded shadow service.
See [oracle results](../experiments/Facet/evidence/independent-shadow-oracle/output.json)
and [provenance/limits](../experiments/Facet/evidence/independent-shadow-oracle/README.md).

## Unchanged production checks and open gates

AISurvivor's Java 17 build ran with its JAR output redirected to a temporary
directory. Source/committed-JAR parity passed for 822 entries. Its unchanged
checkout had 89 tests, five strict replay failures and one skip; failures show
tiny gun-angle floating-point differences on this Mac. See the
[raw build log](../experiments/Facet/evidence/ais-java17-build.txt).
Production source, settings, JAR and checkpoints were not changed, and the
existing parity requirement was preserved. No JDK 25 Retcon launch occurred.

Independent review rejected the initial implementation for premeasurement.
[Review findings](../experiments/Facet/evidence/review-revision2.md) are retained;
passing the correction tests is not an independent approval. At revision 2 the remaining gates included terminal reconciliation, shadow service and controlled two-wave action change. Revision 4 adds these fixtures; live adapter contracts, independent review and required CI remain open. Melee activation is gated until
duel semantics and review pass and needs its own prospective screen.

Facet competitive games/rounds: **0 / 0**. The superseded proposed 200-round screen and newly requested 300-round benchmark have not
run; there are no score shares, wins or improvement claims. No parameter search,
offline training dataset, warm start, confirmation rerun or reserved experiment
was consumed. Decision: **reject premeasurement/promotion at this milestone;
continue the semantic implementation**.
