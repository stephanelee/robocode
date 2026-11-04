# Facet validation and evidence

2026-11-04. Work in progress; this is a buildable duel milestone, not the full
specification or a promotion approval. Tracking: [issue 101](https://github.com/stephanelee/robocode/issues/101).

The bot is a new independent implementation from the supplied source-informed
specification. See [input separation](facet-source-separation.md),
[decisions](../experiments/Facet/decisions.md),
[semantic matrix](../experiments/Facet/semantic-matrix.md) and
[prospective protocol](facet-protocol.md).

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
oracle evidence, not a passing result for Facet's disabled shadow service.
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
passing the correction tests is not an independent approval. Remaining gates
include terminal event reconciliation, a validated runtime shadow solver, a
controlled native two-wave action-change fixture, additional live adapter
contracts, independent review and required CI. Melee activation is gated until
duel semantics and review pass and needs its own prospective screen.

Facet competitive games/rounds: **0 / 0**. The proposed 200-round screen has not
run; there are no score shares, wins or improvement claims. No parameter search,
offline training dataset, warm start, confirmation rerun or reserved experiment
was consumed. Decision: **reject premeasurement/promotion at this milestone;
continue the semantic implementation**.
