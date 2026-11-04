# Facet prospective protocol

2026-11-04. Tracking: [issue 101](https://github.com/stephanelee/robocode/issues/101).
This is a prospective plan, not a record of competitive results or a frozen
acquisition authorization. Specification v1, dated 2026-11-04, governs the gates.

Hypothesis: independently implemented contextual nearest-neighbor targeting,
online virtual selection and circular two-wave movement can satisfy the stated
engineering screen against several opponent families. No improvement, Diamond
equivalence or tournament-strength claim is made.

Before collection, freeze a review-approved candidate's source, committed JAR,
configuration/schema, rule profile, runner, opponent locks and runtime hashes in
the evidence manifest and issue. Source and JAR parity, actual engine chronology,
STOP/reversal/first-step wall safety and the complete semantic matrix are gates.
The independent reviewer must explicitly approve premeasurement. Any unresolved
semantic blocker prevents the screen. Melee has a separate later review/screen.

The official server/runner and API are version 1.4.0. Runner SHA-256:
`02f6d1d8e9346a4aeae1b91e7abb278c98ac1f1d30c605f867758566f5f327cf`.
API SHA-256:
`ab65c4d5cec1808adeb71375adae6d15341ae250def89a6c10fc9da879de0752`.
Official engine source commit:
`c8ad3a8d19a843f6258d6f6f9db7f29229963903`.

Opponent identities are the unchanged repository locks. Diamond Port 1.8.28
and Adaptive Prime use commit `67b68c22e41c18281a3fce9ef638e3c23a0b47cf`
and the existing strong-melee lock; its aggregate source digest covers both
selected opponents and shared code, not Diamond alone. Retcon 0.2.0 uses commit
`28981c7a469ac064194b41001cd7183ec5ecf189` and source SHA-256
`5f7dfaaedefc39bd93d86440a2996342583d57b75b861942dc1e1899b7164bb2`.
Fast samples are the existing pinned official Target and Walls from sample
archive SHA-256
`1707c1011a37c6b854fc79d68549ab204007bac6d33d803fa254152ac0765fe2`.
Opponent execution must remain black-box. No source/model access is granted to
Facet or its implementer.

The initial specification proposal of 200 rounds was **not run**. Before any outcomes, the user superseded the roster and budget with the following **300-round benchmark**, explicitly confirmed as 100 rounds per opponent:

| Cohort | Independent cold games | Rounds/game | Total |
| --- | ---: | ---: | ---: |
| AISurvivor, unchanged production JAR | 5 | 20 | 100 |
| Retcon | 5 | 20 | 100 |
| Diamond Port | 5 | 20 | 100 |

Use Facet seeds 49667, 49668, 49669, 49670 and 49671 by game index. AISurvivor uses default TRAIN behavior in isolated cold state paths, with the same per-game seed. Existing production settings/checkpoints are not changed or consumed. Retcon and Diamond start in fresh processes. Adaptive Prime and official sample transfer remain unmeasured requirements of a later promotion assessment; this requested roster cannot establish them.

No warm start or cross-game file loading. Online learning persists between
rounds in one game. Give each game fresh private model/metrics/trace paths and
record their initial/final hashes. Preserve every attempt, including invalid
acquisitions, timeouts and crashes. No retuning, automatic retries, favorable
subsets or replacement games after viewing outcomes.

Use isolated embedded official runners and never interactive port 7654. Pair
launch order where supported (Facet/opponent, opponent/Facet, Facet/opponent, opponent/Facet, Facet/opponent).
Launch order is not proof of controlled engine seating or initial positions.
The current harness does not establish a controllable server seed; record
starting-position, seat and scheduling limitations before collection. Freeze
actual OS, architecture, JVM builds and turn budget before review; current Mac
offline diagnostics do not establish Ubuntu latency. Build/test under Java 17
before selecting JDK 25 for unchanged Retcon. Every Actions job uses exactly
`ubuntu-26.04`. No JDK 25 runtime or complete match environment is frozen yet.

All scheduled games must complete with valid identities and artifacts. Reject
the profile if any cohort has a numerical failure or unhandled exception, skips
at least 0.5% of its bot turns, or combined wall/bot collisions at least 1% of
its bot turns. Mean equal-game score share must exceed 40% against Diamond and
Retcon. AISurvivor results are descriptive: no preexisting specification threshold for this new cohort is invented. The omitted Adaptive/sample gates remain outstanding for promotion. Every gate is
binding. Failure rejects promotion; passing establishes only a passed screen.

Report individual whole-game scores/shares, official first places, round-index
learning curves, damage, requested/accepted/confirmed shots and physical hits,
bullet contacts, energy economy, collisions/skips, range, scan ages, ambiguous
inferences, gun ratings, subsystem latency quantiles/GC and all fallbacks.
Rounds sharing a learning population are not independent observations.

No confirmation is automatically scheduled. Any opponent-specific strength
claim needs a separate fresh registered confirmation: initially ten cold
20-round games, majority official first places, and a 95% game-cluster bootstrap
lower bound above 50% mean equal-game score share, 10,000 resamples with fixed
analysis seed. Such a claim would remain opponent/regime specific.

Engine-contract diagnostics are a different activity: independently authored
probes, prospective bounded protocol, legal own event traces and no competitive
interpretation. Preserve their failed startup and asynchronous-stop overrun.
See the validation report/evidence bundle for actual counts. No Facet performance
battles have run. Current requested budget is 300 rounds, with no automatic replacement or confirmation. Update improvements.md after each diagnostic/experiment and
link its PR, report and raw evidence before declaring it complete.
