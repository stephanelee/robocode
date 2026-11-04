# Wave-window geometry passes; route advancement fails

**Retain the diagnostic evidence; reject advancement of this mask-only candidate.**
The single registered zero-game acquisition completed validly:2941 poses,5882
contexts/trajectories and294100 forecast rows. It took14.72seconds wall time.
The corrected support window omitted0 oracle first contacts versus5 for the old
window, but chosen-route exposures rose1208→1209 and the registered eligible
wall-constraint count rose16326→16358. Both advancement guards fail. No live game,
rerun, threshold change or promotion follows. [PR94](https://github.com/stephanelee/robocode/pull/94)
· [issue93](https://github.com/stephanelee/robocode/issues/93)
· [prospective protocol and pre-output clarifications](meridian-wave-support.md).

## Registered outcomes

| Metric / guard | Measured result | Disposition |
|---|---:|---|
| States / contexts / trajectories / position rows |2941 /5882 /5882 /294100| Full frozen population retained |
| Actual source route-score bit comparisons |11764, all exact| Pass |
| History / future segment checks |195318 /150876| Within frozen budget; contacts/wall removal censor each ray |
| Eligible / excluded state-ray hypotheses |7877 /6824| All exclusions retained, none silently dropped |
| Common eligible ray/context cohort |15754| Same denominator for selected-route exposure |
| Old / corrected omitted oracle first contacts |5 /0| Geometry diagnostic passes |
| Retired-wave future contacts |0| Existing retirement36 unchanged |
| Literal pinned / distance-oracle disagreements |0| Validity passes |
| Changed action contexts: all / eligible |93 /53| Nonzero eligible intervention guard passes |
| Selected first-contact exposures: old / corrected |1208 /1209| **Fail: not strictly fewer** |
| Selected wall-constraint steps: eligible contexts, old / corrected |16326 /16358| **Fail: increased by32** |
| Selected wall steps: eligible-ray weighted, old / corrected |68874 /69018| Descriptive only; not the wall guard |
| Selected wall steps: all contexts, old / corrected |19938 /19802| Descriptive only; not a rescue cohort |
| Same-input motion comparisons / mismatches |294100 /0| Sampled conformance veto does not trigger |

The wall guard counts a constrained forecast step once per eligible
(stateID,committedSide), including repeated outward constraint steps. It is not
an actual game wall-event count. The all-context wall total fell, but the frozen
eligible-context guard rose. Do not substitute the favorable cohort.

The analytic case reproduced its declared continuing first-step contact, old
omission throughout the50 projected ticks, corrected first-contact admission,
and reversing first-step miss. This does not assert the reversing route remains
safe for50ticks. Full later contacts and both route scores are retained.

## What the geometry result means

The necessary support F−v−18≤r≤F+18 covers the observed omitted contacts on these
fixed hypothetical paths. It is not a sufficient collision predictor or a model
of Diamond's gun. The unchanged smooth risk kernel accumulates repeatedly over
admitted ticks, so changing support can change the relative scale of modeled
waves, switching and wall costs. The results do not isolate a causal explanation
for the unfavorable selected-route counts. No weighting or normalization was
retuned after seeing them.

Every contact/exposure figure is conditional on a finite synthetic population,
five predetermined hypothetical headings (one in the analytic case), cold
head-on/linear prior, stationary source and the existing forecast. Histories
outside the arena or already ended by contact/wall removal are excluded from
future-contact comparisons with explicit reasons. There are no trained danger
profiles, hidden opponent inputs, altered simulator actions after a hit, enemy
motion, damage/energy responses or bot-to-bot collision model. Correlated rows
and duplicate committed-side contexts are not independent samples. Wins, score
share, damage, firing and hit rates are **not applicable:0 game rounds**. Fixture
contact exposure is not a competitive hit rate or strength measurement.

## Forecast conformance and its coverage limit

The independent two-pass reference compared the same incoming forecast pose and
same command at every stored step. Both policies and their argmins retained the
unchanged one-pass forecast. No physics fix was bundled into this candidate.

All sampled transitions matched within1e−9. The independent audit found21442
forecast wall-constraint steps and196 raw translations crossing both axes, but
**zero first-pass results outside the legal centre box**. Thus the matrix did
not enter the known oblique interpolation failure. Zero sampled mismatches does
not repair or disprove the source-level missing-second-pass defect, and it is
not a general simulator-conformance claim. The source finding and earlier
[dual-aim correction](dual-aim-v1-invalid.md) remain valid. Any separate physics
repair needs its own prospective source/test/game protocol.

## Reproduction, evidence and provenance

Acquisition source head e57d71c86789af43e02a4d107fae79054b57cfee,
tree052e4b58bbe0946b80e0ad45df1d9822dedbc010; freeze
093a4ad1e203b16d73461a751e6d7c761bebc316cc41d6894bf2b47d71cd1403;
diagnostic JARaa3889096a5a9b60b62e1f94a8b4fc06446884834df0298bb116d0f699b748ef.
Unchanged main source4a02dfd shares the exactv3 route seam809b880. Runnable
main JAR7667ef48974e67fe1e42ecee15005905bfe915c4c4afe7cba8e52c146ec2cafe
is preserved at the acquisition head; the original4a02dfd milestone had kept
its old committed JAR, so use this head's verified artifact, not that earlier
stale JAR. The held gun was not exercised or promoted here. Production
AISurvivor source/JAR is unchanged.

Actual local runtime: Homebrew OpenJDK17.0.21.0.3+0, Gradle8.14.3,
Python3.15.1, Darwin27.2.0 arm64. Full JDK/Python/15-classpath-JAR/source/class
hashes, commands, environment and official pinned engine source/license blobs
are in the frozen manifest. Java17 source/JAR parity and independent direct
compilation of all9 diagnostic classes passed before execution. Standard Java17
CI and component diagnostics passed on ubuntu-26.04; the CI diagnostic task
compiled only. No engine, server, bot or Java25/Retcon launcher ran in this
acquisition. Resource log:14.72seconds real,17.78user,0.54system; maximum
resident set size403488768bytes as reported by /usr/bin/time−l.

The [independent exact-head review](https://github.com/stephanelee/robocode/pull/94#issuecomment-6622430228)
and [external activation binding](https://github.com/stephanelee/robocode/issues/93#issuecomment-6622439204)
preceded the one execution. Preparation compile/metadata failures and unused
pre-review freezes remain under preparation/; they executed0 cases and were
not replacement matrix attempts.

All11 original acquisition files are preserved byte-for-byte in
[raw/](../../benchmarks/meridian-wave-support/raw/). Large gzip files are split
into at-most2MiB binary parts; the manifest records the original and part
size/SHA256 values. [reassemble.py](../../benchmarks/meridian-wave-support/reassemble.py)
restores them without executing a diagnostic. The complete reassembly was
compared against the original acquisition bytes. Wrapper/timing logs and
packaging verification are in acquisition/. The standalone
[independent audit](../../benchmarks/meridian-wave-support/analysis/README.md)
reconstructed all counts, masks, ray histories/geometry, scores, source bits,
action choices, wall guards and conformance coverage from the stored rows.
It starts no cases and imports no candidate evaluator. No raw rows or failures
were removed; no second acquisition occurred.

Independent [outcome verification](https://github.com/stephanelee/robocode/pull/94#issuecomment-6622703802) confirmed every registered guard and all preserved bytes before publication completion.

## Next decision

Reject the mask-only controller for live advancement. Keep the positive geometry
result and all negative action-level evidence. A distinct future hypothesis may
score each modeled ray once over its exact finite swept segments, instead of
summing repeated temporal contributions. That is a new experiment, not a width
retune or repeat of this candidate. Earlier Breakwater per-wave peak danger and
AIS/Challenger crossing-based methods already exist and include rejections;
maximum danger itself is not a novel or proven-winning strategy. Any follow-up
must state its differences, preserve the fixed gun/baseline or disclose changes,
freeze a coherent prototype and obtain independent review before fresh bounded
games. This fixture opens no implicit live budget or confirmation.
