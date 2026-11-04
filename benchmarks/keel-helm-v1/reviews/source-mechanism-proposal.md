# Keel next-mechanism proposal: stopped-heading rollout consistency

Research only, before gameplay implementation or new measurements. Baseline is merged Keel0.1.0 at `24fb9a5f02065c0bf0b146b8387c2a9d8a272373`; JAR SHA256 `77b8a5c3114641c12a94f36c83083d27c419416ee691d725fd19f0284a69db6f`. Source reviewed in `/private/tmp/robocode-keel-improvement`. No code, imports, numeric helpers, fixtures, games or workflows were executed by this research.

## Recommend one change

**Hypothesis:** preserve body rotation while Keel simulates its first-wave stopped suffix, so second-wave escape options start from the heading the emitted STOP command would actually produce. This may improve Diamond score share while retaining Retcon strength, without changing Keel's successful energy economy.

`Surfer.plan` currently replaces a stopped pose with `new Pose(q.x(),q.y(),q.h(),0)` (around line224), freezing heading. Its emitted immediate STOP command is instead `turn=orbit(me,...,dir)[0]`, speed0 (around line240). `Phys.step` and the pinned engine turn at the new-speed limit after speed0, so a nonzero STOP turn changes heading even though position remains fixed. The second-wave moving options are evaluated from this first-wave endpoint; `orbit` uses endpoint heading for back-as-front and `Phys.step` translates on old heading. Thus this inconsistency can change the second-wave escape geometry and route ranking. It is a static source fact; its frequency or competitive effect has not been measured.

The narrow conceptual seam is the stopped branch inside the first-leg rollout only: use the existing engine-native `Phys.step` with the STOP body's orbit-turn request and targetSpeed0 instead of copying a frozen heading. Keep search population, switch times, two-wave weight0.85, danger densities, risk integration, positions, gun, power, firing eligibility, radar, learning, seeds and warm/cold reset unchanged. The second-wave all-stop branch's heading does not feed another escape and need not be altered in this first treatment. Keep baseline Keel source/JAR unchanged; implement a sibling experimental arm with separate identity, markers and JAR.

## Evidence and distinctness

Keel's six independent confirmation games scored56.53% mean/103of120 firstPlaces, but it lost every damage exchange and had0recorded rams. Do not sacrifice its attrition policy or portray the frozen ram correction as an observed cause. This seam repairs forecast/command consistency while leaving that economy intact.

Inspected prior ledger/platform/Keel tests document exact motion and STOP, but Keel tests do not cover a rotating STOP suffix followed by a moving second-wave escape. Earlier Meridian/Challenger multi-wave trials are prior-art failures; this is not a claim to invent two-wave surfing. It is an isolated new treatment on the independently confirmed Keel baseline. Facet and Prism already implement separate engine-native multi-wave/context-gun systems; do not copy or rerun those whole bots or interfere with their owners.

## Why not the other inspected seams first

The gun uses a near-radius passage proxy with a9px body allowance, whereas engine hits sweep near-to-far. It may produce late labels, but simply changing `+1` to `+2` does not establish the correct learning target. Meridianv3 already used confirmed far-edge crossings; the retained Keel evidence has no private per-wave cohort journal to demonstrate harm or causally attribute its9% hit rate to this proxy. Accepted `setFire` requests are also not confirmed `BulletFired` events, but no accepted-versus-confirmed mismatch was quantified. Both are future source-grounded hypotheses, not demonstrated explanations or a reason to bundle changes.

## Prospective comparison recommendation

Register exact source intervention/docs+issue, candidate/baseline/opponent pins, runtime and budgets before implementation/results. Use Java17 build/parity, ubuntu26.04 and pinned JDK25 for Retcon launch after builds; use the existing Keel Python3.13.17 profile family explicitly rather than STOP24's3.13.18 or a silent runtime substitution.

Suggested screen for root/reviewer to finalize: alternating baseline/candidate cold10-round games, three games per arm against Diamond and two per arm against Retcon; one fast pinned official-sample game per arm. That is120physical rounds if the sample games are10rounds. Do not pool old confirmation games with this fresh comparison. Screen progress could require all games valid/zero own skips, Diamond candidate mean at least3percentage points above contemporaneous Keel and above50%, and Retcon at least90% and no more than3points below contemporaneous Keel, plus the registered sample guard. Three Diamond game blocks give only a screen; fresh independently reviewed confirmation is mandatory before accepting an improvement. Root must choose/freeze confirmation budget and criteria prospectively, not after the screen.

Record raw official score/wins, shots/hits, uncapped damage, energy expenditure, contacts, round variation and invalid rounds. Add bounded per-round model counters (both arms, behavior-neutral) for stopped suffixes considered and chosen, suffix endpoint-heading discrepancy, whether a moving second-wave continuation follows a STOP suffix, and maximum decision time. These diagnose activation; no count demonstrates strength.

Before live clearance, meaningful contract fixtures must cover both orbit signs, forward/backward start speeds, nonzero STOP turn, several suffix lengths, subsequent moving escape, and wall/corner starts. Verify stopped position/velocity and heading against the existing engine contract and ensure unchanged non-STOP rollouts. Do not present fixtures as competitive evidence.

## Main risks

Naively calling `orbit` for every stopped suffix step adds expensive trig/wall-smoothing across thousands of projected poses. Preserve the100ms budget and zero-skip guard. Since position is fixed during STOP, world desired orbital heading can be cached per suffix, while the back-as-front turn is recomputed from evolving body heading; any such optimization needs exact equivalence fixtures. No horizon/search-size reduction should be bundled to pay for it.

Replanning may choose a different STOP orbit sign next tick, so this remains a fixed-policy forecast, not a guarantee of the actual future. A more faithful heading can worsen score if the prior mismatch happened to be beneficial. Reject once if the registered screen fails; retain evidence and do not tune/retry the same candidate.

Source SHA256: Surfer `302a4735b78701b8e0f527f35fb4ea3629f129b1bc7c2d26154776cdba2843df`; Phys `bbd689faf2ebec5f514b94803d93870515873ad727f358d8cf07a4d43ba057c0`; Keel `7fc59025ab263e18773b8e7d1c9d4fc33f3ff72e10e00b3d9d405033ff67f9ff`.
