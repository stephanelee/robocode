# Next Retcon experiments and delegated tests

This is the prospective roadmap after the [source-audited external review](retcon-review-response.md). It supplements [the Retcon strategy](retcon-strategy.md); existing registered foundation experiments keep their original gates. These hints are hypotheses, not measured improvements.

## Assignments and current boundary

- Research agent (`battle_research`): deterministic offline dual-model dodging diagnostic, movement/collision fixtures, feasibility artifact, and a proposed isolated movement controller.
- Foundation/learning agent: first extract a focused behavior-preserving targeting seam, then develop a supervised opponent-motion predictor using legal scan history and our actions. Freeze models and whole-game train/dev/withheld splits before evaluation. Keep any later firing-alignment A/B independent of movement, with its own source/JAR and evidence.
- Independent review agent: review source fidelity, legal information boundaries, fixtures, prospective protocols, CI, and eventual outcomes; merge approved PRs. Implementation agents do not merge their own changes.
- Coordinating agent: track the experiments and document accepted/rejected results with the user. No performance matches are authorized by this document alone: publish an immutable prospective mechanism/score protocol and obtain independent preclearance first.

Retcon is the hard competitive target. The already pinned official Java samples remain a fast 1v1 regression/transfer baseline as requested; sample wins do not establish general strength. No new opponent download or Python bot dependency is required. Live opponent-private telemetry is never used.

## AI forecast data requirements

The next AI benchmark must use only information AIS actually receives: timestamped own state/actions, own fire decisions/outcomes, and opponent scans with age/missingness. Retcon reacts to our shots, so retain our shot/action history. Targets are future positions when legally observed at explicit horizons; do not silently interpolate missing scans or treat unobserved turns as truth. Current aggregate round metrics are insufficient for this dataset. Propose bounded nonbehavioral AIS event instrumentation with source/JAR provenance, a schema, and an immutable acquisition protocol before data runs. Exclude observer state, Retcon native/private statistics and future labels from model inputs. Split by whole fresh game before fitting/preprocessing; never split adjacent turns across partitions. Compare frozen supervised forecasts with head-on/Retcon-style linear baselines on horizon-aware held-out error/coverage before a live candidate. Existing pinned official samples offer fast regression/transfer cases; Retcon remains the hard target. Independent review precedes acquisition and live score protocols; no game behavior change is authorized by these data requirements alone.

## Hints translated into tests

**1. Test both head-on and Retcon's iterative linear aim.** A candidate that escapes one requested aim can cross the other. Use current-heading firing corridors around both model requests, rather than exact rays. The current gun heading is unobserved; scan age and shot timing further widen uncertainty. Test whether any legal command clears all modeled corridors throughout the threat's lifetime.

The offline matrix will cover ranges 150–650, every integer speed −8 through 8, headings parallel/perpendicular to the bearing, wall gaps 0/32/200, and reaction delays 0/1/3. Movement must reproduce Tank Royale v1.4.0 acceleration, its actual target-speed-zero/reversal behavior, movement before body turn, speed-limited body turn, and first-wall contact. Collision checks must sweep the bullet segment against the post-move bot circle; endpoint-only checks are insufficient. Use analytic angular corridors, with explicit hypothetical fresh/stale observation sensitivity profiles. A lack of a safe finite command is not proof that all controllers fail.

Baseline: maintain current travel sign at full speed, without turning. Proposed diagnostic selector: compare a finite set of legal target-speed/body-turn commands using worst-case clearance over both models and plausible powers. This is a geometric reference baseline, not the accepted AIS policy or a score comparison. The selector's best-of-set advantage is an in-sample feasibility result, not generalization evidence.

**2. Bring shadow-surfer fidelity forward as a diagnostic.** Retcon's private scans and wave matching prevent a claim of exact reconstruction. Later compare a legal-information estimate with offline truth, separating oracle fixtures from legal-input tests. Measure direction/bin/wave uncertainty before considering a targeting change.

**3. Separate energy/heat inference from movement.** Fire power does not uniquely identify distance because energy and finishing caps also apply. Test event-level timing and heat/wall hypotheses independently. Aggregate native counters cannot establish phantom-wave frequency or a causal AIS defect.

**4. Correct the GF premise.** AIS histograms update on virtual targeting-wave passage, not actual bullet hits, and cold evaluation freezes them. Do not use the external review's hit-reinforcement explanation as a reason to change cold targeting.

## Controlled component benchmark suite

Version the small deterministic suite independently of competitive battles. Development cases cover open arena, wall and corner contact, positive/negative velocity, turn/brake/reverse commands, scan delays, and legal-observation uncertainty. Freeze a separate predetermined withheld set before examining its output; do not tune against it. Report exact grid coverage and baseline/candidate swept incoming-collision exposure to head-on/linear corridors, wall contacts, and trajectory prediction error against explicit golden fixtures or held-out trajectories. Distinguish simulator conformance error from an opponent constant-velocity forecast error. Fixture exposure is not a gameplay win rate. CI should run the bounded component tests; full grid artifacts remain reproducible offline.

Use the already pinned official samples for fast regression/transfer checks and Retcon for hard-target confirmation. After component development, a separately registered fresh Retcon confirmation is required before claiming improvement against Retcon. Neither development fixtures nor the withheld suite substitutes for it.

## Decision before gameplay implementation

The offline report must disclose uncertified states, sensitivity to stale/private scan history and unknown current gun heading, wall contact, bounded command/horizon scope, and conservative approximations. Proceed to a movement candidate only if legal-observation feasibility remains useful under explicitly stated uncertainty; otherwise document rejection or improve the observation model first.

If supported, register one coherent controller hypothesis: when an inferred incoming shot is actionable, choose a reachable movement command that improves the minimum swept clearance over head-on/linear firing corridors, with a wall/contact veto and fallback when uncertainty prevents certification. Recompute each turn from legal observations. Do not read Retcon exports during a game, combine with gun alignment, or treat the offline oracle as deployable information.

Publish an isolated candidate patch and an immutable mechanism-plus-score protocol before performance runs. The protocol must specify baseline/candidate hashes, uncertainty model, intervention eligibility, mechanism observables, primary policy/objective, complete seed/block budget, stopping rule, effect threshold and all guards, and treatment of failures. Independently review it before execution. Use the same accepted baseline in both arms; document every registered outcome. Rejected gameplay/JAR changes are reverted while retaining their patch and evidence. Later score budgets may be prospectively calibrated for variance; the review's conditional simulation does not establish empirical full-gate power.

## Reproducibility requirements

Retcon remains commit `28981c7a469ac064194b41001cd7183ec5ecf189`; engine mechanics are Tank Royale v1.4.0. CI uses exactly `ubuntu-26.04`; AIS builds/parity checks stay on Java 17 and Retcon source launch uses JDK 25. Offline tools require only Python's standard library. Record immutable source/tool hashes, full artifacts, and limitations. No score or strength claim follows from the planned fixtures.
