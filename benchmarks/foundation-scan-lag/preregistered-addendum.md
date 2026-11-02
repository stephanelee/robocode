# Foundation experiment: bullet scan visibility

Registered prospectively before any candidate-versus-incumbent performance matches. This candidate implements only [issue #18](https://github.com/stephanelee/robocode/issues/18), under [tracker #16](https://github.com/stephanelee/robocode/issues/16). It binds and does not amend the approved gates in [Retcon strategy](../retcon-strategy.md) at immutable commit `9b5a5313f13719920b8abf04094dd9438a74b7df`.

## Hypothesis and scope

Tank Royale 1.4.0 scans before applying bullet hits. Moving only bullet damage and hit-bonus corrections from event turn `k` to effective scan turn `k+1` prevents false inferred shots and missed genuine shots when scan energy has not yet changed. No gun-heat filtering, wall/ram shift, uncertain-scan heuristic, movement, gun, radar, reward, model, or learning-algorithm change is included. Better inference is a correctness mechanism; improved Retcon score is uncertain.

The gameplay patch and rebuilt tracked JAR originate at `07b34db9dd9888e45af7967ec5fda2cf61213631`. Subsequent commits add tests, actual trace evidence, protocol tooling, and integration. The measured candidate source will be the exact PR head checked out by the workflow, recorded alongside both complete immutable bot-bundle hashes.

Incumbent source: `a1417774d4f31c43dca0cb83b13c9b216989fbb0`; accepted JAR SHA-256 `0d120a0461a7bf32b1c6d6277da34f9480309f255caa44e168e6bcdf8636bff2`. Gameplay is the same retained version as the three rejected historical cycles. Retcon remains the unchanged official source at `28981c7a469ac064194b41001cd7183ec5ecf189`, using its own API 1.3.1; official Runner/server 1.4.0 and AIS API 1.4.0 remain pinned by the harness. Strict Retcon native-stat capture is required for every scored game; it is offline evidence, never a bot input.

## Preconditions

Focused tests cover both callback orders, damage 2, small hit bonus 1.5, simultaneous real fire, consecutive scans, gaps, and bot isolation. The [actual pinned-engine recording](../../benchmarks/foundation-scan-lag/engine-trace/README.md) independently verifies both effects at `k+1`; raw files, exact source snapshots, source/runtime hashes, and Ubuntu release are retained. Java tests replay these recorded windows. The comparison job additionally records the probe on its match runtime before scoring begins.

Build/test under Java 17 with tracked-JAR source parity; switch to JDK 25 only afterward for Retcon source launches. Every GitHub job uses exact `ubuntu-26.04`, with actual OS/runtime provenance preserved. All servers are isolated headless embedded-runner processes, never the interactive game.

## Bounded scored protocol

Use `--category correctness --purpose primary` in both stages, immutable original plan plus this addendum, seeds `20261102,20261103,20261104`, 20 rounds per version per seed, alternating version launch order. No starts are paired. Complete every declared block unless process or evidence validity fails; there is no performance early stop.

1. **Fixed primary: 120 scored rounds.** `--policy fixed-minrisk`, three independent games per version. This mode uses no bot RNG; seeds label replicates. No training, imported model, or model writes.
2. **Mandatory online veto after primary passes: 120 scored rounds.** `--policy online-train` maps to actual default `train` mode. Every version/seed starts with its own absent model path and fresh RNG initialized to that seed. No shared or imported checkpoints and no excluded preparation games. Both arms learn throughout all 20 scored rounds. Validate checkpointLoaded=false for every round, counters exactly 1..20, final counter 20, and preserve all six independently written endpoint checkpoint bytes and hashes. All online scores are the veto objective, not excluded training rows.

The full successful sequence is 240 scored rounds, comprising 120 fixed and 120 online rounds. A failed primary rejects the candidate and does not trigger the online stage. A failed online guard vetoes retention. Invalid attempts preserve raw logs/evidence and are disclosed before a corrected retry; they are not recorded as measured losses. Reusing seeds does not pair engine positions or opponent history.

Apply unchanged non-regression guards **separately to each stage**, using unrounded fractions: equal-weight mean score-share delta at least zero; at least two of three nonnegative seed deltas; worst seed loss at most 5 percentage points; candidate total wins no more than three fewer; zero candidate skipped turns; candidate wall collisions per 1,000 bot turns at most 1.2 times incumbent, with zero-to-zero required. Score share and adaptive-opponent game blocks are descriptive; this is a small unpaired non-regression experiment, not a powered equivalence or held-out generalization claim.

Retain only after both stages pass, actual engine-contract proof, required checks, and independent review/merge. Label the outcome correctness retained with no strength gain established unless the unchanged tactical +5-percentage-point criterion and its other guards also pass. Existing historical outcomes and tactical gates are unchanged. If rejected, restore incumbent gameplay/JAR, preserve source patch and raw results, and keep the underlying issue open.

## Evidence and current status

Preserve canonical stage JSON, all 12 official game results/raw AIS metrics/native Retcon exports if both stages run, all six online endpoint models, immutable plans, source/JAR/runtime/opponent provenance, and process failures. Full bundle/JAR snapshots remain in the Actions artifact; do not duplicate the rejected candidate JAR in Git. Final report and independent review must distinguish correctness, compatibility, non-regression, and strength.

At registration, the actual-engine trace and deterministic tests have passed; no candidate-versus-incumbent performance comparison has run. Root authorization and independent workflow preclear are still required before the first comparison.
