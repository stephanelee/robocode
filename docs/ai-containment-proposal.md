# Next AI pilot: select a footprint, not a mean coordinate

Prospective proposal, not yet fitted or approved for live deployment. The first45-round pilot and its original withheld/transfer rejection remain final. This proposal uses only its six training and three development games for development; its already inspected withheld/transfer games cannot become fresh confirmation. The existing model/data/results are preserved.

## Read-only diagnosis

Diagnostic scope was declared to the independent reviewer before execution: unchanged horizons8/16/24/32/40, original frozen ridge, four original baselines, full exact-observed-label masks, per-game/horizon errors and containment, hindsight union of baseline footprints, and three signed transverse-displacement bands. Only train/dev games were passed to the builder. Zero-speed anchors are counted separately; transverse bands use moving anchors only. Training measurements are in-sample, development measurements are exploratory; neither establishes generalization.

Equal-game averages of containment at primary horizons24/32/40:

| Dataset | Current projection | Coordinate ridge | Head-on | Hindsight baseline union |
| --- | ---: | ---: | ---: | ---: |
| Train | 4.5193% | 1.9322% | 5.8918% | 11.2372% |
| Development | 6.5512% | 1.9951% | 5.4912% | 13.3904% |

On development games, current/ridge mean radial error at horizon24 is109.23/100.80pixels while containment falls11.14% to2.89%. Thus the coordinate-error objective conflicts with this aiming proxy even before fresh evaluation. At horizon40, moving-anchor transverse displacement relative to cached target heading falls left of-18pixels in38.17%, within[-18,18] in30.94%, and right of18 in30.88%, averaged equally by game. Both sides occur; this does not prove a conditionally bimodal density or establish that wave surfing caused each turn. Retcon's source supplies a plausible mechanism: its [two-direction path comparison](https://github.com/robocode-dev/rumble-bots/blob/28981c7a469ac064194b41001cd7183ec5ecf189/bots/java/Retcon/src/fnl/bots/retcon/movement/WaveSurfMovement.java#L53) changes motion as waves approach. A single squared-error forecast may average outcomes between frequently visited positions.

The union is descriptive oracle coverage: it uses future labels to say whether *any* candidate contains the target. It is not a deployable policy, a measured learned gain, or a future success threshold. Raw per-game/horizon diagnostics are [retained here](../benchmarks/ai-containment-diagnosis/train-dev.json).

## Proposed minimal model and budget

Keep the exact24 past-only features, eight-frame history, eligibility and fixed horizons from [the original model lock](ai-forecast-model-lock.md). Fit no new representation or hyperparameters. For each horizon, fit four ridge outputs to binary labels `distance(candidatePoint, futureObservedPoint)<=18`, one output per head-on/linear/circular/current candidate defined in the original lock. Use training games only, population z-score normalization, alpha10 and unpenalized intercept; ordinary unweighted squared loss on binary labels. These are containment ranking scores, not calibrated probabilities. Use the highest raw score without probability clipping; exact ties prefer current, linear, circular, then head-on. Emit the selected original candidate coordinates, never their average. Stale/incomplete eligibility falls back to current. No changes to AIS, movement, firing, DQN or acquisition policy.

Fit once on the original six train games. Freeze all source/dependency/model/preprocessing hashes before evaluating the three original development games once. No alpha search, feature retuning, candidate expansion, threshold search or second fit after those results. Report full-mask and eligible-only per-game/horizon containment, errors, selection counts and missing-label coverage; class imbalance and score calibration remain limitations. Input IDs, future information, private Retcon diagnostics and old withheld/transfer outcomes remain excluded from training and selection.

Development screening compares with *current projection*, not the baseline previously selected by coordinate error. Proposed screen: equal-game/equal-primary-horizon containment improves by at least2percentage points; at least2/3 development games strictly improve; no development game's primary containment loses more than2points; each game's short-horizon8/16 containment loses no more than2points. Require>=50labels/game at each primary horizon and some labels at each short horizon. Report radial error as a diagnostic, without silently reinstating the rejected mean-error objective. Insufficient coverage is inconclusive; failed screening rejects before acquiring any additional games.

If independently approved and the development screen passes, freeze the model unchanged and preregister18 fresh confirmation rounds: three new Retcon games of three rounds and one new three-round game each against the pinned Target, SpinBot and Walls. Fresh fixed-minimum-risk processes use the same accepted legal recorder and no trained DQN checkpoint; register exact IDs, seeds, launch order and source/JAR/runtime pins before acquisition. The original pilot's withheld/transfer games are never rerun or relabeled as confirmation.

Proposed fresh Retcon confirmation gates match the development containment screen, including label coverage,2point gain,2/3 improving games and worst/short loss bounds. Each fresh transfer game must lose no more than2containment points at primary horizons and have no more than5pixels additional primary mean error versus current. Zero AIS skipped turns, complete valid records/rounds, no model writes and all raw outcomes remain acquisition requirements; invalid attempts are retained and stop for review without automatic replacements. No whole-match score claim follows from an offline pass: an independently reviewed, bounded live gun-only A/B with actual intercept timing and legal inputs is required afterwards.

This is a proposed next experiment. Independent review must lock the exact rules before fitting; fresh acquisition additionally needs the frozen successful development result and confirmation manifest. It offers a small test of a concrete objective mismatch, not a promise to beat Retcon.

The independent reviewer cleared this fixed design before implementation. The separate tool preserves the original fitter and has two stages; publish the exact model SHA between them. Both stages filter whole games to their permitted split before window construction. The screen never evaluates the spent withheld/transfer games.

```sh
python tools/containment_forecast.py --manifest MANIFEST --events EVENTS --output NEW_FIT_DIR
# Publish/freeze NEW_FIT_DIR/model.json SHA256 before the single screen:
python tools/containment_forecast.py --stage screen --manifest MANIFEST --events EVENTS --model NEW_FIT_DIR/model.json --model-sha256 FROZEN_SHA256 --output NEW_SCREEN_DIR
```
