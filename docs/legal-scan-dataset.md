# Legal-scan forecast dataset v1

This is a small Python-standard-library schema/label builder for the [next AI forecast experiment](retcon-next-experiments.md). It does not collect data, modify AIS, normalize features, train a model, or run matches. Current round aggregates cannot train a trajectory predictor because they contain no scan history.

The research agent owns `tools/legal_scan_dataset.py` and its synthetic leakage fixtures. The learning agent owns the behavior-preserving targeting seam and a later legal recorder; no AIS source changes are included here. Independent review precedes real acquisition or a live candidate.

## Producer contract

Snapshot at the start of an AIS run turn, after processing legal events and before selecting/submitting current commands. Emit one record for each tracked target, with a shared own-state/action/event snapshot across targets. Flush records without blocking battle decisions; a future acquisition protocol must specify write failure behavior and overhead limits. Every game gets a unique stable game/capture ID from the harness. Every round starts a new history and own-bullet identity scope.

| Field | Meaning and availability |
| --- | --- |
| `schemaVersion` | Integer `1`. Unknown fields/schema versions fail validation. |
| `gameId`, `round`, `turn`, `targetId` | Game ID from preassigned manifest; positive integer round/turn/target ID. Never use IDs as learned explanatory features. |
| `own` | Required `x`, `y`, `direction`, `speed`, `gunDirection`; optional nullable `radarDirection`, `energy`. Legal own state, never observer snapshots. Unknown optional fields become null, not zero. |
| `scan` | Null if no cached legal target scan. Otherwise `observedTurn`, `x`, `y`, `direction`, `speed`, optional nullable `energy`. Must not be from the future; repeated copies of one scan must agree and cached observation time must not regress. |
| `previousIntent` | Null or `submittedTurn < turn` plus nullable requested `targetSpeed`, `turnRate`, `gunTurnRate`, `radarTurnRate`, `firepower`, `action`. No current-turn decision: state-only forecast happens before it exists. Missing intent fields stay null. Requested firepower/action is never proof of firing. |
| `confirmedShots` | List of actual **own** `BulletFiredEvent` data: `turn`, `bulletId`, `power`, `direction`, `x`, `y`. Event turn cannot exceed receiving record turn. Same round/bullet ID must agree across records; never substitute opponent exports or inferred private bullets. |

All numbers are finite; booleans are not numbers. Directions use Tank Royale degrees, speed is −8..8, positions fit the declared arena. Required own/scan state is never inferred from later observations. No gun/radar heading for the opponent is recorded: legal scans do not expose it. `confirmedShots` is a delivered-event list; repeated delivery/cached copies are identifiable by bullet ID, not additional shots. If using shot-count features later, deduplicate within game/round and respect the record in which an event became available.

The builder derives `scanAge` and `scanFresh` from the cached scan's timestamp, rather than trusting a contradictory producer flag. Missing whole records within a history are explicit `recordPresent=false` frames with null own/scan/intent and no events. A present record with an unknown scan is different. Targets with no cached scan do not become forecast anchors.

## Manifest, split and label rules

The manifest must be written before processing captures. Required top-level fields: `schemaVersion`, `datasetId`, `provenance`, `games`. Provenance contains full `sourceCommit`, `botJarSha256`, `engineVersion`, `collectorVersion`. Each game declares `gameId`, unique `acquisitionId`, `split` (`train`, `dev`, `withheld`, or `transfer`), `opponent`, immutable `opponentPin`, integer `arenaWidth`/`arenaHeight`, optionally signed-long `aisSeed`. The seed is an AIS seed, not a claim of identical arena randomness. Capture IDs must identify actual physical games; schema validation cannot prove that renamed captures are independent.

The builder rejects repeated game/capture IDs, undeclared games, injected per-row split fields, duplicate event identities, conflicting cached scans/own snapshots, future events, and unknown/private fields. All rounds, targets, windows and labels from one game keep its manifest split. Assign splits before feature windows and before fitting preprocessing. The tool does no normalization: later means/scales, feature selection and model fitting must use training games only. Frozen model/preprocessing hashes are required for dev/withheld evaluation. Keep adjacent turns and repeated future labels within their game; they are correlated observations, not independent trials.

For an anchor at turn `t` and horizon `h`, a label exists only if a record for the same game/round/target at exactly `t+h` contains a **fresh** scan with `observedTurn=t+h`. Otherwise the output label is `observed=false`, `x=null`, `y=null`. Do not interpolate across missing scans, choose a nearby future scan, borrow another target, or join across death/round boundaries. Future label coordinates never enter the feature window, which ends at the anchor. Output JSON separates `features` from `label` and metadata.

Default history is eight turns, horizons1/3/8. Configuration is bounded to64 history turns, eight unique horizons ≤150, at most200,000 records/256MiB input. Limits fail rather than truncate examples or silently drop registered games. A new output directory is required, preserving previous datasets. Outputs are separate `train.jsonl`, `dev.jsonl`, `withheld.jsonl`, `transfer.jsonl`, plus `dataset.json` with manifest, tool/event hashes, per-split/horizon label coverage and input counts for every planned game, including zero-record games.

Fresh-scan labels are an observable subset. Scan availability depends on radar, our actions and opponent movement; supervised errors on that subset do not establish performance on unseen turns. Report coverage by horizon, opponent and whole game as well as displacement error. Our shot/action history matters because Retcon reacts to us; a passive trajectory model is predictive, not a causal counterfactual model of a new firing policy.

## Bounded next acquisition proposal — not started

Learning's current proposal is12 fresh Retcon games ×3 rounds, with whole games assigned6 train/3 dev/3 withheld, plus three rounds each against the already pinned official Target, SpinBot and Walls as frozen transfer cases. This is a proposal, not an approved acquisition protocol or collected dataset. Before collection, independently review exact game IDs/splits, source/JAR/opponent pins, recorder overhead/failure semantics, label coverage rules, horizons, baseline/error metrics, model budget, stopping rule and publication of every game. Keep Retcon native stats/observer truth entirely outside inputs and labels.

Start with current head-on and linear/circular baselines on the same legal input and missingness masks, then a small supervised residual predictor. Register the baseline definitions and metric before fitting. Group uncertainty and error summaries by fresh game; do not promise statistical power from the small budget. Only a frozen model passing a separately reviewed mechanism/score protocol may become a live targeting candidate. The behavior-preserving seam comes first; movement and firing-alignment changes remain independent.

## Reproduction and synthetic outcome

Run `python -m unittest discover -s tools -p test_legal_scan_dataset.py -v`. Nineteen synthetic tests cover exact labels, no interpolation, past-only features, state-only intent timing, confirmed-versus-requested shots, unknown/private fields, duplicate captures, game/round/target boundaries, missingness, finite values, cached scan consistency, order invariance, independent feature objects, and CLI output preservation. They test builder behavior, not bot strength or real recorder completeness.

The checked-in example below uses synthetic positions and zero placeholder hashes, not collected game data:

```sh
python tools/legal_scan_dataset.py --manifest tools/fixtures/legal-scan/synthetic-manifest.json --events tools/fixtures/legal-scan/synthetic-events.jsonl --output synthetic-dataset
```

It contains six input records, producing15 horizon examples with two observed exact labels (nine train/one observed, six dev/one observed); planned synthetic withheld/transfer games have zero records and remain explicitly listed. No real acquisition, preprocessing, training or score experiment has run. Retain this validated preparation tool; it provides no promotion decision for an AI model. CI runs the tests and example on exactly Ubuntu26.04 and publishes `legal-scan-synthetic` as reproducible schema evidence. See the [experiment index](improvements.md).
