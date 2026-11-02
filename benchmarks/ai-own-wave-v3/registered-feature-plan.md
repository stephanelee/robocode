# One old-own-wave context feature family

Prospective declaration before fitting. The first forecast and containment scorer remain rejected under their unchanged criteria. Only the existing six training and three development games may be used for this model's fit/screen; the old withheld/transfer games remain spent. No new matches, live predictor, movement, firing or DQN change is authorized here.

## Measured rationale

The independently reviewed [single coverage diagnostic](own-wave-context-diagnostic.md) reconstructed only already delivered own BulletFired events. Of11,533 training anchors,6,717 had an earliest eligible estimated outgoing wave missing from the original eight-frame history; development had3,013/6,069 such anchors. This is substantial missing legal context, not evidence of a surviving bullet, Retcon's private wave state, causal response or predictive gain. No labels or model fitting entered that count.

Creation timing is fixed: for shot creation turn k and anchor N, speed v=`20−3*power`, estimated radius=`v*(N−k+1)` and next swept segment ends at`v*(N−k+2)`. Availability begins at the first record containing the confirmed event, not its creation timestamp. Never backfill a delayed event or cross rounds.

## Exact isolated change

Preserve the original24 raw features as an identical prefix, eight-frame base history, label masks, eligibility, five horizons8/16/24/32/40, four existing candidates, train-only population normalization, ridge alpha10, unpenalized intercept, ties, current-projection fallback and all v2 development/fresh-confirmation gates.

Append exactly **two** values: `old_missing_wave_present` and `old_missing_wave_eta_turns`. At an anchor, reconstruct already-delivered, deduplicated own shots within the same round. From the cached legal target position, estimate each radial gap using creation+1 timing. Select the global earliest estimate with **gap>bulletSpeed**, breaking ETA ties by bullet ID. Emit `(1, ETA)` only when that estimate also has firstSeenTurn<anchor−7, creation age≤96 and **1<ETA≤40**. Otherwise emit `(0,0)`. The presence flag disambiguates missing context from an imminent estimate. ETA stays in turns and is capped at40 before the unchanged train-only standardization.

This is deliberately stricter than the diagnostic's broad `oldWithin40` field; that field also includes0≤ETA≤1. We do not silently use it as Retcon eligibility. The96-turn bound applies to the frozen800×600 pilot: maximum range is below1,000 and minimum bullet speed11, so larger ages are already beyond that radial bound. Helpers must disclose/restrict this arena scope. The estimates omit hit/removal state and opponent-private detection/anchoring; no hit truth is fabricated.

The model source must preserve v2 behavior when the new block is absent, and tests must verify the24-feature prefix, creation+1, strict first-seen window boundary, delayed delivery, round isolation, future-append invariance, missing/default semantics and explicit split permissions. Fresh-confirmation enrichment may accept only separately approved fresh splits; diagnostic defaults remain train/dev. Freeze helper/model/dependency hashes before fitting.

## Single test and stopping rules

Fit once on the same six training games, publish the frozen model SHA/artifact, then screen once on the same three development games. Do not inspect old withheld/transfer outcomes, search thresholds/features or refit after this screen. Use the identical v2 screen: ≥2percentage-point equal-game/equal-primary-horizon containment gain over current, ≥2/3 improving games, worst primary and each short-horizon loss≤2points, and original coverage requirements. No threshold is lowered.

Only a valid development pass and independently reviewed exact manifest/workflow may trigger18 genuinely fresh confirmation rounds (three new Retcon games×3, plus Target/SpinBot/Walls×3). Freeze weights unchanged. Fresh Retcon/transfer guards remain those in [the v2 protocol](ai-containment-proposal.md); no live gun A/B follows automatically from an offline pass. Publish every outcome/index/PR/raw link, preserve failures, and require independent review/merge.

Baseline error/containment stratification, if produced, is explanatory train/dev-only analysis and cannot change this frozen feature family or gates. The coverage rationale already justifies this one bounded test; optional stratification must not delay its exact-source review or authorize more scope.
