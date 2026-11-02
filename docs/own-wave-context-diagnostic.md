# Own-shot context coverage: prospective read-only declaration

Scope: the original pilot's six train and three development games only. No fitting, new matches, future labels, target outcomes, model selection, old withheld/transfer scoring or private Retcon state. This is a coverage count, not evidence of predictive improvement. Definitions and synthetic timing/leakage fixtures are committed before a single real count.

Sort records by game/round/turn after schema validation. Deduplicate confirmed own bullet IDs across targets and retain the first record turn in which each event became available. Never backfill availability from the event's creation timestamp; never carry shot memory across rounds. A present eight-frame window is `[anchor-7,anchor]`, so first-delivery context is absent when `firstSeenTurn < anchor-7` (delivery age at least eight). Report event age greater than eight separately: delayed delivery can make that condition differ. Also calculate visibility from actual event copies in those eight records and report disagreements, rather than silently assuming repeated copies cannot occur.

For confirmed shot creation turn k and anchor N, estimated radius is `(20-3*power)*(N-k+1)`, including creation+1 timing. Gap is cached legal target position's distance from confirmed fire origin minus radius; estimated time to arrival is gap/bulletSpeed. No scan extrapolation, future position, hit/removal event or opponent-private scan is used. Cached-scan age is disclosed in bands 0, 1–2 and greater than2. Radial estimates with gap>=-18 count as potential/overlap context; Retcon-like eligible estimates additionally require gap>bulletSpeed. Select earliest eligible by time-to-arrival, then bullet ID for stable ties. These are not certified active bullets or actual Retcon wave matches; already removed/hit bullets and stale scans can yield misleading estimates.

Per game, report known-target anchor and missing-scan counts, scan-age bands, deduplicated shot first-seen delays, any potential/eligible wave, any event-age>8 wave, any missing eight-frame context, and earliest eligible wave missing from eight frames. Missing first-delivery waves are counted by ETA bands: overlap(<0), 0–8, 8–16, 16–40 and >40 turns. Separately count anchors with at least one missing first-delivery wave and estimated ETA in[0,40], and expose its nearest ETA for subsequent *separately reviewed* feature proposals. Counts do not select an ETA threshold or optimize a future target outcome. Anchor/shot correlations prevent interpreting these as independent trials.

The pure `contexts(records,manifest)` generator yields legal context keyed by game/round/turn/target. It filters train/dev before reconstruction. IDs are provenance keys, not learned features. Tiny fixtures verify creation+1, delayed delivery, strict window boundary, repeated-shot deduplication, real-copy visibility, round isolation and invariance of earlier context to appended future records.

After source/declaration freeze, execute once:

```sh
python tools/own_wave_context_diagnostic.py --manifest benchmarks/ai-motion-v1/manifest.json --events benchmarks/ai-motion-v1/events.jsonl.gz --output benchmarks/own-wave-context/coverage.json
```

Any later learned feature or fresh confirmation experiment requires its own prospective model/measurement rules. This diagnostic authorizes neither fitting nor gameplay changes.

## Single declared count: result

Source/declaration `4049c449d6ecccb116816ccdf6341eabfeab3c7f` was published before the count. Four synthetic timing/leakage tests passed. The [per-game output](../benchmarks/own-wave-context/coverage.json) records exact tool/manifest/compressed-input hashes. Only nine train/dev games entered reconstruction; no future labels or held-out/transfer metrics were computed.

| Split | Known-target anchors | Unique confirmed shots | Any missing eight-frame radial context | Earliest eligible wave missing | Missing context with ETA in[0,40] |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 11,533 | 609 | 7,426 | 6,717 | 7,045 |
| Development | 6,069 | 278 | 3,347 | 3,013 | 3,165 |

These are pooled descriptive counts, not independent samples or equal-game performance estimates. All 887 actual first-seen deliveries have creation-turn availability in this corpus, and first-delivery versus actual recorded-window visibility has zero disagreements. The delayed-delivery/repeated-copy fixtures remain important for defining a reusable legal boundary rather than assuming those properties universally. No missing scans occurred; per-game cached-scan age bands remain in the output. Old-wave ETA bands are wave-anchor counts and can exceed anchor counts when several estimates coexist.

There is enough observed missing legal context to justify a separately preregistered feature-family test. Nothing here establishes conditional target response, actual surviving bullets, Retcon's private wave state, or improved containment. No model was fitted or changed during this diagnostic.
