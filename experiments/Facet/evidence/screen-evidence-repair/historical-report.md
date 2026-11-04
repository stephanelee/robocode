# Facet registered duel benchmark results

Five cold20-round games per registered opponent (300rounds total). Rounds within each game share online learning and are dependent. This is descriptive measurement, not confirmation or a Diamond-equivalence claim.

Acquisition status: **invalid**; accepted games **6/15**, officially finished games **7/15**, official completed rounds **140/300**. Invalid games and favorable outcomes remain included. No replacement, retuning or automatic confirmation.

| Opponent | Cold games / rounds | Facet official first places | Equal-game mean score share | Registered disposition |
| --- | ---: | ---: | ---: | --- |
| AISurvivor | 5 / 100 | 93 | 87.88% | invalid-or-incomplete-descriptive-only |
| Retcon | 2 / 40 | 22 | 58.82% | invalid-or-incomplete-descriptive-only |

Acquisition error: `Facet completion must contain every round exactly once`. Incomplete scheduled games are not replacement candidates.

## Individual games

| Opponent / game | Seed | Facet / opponent score | Facet first places | Share | Acquisition status | Missing / unusable metric rounds |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| AISurvivor / 1 | 49667 | 2706 / 308 | 19 | 89.78% | complete (screen invalid) | [] / [] |
| AISurvivor / 2 | 49668 | 2687 / 476 | 18 | 84.95% | complete (screen invalid) | [] / [] |
| AISurvivor / 3 | 49669 | 2619 / 406 | 18 | 86.58% | complete (screen invalid) | [] / [] |
| AISurvivor / 4 | 49670 | 2835 / 252 | 20 | 91.84% | complete (screen invalid) | [] / [] |
| AISurvivor / 5 | 49671 | 3016 / 481 | 18 | 86.25% | complete (screen invalid) | [] / [] |
| Retcon / 1 | 49667 | 1340 / 1272 | 10 | 51.30% | complete (screen invalid) | [] / [] |
| Retcon / 2 | 49668 | 1655 / 840 | 12 | 66.33% | running (screen invalid) | [12] / [] |

Own telemetry errors and last usable cumulative round (partial values are not full-game totals):
- AISurvivor/g1: last usable round=20; errors=[]
- AISurvivor/g2: last usable round=20; errors=[]
- AISurvivor/g3: last usable round=20; errors=[]
- AISurvivor/g4: last usable round=20; errors=[]
- AISurvivor/g5: last usable round=20; errors=[]
- Retcon/g1: last usable round=20; errors=[]
- Retcon/g2: last usable round=20; errors=[]

## Facet telemetry

Physical confirmed counts and hit callbacks belong to Facet. Raw event damage can exceed official capped bullet-damage scoring due overkill. Turn denominator is decisions plus reported skipped-turn events. Incoming opponent private telemetry is not collected.

| Opponent | Confirmed shots / hits | Raw bullet damage given / taken | Wall / bot collisions | Skips / bot turns | Mean range | Mean scan age / max | Decision p99 bucket upper bounds by game |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| AISurvivor | 3074 / 764 | 7076.87 / 1425.56 | 0 / 0 | 0 / 60275 | 421.15 | 0.000 / 0 | [4194304, 4194304, 4194304, 2097152, 4194304] ns |
| Retcon | 1907 / 186 | 1613.30 / 823.40 | 0 / 0 | 0 / 36402 | 391.87 | 0.000 / 0 | [2097152, 2097152] ns |

## Work, energy and fallback evidence

| Opponent | Requested / accepted / confirmed | Fire energy / hit bonus | GC count / milliseconds | Maximum recorded predictor steps | Surf truncations / geometry fallbacks | Unmatched confirmations |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AISurvivor | 3074 / 3074 / 3074 | 5425.62 / 4264.16 | 933 / 1162 | 502 | 0 / 0 | 1 |
| Retcon | 1907 / 1907 / 1907 | 3223.08 / 975.50 | 486 / 531 | 555 | 0 / 0 | 0 |

All raw counters, inference ambiguity, emergency/budget/shadow fallbacks, payload gauges and model hashes remain in results.json and original metrics. A missing required raw field is not a zero passing measurement. Above predictor maxima are observed turn maxima; GC totals are per fresh process.

## Limits and promotion

Latency values are powers-of-two histogram bucket upper bounds, not exact nanosecond quantiles. Raw metrics retain every subsystem histogram, GC/memory/work/fallback counter and whole-game extrema. Observed-interval shot/hit deltas are written to learning-by-round.json; gaps use fromRound/throughRound and have no invented individual round label. Missing raw metrics remain missing evidence, not zero counts. These deltas describe adaptation/co-adaptation and do not establish causal learning improvement. Round extrema cannot be obtained by differencing cumulative extrema.

Five game clusters per opponent limit precision; engine seats/seeds/initial positions were uncontrolled, with only launch order alternated. AIS results have no invented strength threshold. Adaptive/sample transfer, competitive melee and the declared unacquired skip/rejected/multicontact contracts still prevent broad promotion claims. Passing a requested cohort establishes only that cohort screen; failures remain evidence. No automatic fresh confirmation or tuning.

Exact registration, source/JAR/config/opponent/runtime identities, full per-game official results, private Facet metrics/traces/model hashes and failed preflight evidence accompany the report. Independent postmeasurement review and required CI are separate from implementation.
