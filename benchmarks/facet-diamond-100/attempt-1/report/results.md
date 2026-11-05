# Facet registered duel benchmark results

Five cold20-round games per registered opponent (100rounds total). Rounds within each game share online learning and are dependent. This is descriptive measurement, not confirmation or a Diamond-equivalence claim.

Acquisition status: **invalid**; accepted games **0/5**, officially finished games **1/5**, official completed rounds **20/100**. Invalid games and favorable outcomes remain included. No replacement, retuning or automatic confirmation.

| Opponent | Cold games / rounds | Facet official first places | Equal-game mean score share | Registered disposition |
| --- | ---: | ---: | ---: | --- |
| diamond-port | 1 / 20 | 0 | 8.15% | invalid-or-incomplete-descriptive-only |

Acquisition error: `[Errno 2] No such file or directory: '/home/runner/work/robocode/robocode/build/facet-diamond-100/diamond-port-g01/FACET_model.dat'`. Incomplete scheduled games are not replacement candidates.

## Individual games

| Opponent / game | Seed | Facet / opponent score | Facet first places | Share | Acquisition status | Missing / unusable metric rounds |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| diamond-port / 1 | 49667 | 262 / 2952 | 0 | 8.15% | invalid (screen invalid) | [] / [] |

Own telemetry errors and last usable cumulative round (partial values are not full-game totals):
- diamond-port/g1: last usable round=20; errors=[]

## Facet telemetry

Physical confirmed counts and hit callbacks belong to Facet. Raw event damage can exceed official capped bullet-damage scoring due overkill. Turn denominator is decisions plus reported skipped-turn events. Incoming opponent private telemetry is not collected.

| Opponent | Confirmed shots / hits | Raw bullet damage given / taken | Wall / bot collisions | Skips / bot turns | Mean range | Mean scan age / max | Decision p99 bucket upper bounds by game |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| diamond-port | 436 / 31 | 262.37 / 1516.51 | 0 / 0 | 0 / 17699 | 446.53 | 0.051 / 23 | [8388608] ns |

## Work, energy and fallback evidence

| Opponent | Requested / accepted / confirmed | Fire energy / hit bonus | GC count / milliseconds | Maximum recorded predictor steps | Surf truncations / geometry fallbacks | Unmatched confirmations |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| diamond-port | 436 / 436 / 436 | 700.68 / 160.40 | 678 / 813 | 481 | 0 / 0 | 0 |

All raw counters, inference ambiguity, emergency/budget/shadow fallbacks, payload gauges and model hashes remain in results.json and original metrics. A missing required raw field is not a zero passing measurement. Above predictor maxima are observed turn maxima; GC totals are per fresh process.

## Limits and promotion

Latency values are powers-of-two histogram bucket upper bounds, not exact nanosecond quantiles. Raw metrics retain every subsystem histogram, GC/memory/work/fallback counter and whole-game extrema. Observed-interval shot/hit deltas are written to learning-by-round.json; gaps use fromRound/throughRound and have no invented individual round label. Missing raw metrics remain missing evidence, not zero counts. These deltas describe adaptation/co-adaptation and do not establish causal learning improvement. Round extrema cannot be obtained by differencing cumulative extrema.

Five game clusters per opponent limit precision; engine seats/seeds/initial positions were uncontrolled, with only launch order alternated. AIS results have no invented strength threshold. Adaptive/sample transfer, competitive melee and the declared unacquired skip/rejected/multicontact contracts still prevent broad promotion claims. Passing a requested cohort establishes only that cohort screen; failures remain evidence. No automatic fresh confirmation or tuning.

Exact registration, source/JAR/config/opponent/runtime identities, full per-game official results, private Facet metrics/traces/model hashes and failed preflight evidence accompany the report. Independent postmeasurement review and required CI are separate from implementation.
