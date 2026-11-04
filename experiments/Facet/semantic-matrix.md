# Current revision2 correction status

Current package passes18 groups/10,034 assertions, including prior17/9916 gates. Export schema2; melee remains gated. [Revision2 result](revision2-corrections.md) contains blocker mapping, evidence/hashes, failed attempts and remaining actual-contract/independent/CI/performance gates. No battles or strength claims. Earlier entries below describe historical revisions and are superseded where this result differs.

# Semantic acceptance matrix

Latest additional-correction offline run: 17 groups, 9,916 assertions; Java 17.0.21.0.3, 256 MiB heap. See `evidence/semantic-tests.txt`, `evidence/lifecycle-fixture.txt` and `evidence/build.json` (older correction-final evidence is historical). A passing fixture is not engine equivalence or competitive acceptance.

| Family | Actual result | Remaining gate |
|---|---|---|
| Native angles/GF | PASS cardinal/wrap/reverse/asymmetric sign round trip | Independent review |
| Engine motion | PASS 433 Alpha JSON fixtures + 433 Beta raw transitions; position/speed <=1e-6, headings <=1e-6 degrees; 433 creation/cooling comparisons | Actual bot-bot contacts and skip acquisition absent |
| Fire clocks | PASS approved raw request55/event56/first-motion56, origin/heading and heat1.02; physical callback count exactly1 | Actual rejected request acquisition absent; rejection bookkeeping is synthetic |
| Energy visibility | PASS four raw isolated damage/bonus windows, synthetic simultaneous correction, exact hot/cool checks, gap ambiguity | Missing-scan hostile multi-shot schedules unresolved, conservatively enveloped |
| Geometry | PASS swept-circle wrap/tangent/source-inside; positive missing-geometry fallback | Numerical sweep is conservative subdivision, not exact collision solver |
| Neighbor math | PASS brute-force weighted distance, changed weights, duplicate ties, FIFO and finite handling | Runner latency acceptance |
| Origin quality | PASS frozen missing mask, half-weight once, mirrored CV and supported normalized prior | Independent model review |
| Cohort chronology | PASS unconfirmed virtual cannot rate; common confirmed denominators; frozen request predictions; duplicate labels suppressed | Live adapter independent review |
| Contact association/visits | PASS same-batch scan-before-association, center retained until clear, exact visit once, multi-victim dedup, impossible Wilson rates rejected | Actual multi-contact and skipped-turn acquisitions absent |
| Lifecycle/export | PASS synthetic final fired/hit/death/scan batch, duplicates, both completion orders, no terminal planning, four file-backed unique game-round rows and cold reused Facet | Actual live API callback dispatch acquisition/review pending |
| Two-wave search | PASS native north-wave controlled action1 to STOP, safe paths and overlap-before-first-center; union charged once; already-passed second takes zero steps | Independent fixture review; old1/1 diagnostic retained as inadequate action-change evidence |
| Shadow service | PASS active annular quadratic clipping, native independent segment/broad-phase sample certificates, wrap/zero/disjoint, wrong tick, death/disappearance, quotas/wall termination | Continuous uncertainty explicitly zero; independent numerical review and live adapter validation pending |
| Melee/radar | PASS helper first-step safety, unsafe fallback, negative-speed displacement decode, survivor typed data retained, stale/unknown radar cases | ACTIVATION GATED pending independent duel review and separate protocol |
| Failure/budget | PASS 240-turn deterministic synthetic stream <=2400 total steps, hard quota throws, round disable reset; analytic gun still fires when learned gun disabled | No configured-runner p99/skip evidence |
| Worst-case fixture | PASS 4,999,680 estimated store bytes, 256 waves, 20 planner iterations; final-run p50 678,042ns/max1,237,083ns | Offline diagnostic timings only |
| Package | PASS Java17 compile, offline smoke, 831-entry manual rebuild payload parity | Gradle/cache build, root source/JAR parity and required CI pending |
| Strength | NOT RUN, battles=0 | Prospective issue registration, independent premeasurement review, authorized screen |

Additional exact semantics: enemy energy0/.1/unknown returns no affordable power; .11 returns legal strict power below energy before any context/wave construction. Fresh cool forecasts use current request time, heat.1/cooling.1 uses next decision time; stale/unknown scans create no forecasts. Fresh observations replace prior forecast geometry. Wall context uses ceil(1.5H) with explicit global-quota missing flags instead of the unexplained150 cap. Independent review findings exist; premeasurementApproved remains false.
