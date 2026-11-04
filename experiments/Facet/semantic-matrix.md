# Semantic acceptance matrix

Final offline run: 17 groups, 7,119 assertions; Java 17.0.21.0.3, 256 MiB heap. See `evidence/correction-final.txt` and `evidence/build.json`. A passing fixture is not engine equivalence or competitive acceptance.

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
| Lifecycle/export | PASS synthetic round persistence, cold new Core, deterministic bounded schema export | Reused API object lifecycle is source-reviewed, not live acquired |
| Two-wave search | PASS corrected physics adds second-wave cost; union charged once; already-passed second takes zero steps; bounded paths | Original action-change fixture no longer changes action; broader controlled decision acceptance pending |
| Shadow service | PASS explicitly disabled service clears credit/counters; interval death/wrap arithmetic | DISABLED: annular clipping, wrong-tick and broad-phase solver certification absent |
| Melee/radar | PASS helper first-step safety, unsafe fallback, negative-speed displacement decode, survivor typed data retained, stale/unknown radar cases | ACTIVATION GATED pending independent duel review and separate protocol |
| Failure/budget | PASS 240-turn deterministic synthetic stream <=2400 total steps, hard quota throws, round disable reset; analytic gun still fires when learned gun disabled | No configured-runner p99/skip evidence |
| Worst-case fixture | PASS 4,999,680 estimated store bytes, 256 waves, 20 planner iterations; final-run p50 678,042ns/max1,237,083ns | Offline diagnostic timings only |
| Package | PASS Java17 compile, offline smoke, 831-entry manual rebuild payload parity | Gradle/cache build, root source/JAR parity and required CI pending |
| Strength | NOT RUN, battles=0 | Prospective issue registration, independent premeasurement review, authorized screen |
