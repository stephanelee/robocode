# Facet registered benchmark: invalid acquisition

2026-11-04. [PR103](https://github.com/stephanelee/robocode/pull/103) / [issue101](https://github.com/stephanelee/robocode/issues/101). **INVALID; stopped after140/300 official physical rounds. No continuation, replacement, tuning, confirmation, promotion or merge.** The remaining60Retcon+100Diamond Port rounds did not run.

The acquisition completed five20-round AISurvivor games and two20-round Retcon games. Retcon game2 has only19completion rows, missing round12, so the frozen completeness gate stopped acquisition. Its official20rounds and all outcomes are retained, including its favorable score; it is not silently excluded. The trace cap reached round9/turn626, so the exact missing-row cause is unidentified.

AISurvivor game5 also recorded boundary:ConcurrentModificationException near round9terminal observation/model hashing; one unmatched confirmation was reported in AIS game1. Independent source review found Core decision mutations outside the monitor used by terminal observation/hash/snapshot and completion markers before successful output. These are concrete hazards warranting semantic repair; they do not prove the missing-row cause. The source-isolated adapter repair is implemented and independently reviewed offline, without strategy/parameter changes or new matches. It has no battle result; see [repair and verification](facet-adapter-lifecycle-repair.md).

## Official outcomes

Equal-game share is the mean of each whole-game Facet score divided by both scores. Five AIS and two Retcon cold game clusters are used; rounds share online learning and are dependent. No confirmation or broad strength conclusion.

| Opponent | Games / official rounds | Facet official first places | Equal-game score share | Completion status |
| --- | ---: | ---: | ---: | --- |
| AISurvivor | 5 / 100 | 93 / 100 | 87.878524% | 100/100metric rows; one caught boundary exception |
| Retcon | 2 / 40 | 22 / 40 | 58.817175% | 39/40metric rows; missingg2/r12; only2/5games |
| Diamond Port | 0 / 0 | not measured | not measured | 100rounds unrun |

## Every official game

| Game | Facet / opponent score | Facet first places | Share | Metric rows / missing |
| --- | ---: | ---: | ---: | --- |
| AISurvivor-g01 | 2706 / 308 | 19 | 89.781022% | 20 / [] |
| AISurvivor-g02 | 2687 / 476 | 18 | 84.950996% | 20 / [] |
| AISurvivor-g03 | 2619 / 406 | 18 | 86.578512% | 20 / [] |
| AISurvivor-g04 | 2835 / 252 | 20 | 91.836735% | 20 / [] |
| AISurvivor-g05 | 3016 / 481 | 18 | 86.245353% | 20 / [] |
| Retcon-g01 | 1340 / 1272 | 10 | 51.301685% | 20 / [] |
| Retcon-g02 | 1655 / 840 | 12 | 66.332665% | 19 / [12] |

## Reported Facet telemetry

Cumulative final metric rows include all20round counters even when a prior completion row is missing; this is reported telemetry, not independent proof that every callback was captured. Raw bullet-event damage can exceed official capped bullet-damage scoring due overkill. Collisions/skips mean reported event counts.

| Opponent | Confirmed shots / hit callbacks | Raw bullet damage given / taken | Wall / bot collisions | Reported skips / decisions | Mean legal-scan range | Scan age mean / max | Decision p99 interval |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| AISurvivor | 3074 / 764 | 7076.867 / 1425.559 | 0 / 0 | 0 / 60275 | 421.155 | 0.000 / 0 | [2.097152,4.194304)ms |
| Retcon | 1907 / 186 | 1613.299 / 823.398 | 0 / 0 | 0 / 36402 | 391.874 | 0.000 / 0 | [1.048576,2.097152)ms |

Latency comes from log2 buckets, not exact quantiles; range uses the latest legal scan. Both cohorts reported zero collisions and skips, but this does not override the exception/completion failure. Shadow fallback observations include cumulative gauges and must not be naively summed as event counts.

| Opponent | Fire energy / hit bonus | GC count / milliseconds | Max observed predictor steps | Unmatched confirmations / cancelled | Surf truncations / geometry fallbacks |
| --- | ---: | ---: | ---: | ---: | ---: |
| AISurvivor | 5425.623 / 4264.164 | 933 / 1162 | 502 | 1 / 1 | 0 / 0 |
| Retcon | 3223.083 / 975.496 | 486 / 531 | 555 | 0 / 0 | 0 / 0 |

## Frozen identity and evidence

- Measured JAR SHA256:7729fa0419dc8a6efc08ee7d5812ad28aca76195d51ce44bb53056b6c064788d. Source/input:ea6ead0e635a78a9190ca435f476def1eeb33179ca5a4355f87a6f18c362a6d8. Config:9ee52f73dc46e8503fe784d116e247fefc566dcb4e183215385840869d412a25.
- Actual registration SHA256:4fb16fa82901a376d87f39bc9d88a2c4e462542dad14b7d00c2816b91582eb80; acquisition commit8a129ab3f2e437585dd5d894d71ba824a00e9873. All bot/opponent/runtime/game pins are in the preserved registration.
- Ubuntu26.04x86_64; CPython3.13.9 built from pinned official source, GCC15.2/O3/OpenSSL3.5.5; Temurin17build before frozen Temurin25.0.1build8 runtime. Fresh models/processes per game; no cross-game load. Seeds49667–49671; engine seeds/seats/initial positions uncontrolled; only launch order alternated.
- [Original status/results](../benchmarks/facet-registered-300/attempt-1/results.json), [independent full calculations](../benchmarks/facet-registered-300/attempt-1/independent-analysis.json), [review](../benchmarks/facet-registered-300/attempt-1/independent-review.md), [raw manifest/parts](../benchmarks/facet-registered-300/attempt-1/raw-manifest.json).
- [Full Actions artifact](https://github.com/stephanelee/robocode/actions/runs/41858182879/artifacts/13338268667) ZIP SHA2564acbef1a77860739a34f502f181558b413514774794db9b21ea44a7e099a08ee; durable verified1MiBparts are committed beside each game. Reassemble with tools/facet_reassemble_evidence.py; no models are loaded by analysis.
- All three zero-game preflight failures and independent administrative repairs are preserved in experiments/Facet/evidence/preflight-*/; none is counted as played or a rejected gameplay cohort.

## Decision and remaining authority

**Reject promotion; screen invalid/incomplete.** Positive observed shares do not waive the completeness/engineering gates. No automatic remaining160or new300round acquisition is authorized under the registered stop rule. A user exception/new registered decision is required; a semantic fix alone grants no measured budget. No confirmation or parameter search ran. Adaptive/sample transfer, competitive melee and declared actual skipped/rejected/multicontact contracts also remain unestablished. The repaired candidate must be hashed/built/reviewed independently and must not inherit this predecessor measurement claim.

## Final user decision

On2026-11-04 the user explicitly selected **“Stop with partial results”** in response to the proposed300 NEW-round budget (440total including retained140). Zero further games are authorized or scheduled. The disabled fresh-screen proposal remains historical planning evidence; both approval flags remain false. PR103 remains draft, unmerged and not promoted. The current repaired candidate is published at67f43dffd185019780444a659c7afd8590c65d50 with zero battle rounds.

## Java17 CI portability follow-up

The first repaired-candidate CI run41869600041 failed before gameplay because AdapterConcurrencyTest required the macOS TMPDIR environment variable. The single test-only change uses Files.createTempDirectory(prefix), which honors Java’s platform temporary-directory contract. With TMPDIR unset, Java17 Gradle clean build verifyJarParity passes20groups/10243assertions+54adapterassertions and840-entry parity. RunnableJAR remains817a9d53ce944d27e3f536a92869eeca2e0aec20ef5c52b2dab882e931205792; current canonical source/input isdc5cc37c364b7ff76b122116cdc6a3d65a85abd102202a2f75f8add65b68ec5d. Previous12ed1e... remains the exact reviewed earlier input; gameplay and parameters are unchanged. No new battles. The declined proposed registration remains historical and disabled.
