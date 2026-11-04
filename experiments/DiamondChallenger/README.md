# DiamondChallenger0.2.0

**Rejected context-gun prototype:** the [fresh0.2.0 pilot](../../docs/improvements/diamond-context-v2.md) produced0/20 wins/10.79561% share despite185 learned shots and227 accepted passages. The60-round confirmation was not run. Production AIS is unchanged; this remains experimental reference code, not a promoted bot.

**Rejected first prototype:** the [registered20-round pilot](../../docs/improvements/diamond-pressure-v1.md) produced0/20 wins and2.59294% score share. The fresh60-round confirmation was not run. This historical experiment does not replace production AIS or claim competitive strength.

A separate experimental1v1 bot for the [pinned Diamond counter-hypothesis](../../docs/opponents/diamond-port.md). It does not subclass or rename AISurvivor, load an AIS/DQN model, or change production sources/JARs.

Version0.2 changes only targeting capacity: a512-example FIFO conditional guess-factor model uses seven fresh legal context features and confirmed-own-shot wave-center passage proxies. It requires32 available past examples and16 sufficiently close neighbors; a47-point triangular density estimate has neighbor mass16 plus linear prior mass2. Missing/unsupported contexts retain the original three-model selector. Movement, power and radar are unchanged. Model snapshots now include the bounded sample memory and auditable reference/actual-origin/passage geometry; telemetry separates learned/fallback aiming, confirmed learned shots and accepted samples. These are activation diagnostics, not proof of strength or actual-collision labels. The v0.1 rejection remains immutable.

The whole design combines short-range interception pressure (preferred140px), independently seeded20–45-turn flank decisions, pinned1.4 movement/wall simulation, independent radar, three-model online virtual targeting, approach-energy conservation and stronger close-range fire. It attempts to shorten Diamond's long-range surfing window; Diamond's retreat and2.95 close/rammer counterfire can defeat that hypothesis. No winning claim follows from building this prototype.

Build/test with Java17 from the repository root:

```sh
./gradlew -p experiments/DiamondChallenger --no-daemon clean build
```

The standalone Gradle project compiles its own entry/movement/gun and four unchanged pure geometry/radar/forecast/physics helpers. It does not compile/include AISurvivor, SurvivorBrain or DQN. Runnable output is `experiments/DiamondChallenger/build/bot/DiamondChallenger/`, with its own JAR/descriptor/launchers. Root Gradle build inputs remain unchanged.

Set `CHALLENGER_SEED`, `CHALLENGER_MODEL_PATH` and `CHALLENGER_METRICS_PATH` before launch. Each benchmark game requires distinct initially absent model/metrics paths and a new process. The schema1 JSON model snapshot records name/version, seed, completed rounds and three virtual-gun attempt/hit counters. It is endpoint evidence of in-process learning, **not exact process resumption**; this cold-game prototype does not load it.

Schema1 per-round JSONL records name/version, mode=train, seed, round, roundsCompleted, reported turns, checkpointLoaded=false, actual shots/hits, damage dealt/taken, `energySpent` (sum of confirmed own bullet powers), wall/bot collisions, skips and won/died flags. Range diagnostics are `rangeObservedTurns`, `rangeSum`, `closeTurns` (range<180) and `phaseDecisions`. Gun labels are geometric virtual outcomes based only on confirmed own shots and consecutive fresh legal scans, not opponent private state or proof of neural learning. Source/runnable hashes, private endpoints and every failed acquisition must accompany a prospectively reviewed fixed-budget battle.


v0.3 inferred-wave defense pilot:20 rounds rejected,0wins/1.524562% share; no fresh60. [Registered family and outcome](../../docs/improvements/diamond-surf-v3.md). Production AIS unchanged; ContextGun/power/radar code fixed, no paired-causal claim.


v0.4 relevant-wave termination correction: deterministic late-wall scoring defect reproduced, but fresh20-round progress pilot rejected (1win/8.301405%share); no fresh60. [Source proof/protocol/outcome](../../docs/improvements/diamond-wave-horizon-v4.md). No production transfer or paired-causal claim.


Default-OFF bounded diagnostic (CHALLENGER_DIAGNOSTIC_PATH) preserves tested fixed-input policy commands/state. One5-round capture completed with0drops/skips; [schema/findings](../../docs/improvements/challenger-diagnostic-v5.md). Logging overhead remains possible, and passage geometry is not actual hit truth. No strategy or model tuning in this capture.


v0.5 guarded learned-ray selector: shared four-ray completed cohorts with fixedN8/strict2pp margin, unchanged fallback/model/power/radar/defense. Fresh20-round pilot rejected (1win/8.161933%share); no fresh60. [Prospective family and outcome](../../docs/improvements/diamond-guarded-ray-v6.md). No production transfer or causal improvement claim.


v0.6 two estimated own-wave contexts: sole20physical-round attempt INVALID (endpoint531vs logged530cohort completions),0validated summaries/no strength decision/no60. [Fixed semantics and failed evidence](../../docs/improvements/diamond-outgoing-context-v7.md). No retry/waiver/production transfer; actual callback cause remains unknown.


Version0.6.1 is a [closed-round correctness repair](../../docs/improvements/challenger-round-lifecycle.md), validated with deterministic tests only. It stops decisions at RoundEnded, retains the API final-event drain, and seals model/metrics before reset or at GameEnded. Future harnesses must wait for final output before killing the process. No new matches, strategy promotion or rehabilitation of the invalid0.6.0 cohort is implied.
