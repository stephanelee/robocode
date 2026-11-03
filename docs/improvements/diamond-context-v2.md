# DiamondChallenger0.2.0: fixed online context gun

Prospectively registered before any games under [issue53](https://github.com/stephanelee/robocode/issues/53). The user requested a separate bot against Diamond Port after analysis, then later transfer to AIS. Production AIS remains accepted9b1624/JARb41b: no source, root Gradle, descriptor or JAR changes. PR54 and55 rejections remain unchanged and are not pooled.

## Frozen design and artifacts

Experimental source `727132a4a3bd11af4a23054d229ff6969a4eae9a`, distinct DiamondChallenger0.2.0 JAR SHA256 `be9fd6bc16183402e485a8b7dc2a50f3a0179da228e5b5f1651abc7385fa15f7`. Production AIS remains accepted9b1624/JARb41b, and the rejected0.1.0 source/result stays separate. Movement (`PressureMove`), power function and radar remain fixed from0.1.0; only gun capacity and its minimal activation telemetry change. This is not another pressure-constant trial.

The first prototype had0/20 wins/2.59294% share and11/274 actual hits. Its three kinematic candidates cannot represent a context-dependent nonlinear response. That is a capacity limitation, not proof it caused all losses. This variant adds ONE online context-conditioned GF density predictor from legal confirmed-shot passage observations. It has no offline fitting, pretrained challenger data, model search or private opponent inputs.

Fixed family: FIFO memory512 passage samples;7 features are flight-time/60, absolute lateral velocity/8, advancing velocity/8, lateral-mirrored turn rate/10, change in absolute speed/8, wall margin/120, and age of last confirmed own shot/40. Wall margin is min(x−18,width−18−x,y−18,height−18−y). Require current scan turn==decision turn and previous scan==turn−1 for learned queries; the old kinematic fallback retains its age≤1 behavior. Clamp normalized nonnegative features to[0,1] and signed features to[−1,1], with no prior own shot represented at maximum age. Lateral sign is−1 only for a strictly negative lateral velocity; both signed zeros map to+1. Decisions freeze the reference target point/features; actual firing confirmation supplies the origin and authorizes a power/turn/target-matching cohort. Labels re-anchor that same frozen target point around the actual confirmed origin, rather than mixing coordinate frames. The label is the first consecutively observed target center reached by radius v*(scanTurn−creationTurn+1); geometric offset from the frozen reference is divided by escape angle and mirrored by that frozen sign, then clamped[−1,1]. It is an observed-center passage estimator, not actual collision truth. Scan gaps, round boundaries and target changes discard unresolved cohorts; past memory persists only for the same opponent across rounds. No same-shot future observation enters its query.

Require at least32 samples and16th-neighbor weighted squared distance≤1. Distance is sum of weights[2,2,1,1,1,1,1] times squared feature differences, without a square root; deterministic insertion order breaks neighbor ties. Use16 nearest samples with weights1/(0.05+d), normalized to total mass16, plus two unit-mass pseudo-samples at current linear GF. Evaluate47 equally spaced GF grid points over[−1,1] with triangular KDE bandwidth=max(2/46,bot angular half-width/escape angle); ties prefer nearest linear GF then lower GF. This support/distance rule is a fixed heuristic, not calibrated confidence. Learned aiming is primary only when qualified; otherwise retain the original three-model selector. Use the learned angle consistently in the current-ray gate at the existing linear forecast range, an estimated intercept range, not a guarantee of future collision.

Source/tests freeze feature/sign/lifecycle details before games. There is no support/cap/kernel/phase/power search or outcome-selected alternative. Telemetry records learned versus fallback aim decisions, accepted passages and endpoint memory size, with actual learned shots when available. These diagnose activation without adding a performance threshold or per-turn logging framework. The old movement/radar/energy/shot-gating policy outside the selected forecast remains unchanged; confirmed shot energy and range diagnostics remain reported.

## Exact budget, identities and cold state

- **Pilot:** one fresh20-round 1v1 game, ID `diamond-context-v2-pilot`, seed20263201.
- **Only if the pilot passes:** three fresh20-round games, IDs `diamond-context-v2-confirm-g01` through`g03`, seeds20263211–20263213. Maximum80 physical rounds, four independent challenger processes and four distinct initially absent model/metrics paths. No pretraining or model reuse; online learning within each20-round game is part of the default prototype.
- Opponent is unchanged native Python **Diamond Port1.8.28** at repository `fduch-stranger/robocode-bot` commit `67b68c22e41c18281a3fce9ef638e3c23a0b47cf`. Preserve selected source/shared core, Apache/Zlib notices and exact [110-file/12-wheel lock](../../tools/opponents/strong-melee.lock.json). Private pinned CPython3.13 environment/BotAPI1.3.1; no `.env`/gun overrides or opponent logic edits.
- Official Tank Royale1.4.0 runner,800×600 1v1 arena,100000µs turn timeout,900-second per-game ceiling; embedded ephemeral ports only. Interactive7654 is untouched. Exactly Ubuntu26.04, Java17 clean standalone build/tests/parity, then Java25 headless matches. AIS JAR must remain `b41b5de479b1288f087e0cbc9efa64abb68f12381b41d33ad65ff48be5914800` before and after building.
- The challenger uses `CHALLENGER_SEED`, `CHALLENGER_MODEL_PATH`, `CHALLENGER_METRICS_PATH`. Model snapshots are its own schema1 JSON with name/version/seed/roundsCompleted and attempt/hit counts for the three original virtual guns plus bounded context memory and activation counters; **not AIS checkpoints or exact process-resumption models**. Cold startup refuses existing model/metrics files. Every round's `checkpointLoaded` is false and counters run1..20. No loader implementation or persistent state is shared with AIS.

The chosen integers were checked against earlier committed plans; old Diamond/Adaptive/Retcon observations are spent evidence. AIS/challenger RNG seeds do not determine engine starting positions or all opponent behavior. Opponent learning starts afresh per game but can persist across its rounds. Fresh confirmation excludes pilot rows and all prior games.

## Binding decisions

Require matching two official identities/versions, complete20-round official arrays and unique sequential metrics, finite valid counters and a private finite own-format counter20 endpoint. Preserve every round, including skipped turns. A nonzero challenger skip is a failed guard, not an excluded round or permission to replace its game. Missing/wrong evidence invalidates collection and stops; preserve partial output.

**Pilot advances only if all hold:** official challenger score share strictly above20%, at least1/20 official first places, zero challenger skipped turns, and all evidence valid. This is a **progress screen**, not a claim of beating Diamond.

**Fresh confirmation succeeds only if all hold:** all three new games valid, equal-game mean official challenger share strictly above50%, total official first places strictly above30/60, at least two of three game shares strictly above50%, and zero challenger skipped turns. No Retcon, Adaptive, sample or melee constraints apply to this isolated specialist. These become relevant only for a separate later AIS transfer protocol.

All three confirmation games run after a valid passing pilot, regardless of intermediate score; no early favorable subset. Record each share/win count, shooting/damage/energy, collisions, skips and achieved-range diagnostics. Any game-level t interval is descriptive with independence/distribution assumptions, not an added gate or60 independent-round proof. Passing is bounded evidence on this pinned opponent/runtime, not a universal Diamond championship claim.

No threshold waiver, seed selection, replacement games, reruns, support/kernel/phase/power tuning or optional budget extensions. A rejection leaves production AIS unchanged and retains experimental source/patch/evidence; further hypotheses need new prospective registration and independent review. An acquisition continuation also requires separate prospective review, not an automatic Actions rerun.

## Execution and publication

Independently review exact source/JAR, this immutable Git plan, thin direct official-runner driver, native packaging and workflow before the sole execution. No extra root acknowledgment is required after that exact clearance. Always preserve original logs/official results/metrics, every own-model endpoint, source/JAR/runtime/asset/opponent hashes, identities, executed tooling and failures. Update docs/improvements.md with measured budget, decision, report/raw and PR links; a separate reviewer audits and merges after protected CI. No production AIS promotion or transfer is implied.

Local Java17 standalone clean build passed all11 tests (five unchanged strategy cases and six context/gap/support/causality/geometry cases). Power-function and main run movement/radar wiring are text-identical to0.1.0, as is PressureMove. These are source checks, not competitive outcomes.

## Actual pilot: learned predictor active, progress gate rejected

[PR #60](https://github.com/stephanelee/robocode/pull/60) / [sole run41670832690](https://github.com/stephanelee/robocode/actions/runs/41670832690), execution `046eb89921a4b1982e4f66114a95993ae1628f24`, completed all20 pilot rounds with valid official identities, sequential cold metrics and one finite own counter20 endpoint. [Original raw,227-sample memory and source/runtime evidence](../../benchmarks/diamond-context-v2/README.md) preserve the immutable d2739aa registration and canonical decision.

**Reject at the development screen:** Challenger403 versus Diamond3330 official points = **10.79560675% share**, **0/20 first places**. Both unchanged progress checks failed (>20% share and at least one win). Evidence completeness and zero-skips passed. The conditional fresh60-round confirmation was not run; no repeat, threshold change, sample reuse or tuning followed.

| Challenger measurement | Original pilot result |
| --- | ---: |
| Actual hits / shots | 40 /244 (16.393443%) |
| Damage dealt / taken | 403.2 /1771.5 |
| Confirmed shot energy spent | 464.8 |
| Wall contacts / bot contacts / skipped turns | 0 /0 /0 |
| Reported round turns | 4,877 |
| Fresh range observations / mean range | 4,735 /205.707123px |
| Fresh observations below180px | 1,926 (40.675818%) |
| Phase decisions | 159 |
| Learned / fallback aim decisions | 3,320 /1,377 |
| Confirmed learned shots | 185 of244 |
| Accepted center-passage samples / endpoint memory | 227 /227 |

The activation counters establish that the learned branch actually supplied aims and shots in this cohort; it was not an entirely cold fallback-only run. They do not establish a causal strength benefit or calibrated confidence. All227 stored GF proxies can be reconstructed from their recorded confirmed origins, frozen reference points, observed centers, sign and escape angle; sequence/generation metadata distinguish past availability and round boundaries. The three old kinematic virtual models have226 completed cohorts each, hits[39,38,48]; these differ from both227 center-passage labels and40 actual hits by design. Unresolved cohorts are not counted as misses.

The prior0.1.0 pilot had2.59% share/4.01% hit rate, but it used a different fresh seed/game/arena population. The source scope was controlled (movement/power/radar fixed), **not a paired causal A/B**. Do not pool its20 rounds with this pilot or claim that learning caused the numerical difference. Higher observed hit rate still did not produce a single win or pass the progress floor. Incoming damage, range and energy diagnostics do not independently identify the failure's cause.

Production AIS remains accepted9b1624/JARb41b unchanged. This failed0.2.0 experimental source and distinct JAR remain for reproducibility, not as a promoted strategy or AIS transfer. Any next hypothesis needs new source and prospective review/fresh games; no same-candidate rescue is authorized.
