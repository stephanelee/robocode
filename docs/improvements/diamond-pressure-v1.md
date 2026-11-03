# DiamondChallenger0.1.0: first fixed pressure prototype

Prospectively registered before any games under [issue53](https://github.com/stephanelee/robocode/issues/53). The user requested a separate bot against Diamond Port after analysis, then later transfer to AIS. Production AIS remains accepted9b1624/JARb41b: no source, root Gradle, descriptor or JAR changes. PR54 and55 rejections remain unchanged and are not pooled.

## Frozen design and artifacts

Experimental source `3bd75c645b4fe759caccbc4fdadeb62fc58c9544`, distinct JAR SHA256 `1a39bc8eac53b9e3ed8c8b692e4d4de4174bf8516b1798737a038d262964b983`, official identity **DiamondChallenger0.1.0**. Standalone project `experiments/DiamondChallenger` compiles its own entry point, pressure movement and virtual gun, plus four unchanged pure helpers. It does not subclass/include AISurvivor, SurvivorBrain or DQN. The root build and AIS artifact remain unchanged. Build/test with Java17; preserve original and fresh-build JAR parity before execution.

The source-grounded hypothesis is that closing toward140px with interception pursuit and independently seeded20–45-turn flank decisions changes the long-range engagement on which the [pinned Diamond gun/surfing stack](../opponents/diamond-port.md) is designed to operate. Diamond has strong retreat, a pretrained cold gun and2.95 close/rammer counterfire; pressure is unproven and can increase incoming hits or collisions. This is one whole strategy, not an isolated attribution or a parameter search.

The prototype combines physical wall-path safety/instant-stop fallback, independent radar, and head-on/linear/circular virtual targeting. Initial selection favors linear; at least8 confirmed, observed cohorts are required before empirical alternate-gun selection. Virtual scores use only confirmed own shots and consecutive legal fresh scans; missing observations discard a cohort rather than count a miss. Range-dependent power is3 below180px,1.4 below350 and0.6 farther, with low-energy reserve and strict finishing logic. These policies and seed rules are frozen together; no change follows pilot outcomes within this experiment.

All inputs are own state, legal scans and combat events. Opponent source/private logs are offline research only. There is no DQN, pretrained challenger model, hidden opponent state, external fitting or AI-strength guarantee. Metrics report actual confirmed-shot energy, range observations, shooting/damage, collisions and skips to assess the mechanism without inventing an extra acceptance threshold.

## Exact budget, identities and cold state

- **Pilot:** one fresh20-round 1v1 game, ID `diamond-pressure-v1-pilot`, seed20263101.
- **Only if the pilot passes:** three fresh20-round games, IDs `diamond-pressure-v1-confirm-g01` through`g03`, seeds20263111–20263113. Maximum80 physical rounds, four independent challenger processes and four distinct initially absent model/metrics paths. No pretraining or model reuse; online learning within each20-round game is part of the default prototype.
- Opponent is unchanged native Python **Diamond Port1.8.28** at repository `fduch-stranger/robocode-bot` commit `67b68c22e41c18281a3fce9ef638e3c23a0b47cf`. Preserve selected source/shared core, Apache/Zlib notices and exact [110-file/12-wheel lock](../../tools/opponents/strong-melee.lock.json). Private pinned CPython3.13 environment/BotAPI1.3.1; no `.env`/gun overrides or opponent logic edits.
- Official Tank Royale1.4.0 runner,800×600 1v1 arena,100000µs turn timeout,900-second per-game ceiling; embedded ephemeral ports only. Interactive7654 is untouched. Exactly Ubuntu26.04, Java17 clean standalone build/tests/parity, then Java25 headless matches. AIS JAR must remain `b41b5de479b1288f087e0cbc9efa64abb68f12381b41d33ad65ff48be5914800` before and after building.
- The challenger uses `CHALLENGER_SEED`, `CHALLENGER_MODEL_PATH`, `CHALLENGER_METRICS_PATH`. Model snapshots are its own schema1 JSON with name/version/seed/roundsCompleted and three virtual gun attempt/hit arrays; **not AIS checkpoints or exact process-resumption models**. Cold startup refuses existing model/metrics files. Every round's `checkpointLoaded` is false and counters run1..20. No loader implementation or persistent state is shared with AIS.

The chosen integers were checked against earlier committed plans; old Diamond/Adaptive/Retcon observations are spent evidence. AIS/challenger RNG seeds do not determine engine starting positions or all opponent behavior. Opponent learning starts afresh per game but can persist across its rounds. Fresh confirmation excludes pilot rows and all prior games.

## Binding decisions

Require matching two official identities/versions, complete20-round official arrays and unique sequential metrics, finite valid counters and a private finite own-format counter20 endpoint. Preserve every round, including skipped turns. A nonzero challenger skip is a failed guard, not an excluded round or permission to replace its game. Missing/wrong evidence invalidates collection and stops; preserve partial output.

**Pilot advances only if all hold:** official challenger score share strictly above20%, at least1/20 official first places, zero challenger skipped turns, and all evidence valid. This is a **progress screen**, not a claim of beating Diamond.

**Fresh confirmation succeeds only if all hold:** all three new games valid, equal-game mean official challenger share strictly above50%, total official first places strictly above30/60, at least two of three game shares strictly above50%, and zero challenger skipped turns. No Retcon, Adaptive, sample or melee constraints apply to this isolated specialist. These become relevant only for a separate later AIS transfer protocol.

All three confirmation games run after a valid passing pilot, regardless of intermediate score; no early favorable subset. Record each share/win count, shooting/damage/energy, collisions, skips and achieved-range diagnostics. Any game-level t interval is descriptive with independence/distribution assumptions, not an added gate or60 independent-round proof. Passing is bounded evidence on this pinned opponent/runtime, not a universal Diamond championship claim.

No threshold waiver, seed selection, replacement games, reruns, cap/phase/power tuning or optional budget extensions. A rejection leaves production AIS unchanged and retains experimental source/patch/evidence; further hypotheses need new prospective registration and independent review. An acquisition continuation also requires separate prospective review, not an automatic Actions rerun.

## Execution and publication

Independently review exact source/JAR, this immutable Git plan, thin direct official-runner driver, native packaging and workflow before the sole execution. No extra root acknowledgment is required after that exact clearance. Always preserve original logs/official results/metrics, every own-model endpoint, source/JAR/runtime/asset/opponent hashes, identities, executed tooling and failures. Update docs/improvements.md with measured budget, decision, report/raw and PR links; a separate reviewer audits and merges after protected CI. No production AIS promotion or transfer is implied.

## Actual pilot: rejected, no fresh confirmation

[PR #59](https://github.com/stephanelee/robocode/pull/59) / [single run41665190242](https://github.com/stephanelee/robocode/actions/runs/41665190242), execution `909fdbde4f8abbc44dc125357244ac698dd9c0bd`, completed its20-round pilot with valid official results, all20 cold sequential metric rows and one own finite counter20 JSON endpoint. [Preserved raw/source/runtime evidence](../../benchmarks/diamond-pressure-v1/README.md) includes the exact immutable38b069 registration, own JAR/source hashes and unchanged native Diamond pin.

**Reject the pressure strategy at its development screen.** Challenger score83 versus Diamond3118 gives **2.5929397% share**, with **0/20 official first places**. Both required progress checks failed (>20% share and at least one win). The complete-evidence and zero-skip guards passed. The conditional three-game60-round confirmation was not run; no replacement, repeat, tuning, baseline pooling or threshold change followed.

| Challenger measurement | Original pilot result |
| --- | ---: |
| Actual hits / shots | 11 /274 (4.014599%) |
| Damage dealt / taken | 83.2 /1603.3 |
| Confirmed shot energy spent | 452.5 |
| Wall / bot contacts / skipped turns | 0 /0 /0 |
| Reported round turns | 5,064 |
| Fresh-scan range observations | 4,905 |
| Mean observed range | 227.276528px |
| Observations below180px | 1,419 (28.929664% of those observations) |
| Seeded phase decisions | 166 |

The controller sometimes reached close range, but did not consistently hold the140px design target. Range diagnostics are fresh-scan/run-loop observations, not an independently sampled lifetime distribution. They do not establish why it lost. Diamond's close counterfire, the approach policy and aiming errors remain hypotheses, not isolated causal findings. The gun snapshot contains255 completed virtual attempts per model and hit counts[21,15,22]; these geometric counterfactual labels differ from actual11bullet hits and are not calibrated physical hit rates or proof of learning benefit.

This was one unpaired development game on a pinned experimental Python Diamond Port, not original Java Diamond or a tournament sample. It does not establish a comparative regression against AIS's older, separately collected baseline. Production AIS source/JAR/root build remain exactly accepted9b1624/b41b. The failed experimental source and distinct JAR are retained for reproducibility, clearly marked unproven/rejected rather than promoted. Any next challenger strategy needs new prospective source, protocol and fresh games; no automatic rescue is authorized by this outcome.
