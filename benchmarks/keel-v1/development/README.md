# Keel v1 exploratory development games (not evidence)

These local games were used to **build and select** the Keel v1 candidate before registration. They are exploratory, selected among variants, and therefore biased upward. They must not be pooled with or substituted for the registered games in [`../`](../).

**Runtime.** The games ran in a cloud container: 4 vCPU, Ubuntu 24.04, OpenJDK 25.0.4 (Ubuntu package) for battles and OpenJDK 17.0.20 for builds. They used the pinned Tank Royale 1.4.0 runner and pinned opponents (Diamond Port 1.8.28 lock, Retcon 0.2.0 lock, production AISurvivor 1.2.1 cold TRAIN).

Two games often ran concurrently, so CPU contention may have affected the Python opponent's timing. Diamond's own skipped turns were not observable.

The scripts are in [`scripts/`](scripts/):

- `devgame.py` stages the bots like the registered harness;
- `DevBattle.java` is the observer;
- `analyze.py`, `rounds.py` and `gfstats.py` are reducers.

Per-game observer summaries and the recording SHA-256 values are in [`games.json`](games.json). The raw dev recordings (up to 14 MB each) were not retained in Git.

## Variants vs Diamond Port (cold 10-round games)

| Variant | Games | Mean share | Round wins | Diamond hit rate | Keel hit rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| v0: surfing + KNN guns, 1.95 power | 4 | 25.5% | 4/40 | 11.5% | 7.7% |
| v0 + bullet shadows | 4 | 23.1% | 4/40 | 11.9% | 6.7% |
| v0 with 1.2 power | 4 | 31.2% | 8/40 | 12.5% | 9.1% |
| v0 with 650 px preferred distance | 4 | 23.4% | 3/40 | 12.2% | 7.1% |
| 1.2 power + context flattener 2.0 | 6 | 30.6% | 10/60 | 10.4% | 8.4% |
| **+ energy-conserving fire (frozen v1)** | 6 | **54.9%** | **51/60** | 10.5% | 9.2% |

One earlier 5-round v0 game (36.1%, 1/5) is listed in `games.json` but not in the table.

## Guards (frozen v1 configuration)

| Opponent | Rounds | Share | Wins |
| --- | ---: | ---: | ---: |
| Retcon 0.2.0 | 10 | 93.7% | 10/10 |
| AISurvivor 1.2.1 cold TRAIN | 10 | 90.6% | 10/10 |

v0 also beat TrackFire (0/34 enemy hits) and Walls (1/22), which checks that the surfer dodges a head-on gun.

## What the development games suggested

These are hypotheses for the registered test, not findings.

- **Defense.** The flattener lowered Diamond's hit rate by about 1–2 percentage points in every game. Bullet shadows had no measurable effect. The engine-exact shadow interval is nearly zero when our bullet flies radially toward the wave origin, which is the normal case.
- **Economics.** With firing at 1.2–1.95 power, rounds were lost mainly to our own firing cost: about 100 energy per round at an ~8% hit rate. Diamond then fired 0.1-power finishing shots, because its power is capped by our energy.
- **Eco policy.** Under the eco fire policy, rounds lasted about 4,500 turns. Diamond ended them near zero energy after spending its own energy. Keel kept 3–14 energy but dealt only about 15–20 bullet damage per round. Share is limited by Diamond's damage dealt on us (about 45 per round) even in rounds we win.

## Post-freeze exploration toward v2 (dev only, not evidence)

These variants were tried on branch `claude/keel-v2-dev` after v1 was frozen. Each was one change on top of v1, tested in 4 cold 10-round local games vs Diamond.

| Variant | Mean share | Round wins | Diamond hit rate | Disposition |
| --- | ---: | ---: | ---: | --- |
| + half-speed orbits and mid-path reversal options (`1c31b6e`) | 53.2% | 33/40 | 10.7% | No gain over v1 (54.9%); dropped |
| + finisher: 1.95 power when Diamond < 20 energy and Keel leads by > 25 | 46.1% | 28/40 | 10.7% | Worse; dropped |
