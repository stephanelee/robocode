# Aim-corridor movement with two-way wall smoothing (v2)

Status when this file is first committed: **preregistered, not yet measured.** The hypothesis, budget, metrics and decision rule are fixed before any registered match. Results are appended under the final heading in a later commit; nothing above that heading is edited after the run starts.

## Baseline and candidate

- **Accepted baseline:** master `116c47ccc77e85f9c8f570c554f5e626a4ed76d4`. Its gameplay is identical to accepted `204ad4c9`/`cea19c8`. Runnable JAR SHA256 `1b99fa0ac303e5e5f89ed91bb3803841ded54910628f335fe1aeaeba65e34c9a`.
- **Candidate:** the commit containing this file, with its source, tests and Java 17 JAR. The workflow records the exact SHA and JAR hash.
- **Opponents and runtime:**
  - Retcon 0.2.0, unchanged, at `28981c7a469ac064194b41001cd7183ec5ecf189`, with API 1.3.1 and JDK 25.
  - The pinned Tank Royale 1.4.0 Runner, plus the official 1.4.0 Target, SpinBot and Walls bots.
  - AISurvivor built with Java 17; the job runs on exactly `ubuntu-26.04`.

## What changed since v1, and why (facts)

[v1](aim-corridor.md) added head-on/linear aim-corridor danger to minimum-risk movement. It was rejected: score share fell by 0.577 pp, wall collisions per 1,000 turns rose from 2.67 to 14.74, and Retcon's hit rate on AISurvivor rose from 17.6% to 21.1%.

Local development diagnosis followed. It ran on Ubuntu 24.04 with OpenJDK 17/25 and is *not* evidence. It used only development seeds 9001, 9101–9104, 9201–9202 and 9301–9306, never the registered seeds below. An instrumented build traced every wall hit. In the traced corridor games, they came from the accepted `wallSmooth`:
- It searches only 36 headings in one rotation direction (half a circle).
- When none is safe, it returns an **unchecked heading pointing into the wall**.

Two recorded examples:
- At (275, 582) with requested heading 330°, it returned 150°. The safe arc there is 231°–309°.
- At (782, 402) with requested 243°, it returned 62°.

The bot then drives into the wall at speed 0 while slowly turning, and hits it again every two turns. The corridor term causes this more often because it selects destinations at angles along or behind walls. The defect is pre-existing: one development baseline game had 80 wall hits.

### Candidate change

The v2 candidate is the unchanged v1 corridor code plus one replacement. `wallSmooth` now calls a pure `CombatMath.nearestSafeHeading`:
- It searches 5° steps in **both** directions over the full circle, preferring the smallest deviation (positive first on ties).
- It returns the first heading whose 80-unit stick endpoint lies strictly inside the 80-unit envelope.
- If no heading qualifies, as in a deep corner, it heads toward the arena centre.

This differs from rejected cycle 2, which searched one direction only and would pick the far-side heading, e.g. 235° instead of 305° in the first example. The candidate bundles this wall fix with the corridor term. Any result is attributed to the pair, not to either part alone.

### Development observations (exploratory; disclosed, not evidence)

Six 5-round Retcon games per variant:

| Variant | Mean share | Walls per 1,000 turns | Retcon actual hit rate |
| --- | ---: | ---: | ---: |
| Baseline | 18.9% | 1.38 | 15.7% |
| v2 (this candidate) | 19.0% | 3.79 | 19.9% |
| Baseline + wall fix only | 20.7% | 0.18 | 16.5% |

An earlier 4-game sweep showed that a "reverse when nose is at a wall" rule did not fix the corridor's walls and was discarded. These numbers have very large game-to-game variance (single-game shares 5.6%–54.9%) and were not used to choose any threshold below. They suggest the candidate may fail; it is registered and measured anyway because it was requested.

## Hypotheses (not facts)

- **H1:** With the wall defect removed, corridor danger lowers Retcon's hit rate on AISurvivor.
- **H2:** The combination raises AISurvivor's Retcon score share by at least 5 pp.
- **H3:** It does not harm the sample bots or online-training play.

Known risks:
- The development data above already contradicts H1.
- The corridor reconstruction approximates Retcon's gun timing.
- False waves add spurious danger.

**Information boundary:** inputs are only AISurvivor's own state history, legally inferred waves and arena geometry. Retcon's private statistics remain offline diagnostics only.

## Budget and procedure (single run, fixed in advance)

Everything runs in one `Aim corridor v2 experiment` workflow job:

```text
tools/aim_corridor_experiment.py --plan-path docs/improvements/aim-corridor-v2.md --seeds 20261211,20261212,20261213 --sample-seed 20261214
```

There is no early stopping, no favorable rerun and no seed replacement. An infrastructure failure marks the run **invalid**; partial evidence is preserved and any retry is disclosed.

- **Stage A, tactical primary (120 Retcon rounds):**
  - command: `compare.py --category tactical --policy fixed-minrisk --purpose primary --rounds 20 --require-retcon-stats`;
  - seeds: 20261211, 20261212 and 20261213;
  - arm order alternates by block.
- **Stage B, sample regression (60 rounds):** Target, SpinBot and Walls, fixed minimum risk, 10 rounds per arm, seed label 20261214. This stage always runs.
- **Stage C, online-training veto (120 Retcon rounds):**
  - runs only if Stage A retains;
  - command: `compare.py --category correctness --policy online-train` with the same seeds;
  - fresh models per arm and seed.

Maximum budget: 300 scored rounds.

## Acceptance criteria (unchanged from v1; all must hold)

**Stage A** uses `compare.py`'s tactical thresholds on unrounded values:
1. mean score-share gain of at least **+5 pp**;
2. at least **2 of 3** blocks strictly improved;
3. no block losing more than **5 pp**;
4. candidate wins at most **3** below baseline;
5. **zero** candidate skipped turns;
6. candidate walls per 1,000 turns at most **1.2×** baseline.

**Stage B:** for each sample bot, the candidate share is at least the baseline share minus **5 pp**, all 10 rounds per arm complete, and the candidate has zero skipped turns.

**Stage C:** `compare.py` correctness thresholds:
- mean share change ≥ 0;
- ≥ 2 of 3 blocks nonnegative;
- no block losing more than 5 pp;
- wins at most 3 lower;
- zero skipped turns;
- wall rate at most 1.2×.

**Decision:** retain only if A, B and C all pass. Otherwise reject:
- restore the baseline gameplay source and JAR exactly;
- keep the patch, protocol and raw evidence.

A retained result is a Retcon-specific tactical gain under this protocol. It is not proof of beating Retcon or of general strength.

## Mechanism observables (descriptive)

The report covers, per arm:
- Retcon actual hits/shots on AISurvivor;
- Retcon virtual linear and head-on hits/completed shots;
- AISurvivor damage taken per 1,000 turns, hit rate, shots, wins and wall collisions.

## Results

Not yet measured.
