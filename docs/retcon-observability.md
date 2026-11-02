# Retcon native observability

This adds offline diagnostics from Retcon 0.2.0's existing `StatsWriter`. Neither bot's gameplay source or committed JAR is changed. Retcon stays pinned to official commit `28981c7a469ac064194b41001cd7183ec5ecf189`, source-tree SHA256 `5f7dfaaedefc39bd93d86440a2996342583d57b75b861942dc1e1899b7164bb2`.

## Capture contract

The harness reads the selected JVM's actual `java.io.tmpdir`, creates a unique direct child for each game, and sets `RETCON_STATS_DIR` only for that invocation. Retcon's unmodified writer rejects directories outside Java temp. The harness retains ownership with a unique marker, rejects symlinks/unexpected layouts, copies bounded JSON files to the game's `retcon-stats/` artifacts, then removes only its owned files and directory. It never recursively deletes an arbitrary configured path. Existing games and unrelated temp files are untouched.

The native filename includes the Retcon bot ID and a monotonic timestamp. The wrapper matches that ID to official Runner results, checks version 0.2.0 and the source hash, validates counters and coverage, and preserves the original native file plus a `capture.json` manifest containing its SHA256. Official battle results now include ID and version for this matching.

`--require-retcon-stats` fails on missing, partial, or invalid exports. Retcon CI always enables it; sample CI does not. Optional capture still records explicit status and never substitutes missing observations with zero. The wrapper schema is version 1; upstream has no schema-version field, so its exact source pin defines the supported native layout.

## Cumulative snapshot semantics

Retcon creates one stats filename per game and atomically overwrites it after each round and at game end. All exported probe/gun/wave counters accumulate across that game. We retain exactly one final snapshot per game and aggregate only across distinct games. We do not sum successive round snapshots. `roundsRecorded` must equal the requested budget for complete coverage; earlier snapshots remain labelled partial.

The Java harness waits at most three seconds after official game completion for the final atomic export before closing bot processes. This avoids stopping the event handler during its last write. Python validates the final JSON after process shutdown. A failed game still retains available raw diagnostics; it does not become a complete measurement.

## Reading the fields

| Native field/group | Meaning and limits |
| --- | --- |
| `gun.bulletsFired`, `gun.bulletHits` | Actual Retcon bullets/hits observed by its timing probe. The wrapper computes hits/shots when shots are nonzero; zero shots yields an undefined/null rate. |
| `gun.virtualGuns` | Parsed into separate per-model hits/completed virtual shots. Virtual shots are counted when a wave reaches a live, freshly observed target; dropped/in-flight waves are excluded. These denominators differ from actual fired bullets. |
| `waves.firesDetected`, `unanchoredShots` | Retcon's own inferred enemy-fire observations. They are not a complete ground-truth count of AISurvivor actions. |
| `waves.hitsTaken`, `hitsOnWave`, power/residual counters | Retcon's association of bullets that hit it with inferred waves, and its own geometry residual diagnostics. These do not directly measure AISurvivor's wave implementation. |
| `movement.*` | Retcon's prediction of its own next motion versus server state. Collision/disabled/skipped transitions can be excluded. `moveWallHitsChecked` is not the same as an unconditional wall-collision event count. |
| `timing.*` | Retcon's shot-creation/origin/heading checks against server observations, plus bounded mismatch examples. |
| `skippedTurns`, `skippedPerRound` | Retcon's skips and their round layout. The array includes an index-zero slot. Missing fields fail validation. |
| compute timing fields | Retcon's own measured compute cost; percentiles are rounded to 10-microsecond histogram bins. Quantiles are retained per game, never summed across games. |

Instrumentation writes occur after rounds; they can add overhead. Both arms of future comparisons must enable the same capture settings. AISurvivor never reads these exports or opponent private state at runtime. They are offline evidence for hypotheses, not an in-game advantage.

## Diagnostic cohort

To repeat the approved baseline after building/verifying AISurvivor with Java 17:

```sh
python tools/retcon_diagnostics.py --java /path/to/jdk25/bin/java --expected-ubuntu 26.04 --expected-bot-sha256 0d120a0461a7bf32b1c6d6277da34f9480309f255caa44e168e6bcdf8636bff2
```

The driver requires an empty output directory, an explicit baseline JAR hash, complete native stats, and the stated actual Ubuntu release. Its defaults are three five-round fixed-policy games. Use a new `--output` path for each repeat.

Two Ubuntu 26.04.1 cohorts were recorded, each containing three independent five-round fixed-minimum-risk games. All 30 rounds used the same unchanged AISurvivor JAR SHA256 `0d120a0461a7bf32b1c6d6277da34f9480309f255caa44e168e6bcdf8636bff2`, Retcon source pin, and Temurin 25.0.5+8 match runtime. Java 17 source/JAR parity passed before switching runtimes. The Linux kernel was `7.0.0-1017-azure`, architecture `x86_64`; the diagnostic driver checked actual os-release ID `ubuntu`, VERSION_ID `26.04`. No runner-label fallback was used.

Cohort 1 was recorded before merging the all-workflow runner pin; cohort 2 repeated the same observability code after that configuration integration had already queued a run. Both are retained separately, with no selection based on outcomes. A preceding [Ubuntu 24.04.5 validation run](https://github.com/stephanelee/robocode/actions/runs/41502621709) is preliminary and excluded from these cohorts.

- Cohort 1: [Actions run](https://github.com/stephanelee/robocode/actions/runs/41503066130), [original artifact](https://github.com/stephanelee/robocode/actions/runs/41503066130/artifacts/13170530565), [canonical JSON](../benchmarks/retcon-observability-2026-11-02/cohort-1/results.json). Evaluated head `43cb592693ba6c8adb0fd733f2c14bcdeae6deaa`; source/merge commit `b85556a1d45bb0141f38f5d82a671f5ae125cf53`.
- Cohort 2: [Actions run](https://github.com/stephanelee/robocode/actions/runs/41503742425), [original artifact](https://github.com/stephanelee/robocode/actions/runs/41503742425/artifacts/13170097367), [canonical JSON](../benchmarks/retcon-observability-2026-11-02/cohort-2/results.json). Evaluated head `46a709b8776d2aac673734683665c3930f1e425a`; source/merge commit `050b067ba3efdb3e7435e8912bee3774cc61f4ac`.

The data directory retains each original native JSON byte-for-byte, its SHA256 manifest, AISurvivor JSONL telemetry, and official battle results. Downloaded native hashes were verified against the recorded capture manifests. Block labels/seeds identify separate unpaired games; fixed-policy behavior does not become RNG-paired because a seed is present. No candidate-promotion or policy-superiority claim follows from these one-code cohorts.

| Cohort | Rounds | Retcon actual hits/shots | Linear virtual hits/completed shots | Head-on virtual hits/completed shots | Matched waves/observed hits taken | Retcon own-motion matches/checks | Retcon skips | Over soft compute budget |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 15 | 115/611 (18.8%) | 96/591 | 78/591 | 54/54 | 9484/9484 | 0 | 0 |
| 2 | 15 | 106/682 (15.5%) | 100/666 | 98/666 | 52/52 | 10592/10592 | 0 | 2 |

Retcon inferred 515 enemy fires in cohort 1 and 564 in cohort 2, with zero unanchored shots in these exports. All 54 and 52 respectively observed hits taken matched an exact-radius wave; neither cohort exported off-by-one-step or other-residual matches. These are Retcon-internal observations within its coverage, not proof that every terminal event was observed or a direct measure of AISurvivor defects.

Cohort 1 had 14 excluded motion transitions and cohort 2 had 19; their prediction checks were not unconditional coverage of every turn. Cohort 2 recorded four predicted wall contacts in checked transitions. Its two soft-budget exceedances occurred without skipped turns; the soft compute budget was 16,666 microseconds against a 100,000-microsecond server timeout. Quantiles below remain per game.

| Cohort/block | Block label/seed | AISurvivor fixed wins | Retcon P50 µs | P99 µs | Max µs |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1/1 | 20261102 | 1/5 | 30 | 140 | 12992 |
| 1/2 | 20261103 | 0/5 | 30 | 140 | 15765 |
| 1/3 | 20261104 | 1/5 | 30 | 120 | 12539 |
| 2/1 | 20261102 | 0/5 | 30 | 140 | 16416 |
| 2/2 | 20261103 | 1/5 | 30 | 140 | 22704 |
| 2/3 | 20261104 | 1/5 | 30 | 160 | 38354 |

The observed virtual linear counts exceed head-on counts in each cohort aggregate, but the gap differs substantially between cohorts and the evaluations are correlated. They support studying the selector; they do not prove a generally superior gun or diagnose an AISurvivor implementation defect. Complete status means final round-snapshot coverage, not ground truth for every combat event; compare native and AISurvivor counters only after aligning their recording lifetimes.

No malformed/missing/partial export occurred in the six accepted games. Focused tests deliberately exercise those failure paths, ownership refusal, Java-temp layout, identity/counter validation, and cumulative aggregation. The exploratory multi-block PR step is removed after recording this evidence; ordinary Retcon CI retains the strict one-round native-stat gate.

See [the strategy study](retcon-strategy.md) and [improvement log](improvements.md) for follow-up hypotheses. Native counters alone cannot establish the causal frequency of an AISurvivor defect; align them with AISurvivor telemetry and controlled comparisons before drawing that conclusion.
