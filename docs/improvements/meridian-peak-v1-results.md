# MeridianPeak v1: valid screen rejected

**Decision: reject advancement.** The sole acquisition completed all five games,
25/25 valid physical rounds, zero invalid rounds and zero skipped turns. The
candidate failed both the Diamond share guard and incoming-damage guard. The
conditional180-round confirmation never opened; no replacement or rerun is allowed.
Production AISurvivor and all previous experimental bots remain unchanged.

[PR #96](https://github.com/stephanelee/robocode/pull/96),
[prospective issue #95](https://github.com/stephanelee/robocode/issues/95),
[registered protocol](meridian-peak-v1.md),
[sole run41783652109](https://github.com/stephanelee/robocode/actions/runs/41783652109).

## Hypothesis and frozen comparison

The new isolated MeridianPeak0.1.0 forks actual Meridianv3 source809b880. It takes
one finite-segment exposure peak separately for each ray hypothesis, then sums
rays/waves, using route-only two-pass engine physics. The original v3 gun and
learner policy bodies, gun Flight forecast and numerical orbit/wall/switch costs
are retained. This is a whole-defense prototype; removing dwell-time summation
changes danger's relative scale. Earlier Breakwater/AIS/Challenger peak/crossing
work is prior art, and no isolated MAX benefit is claimed.

Five cold five-round games ran in the registered order: v3/Diamond control,
Peak/Diamond, Peak/Retcon, Peak/AIS TRAIN, Peak/official Walls. Own seed61000
(and AIS61000), ports32900–32904,800x600, cooling0.1,inactivity450,
turn100000us/ready10000000us, defaultTPS-1. Each game used fresh processes,
server, stage, HOME, private outputs and cold models; learning persisted only
inside its five rounds. Seeds do not pair uncontrolled engine/opponent starts.

All five valid games were completed despite scores. No favorable subset or
historical game was pooled. The screen required candidate Diamond share above
its contemporaneous control AND lower uncapped incoming damage per100 enemy
confirmed shots; each candidate matchup also required share strictly>35% and
at least one official first place. All own/enemy shot denominators were nonzero.

## Complete measured screen

Hit/shot counts are confirmed observer events; damage is uncapped event damage,
not the capped/integer official damage score. Firing rate uses all recorded
observer ticks as its denominator.

| Game | Score share | Wins | Own hits/shots | Damage dealt / taken | Incoming damage /100 enemy shots | Own shots /1000 observer turns | Walls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v3 control / Diamond | 12.397820% | 0/5 | 17/205 (8.293%) | 91.467352 / 290.836606 | 98.924016 | 48.452 | 0 |
| Peak / Diamond Port | 25.505051% | 1/5 | 21/197 (10.660%) | 134.466848 / 298.659874 | 128.732704 | 56.593 | 2 |
| Peak / Retcon | 93.128391% | 5/5 | 25/236 (10.593%) | 188.326667 / 38.000000 | 15.702479 | 57.872 | 1 |
| Peak / AISurvivor | 93.631436% | 5/5 | 45/190 (23.684%) | 337.400000 / 47.200000 | 27.764706 | 58.534 | 0 |
| Peak / Walls | 100.000000% | 5/5 | 50/95 (52.632%) | 413.466667 / 0.000000 | 0.000000 | 57.927 | 0 |

All games had zero bot contacts and zero skips. Enemy hits/shots were52/294,
38/232,5/242,7/170 and0/25 in that same order. Enemy hit rates were17.687%,
16.379%,2.066%,4.118% and0%. All25 individual rounds, score deltas, damage,
shot energy and denominators are retained in [rounds.csv](../../benchmarks/meridian-peak-v1/analysis/rounds.csv);
[games.csv](../../benchmarks/meridian-peak-v1/analysis/games.csv) and
[full decision.json](../../benchmarks/meridian-peak-v1/screen-evidence/decision.json)
retain every matchup.

Diamond total score was91/734 for control and202/792 for Peak. The observed
share difference is+13.107230 percentage points; it is not a confirmed treatment
improvement. Peak's25.505051% is below the required>35%, and128.732704 incoming
damage per100 enemy shots exceeds control98.924016. Retcon/AIS/Walls passed
all absolute transfer guards; they do not rescue the Diamond failures.

Peak's incoming hit frequency was slightly lower, while mean uncapped damage per
incoming hit was7.859470 versus control5.593012. Thus the guard failure cannot
be summarized as a higher collision frequency. Shot-power/contact composition,
movement, opponent behavior and unpaired starts remain confounded. This arithmetic
description does not waive the fixed damage gate or establish a causal defense effect.

## Variation and limits

Within-game Diamond round shares ranged0–29.496403% for control and
1.438849–79.699248% for Peak; sample SDs11.362485 and30.281480 percentage points.
Peak's one winning round was round3; every other round is retained. Retcon round
shares ranged86.440678–100%, AIS85–100%, Walls100–100%. Full arrays and descriptive
variation are in [screen-summary.json](../../benchmarks/meridian-peak-v1/analysis/screen-summary.json)
and the independent audit. These are adaptive within-game rounds, not independent
cold-game replications. One game per arm/opponent gives no robust estimate of a
general effect; no significance claim, favorable rerun, power-normalized rescue,
or inference against championship Java Diamond is made.

## Integrity, lifecycle and cost

The standalone [independent auditor](../../benchmarks/meridian-peak-v1/analysis/independent_audit.py)
imports no repository reducers. Its first outcome execution passed all five games,
reconstructed official boundaries/ticks/scores/events and registered rejection,
and reconciled legal private model updates/seals, crossing bank, virtual skill,
cold starts, seed counters and observer-deliverable callback cohorts. All observed
outgoing hits in this sample were deliverable; observer-only outgoing counts and
damage were zero. That observation does not remove the source-defined cohort limit.
Original v3 has markers only, with no comparable private model/decision-time DTO.

The owned runner/process-tree elapsed sum was140.560408s; waited-child user CPU
187.175246s and system CPU11.434633s. These RUSAGE_CHILDREN deltas include the
server, observer and opponents, not only bot policy. Each game's commands, staged
hashes, logs and costs are archived. The highest recorded Peak decision-path wall
time was9.852099ms (Retcon); Diamond's maximum was3.125241ms. System.nanoTime
path timings exclude other callback/startup work and are not process CPU or a
worst-case latency guarantee. No timing fallback altered policy.

## Provenance and once-only execution

- Acquired payload: `c5239a624426b88740445f2a94675c9db995ee20`.
- Freeze SHA256: `07e72b257dea6f9c81217ff1143bb87addb300fa650bf0d7374b216ac24e973c`.
- Peak main source SHA256: `c8b6ab91b85460ecf910abc8c090ac4a8cce739f09a85cf0884cda702c7ffdda`.
- Peak JAR SHA256: `7ee19c304744a807ffd29cc2b282464ae8e8b999c597225729b41c91b79c0d9c`.
- v3 source809b880; control JAR SHA256: `5875565a13b5759940ccb080c0c7f237caec0bae525a68e68643a35890a1a8a9`.
- Permanent screen receipt: `refs/tags/experiment-receipts/meridian-peak-v1-screen`,
  annotated object `5c4e19bb19b32ee47ec186a608a1b376f1102693`. Never reset/delete it.
- Full original artifact ZIP:13,352,108bytes, SHA256
  `9479829d95903329fc12e17ab4164943f4940f5be38940636956d69a52940afa`.

Actual Ubuntu26.04 image20261026.180.1,x86_64; Temurin17.0.21+9 builds/tests and
AIS/existing Meridian/Peak/control source-JAR parity preceded Temurin25.0.5+8
battle/source launch. Native Python3.13.18, installed package manifest52b7bfcf…
matched the frozen environment. The full [runtime profile](../../benchmarks/meridian-peak-v1/screen-evidence/profile.json),
[prepared environment](../../benchmarks/meridian-peak-v1/screen-evidence/prepared-environment.json),
[host](../../benchmarks/meridian-peak-v1/screen-evidence/host.json),
[activation](../../benchmarks/meridian-peak-v1/screen-evidence/activation.json),
[freeze](../../benchmarks/meridian-peak-v1/screen-evidence/freeze.json) and
[remote receipt](../../benchmarks/meridian-peak-v1/analysis/github-screen-receipt.json)
bind exact source/tool/package hashes, boot58592e4f…, workspace and run_attempt1.

Diamond Port1.8.28 uses pinned upstream67b68c22…/selected sourcea9bf0b5d…;
Retcon0.2.0 uses28981c7a…/source5f7dfaae… and API1.3.1acb1858a…;
AIS TRAIN1.2.1 JARb41b5de4…; official Walls1.0 comes unchanged from pinned
sample-bots-Java-1.4.0.zip1707c101…; Runner/server1.4.0 JAR02f6d1d8….
Full, unabbreviated pins and source/license inventories are in the registered
protocol, frozen profile and staged hashes, rather than inferred from version labels.

Preparation failures remain evidence. Run41780412075 stopped before activation
or any game because fresh-venv creation was called twice; it consumed0physical
rounds and no receipt. The original ZIP/log and original c280 freeze are retained.
The reviewed infrastructure repair collects once and only verifies afterward;
repaired zero-game run41782792500 passed one collect plus two read-only checks.
No preparation failure was scored or hidden, and no completed battle was replaced.

## Evidence and next work

[All400 original acquisition files](../../benchmarks/meridian-peak-v1/screen-evidence/)
include five recordings, complete private rows/models/journals, AIS models,
commands/logs, original399-entry manifest, staged source/JARs and runtime bindings.
[Seven exact original ZIP parts, full workflow log and reassembly](../../benchmarks/meridian-peak-v1/screen-artifact-41783652109/)
preserve the complete hosted artifact, including preparation. Independent
[audit output](../../benchmarks/meridian-peak-v1/analysis/original-screen-audit.json),
[provenance/cost/variation audit](../../benchmarks/meridian-peak-v1/analysis/independent-outcome-provenance.json)
and raw links preserve the rejection. The confirmation receipt remains unused;
its180conditional rounds were never authorized by this failed screen.

Any next mechanism must be a distinct, prospectively registered hypothesis with
its own frozen budget/review. This spent candidate will not be retuned or rerun
to rescue its gate. The [all-round exploratory mechanism audit](../../benchmarks/meridian-peak-v1/analysis/next-mechanism-audit.md)
identifies a possible bounded continuation-planning hypothesis, without reconstructing
alternate paths or claiming causation. It can guide a separate registration, but
cannot turn this result into competitive confirmation.

Public independent outcome review: [PR96comment6624153988](https://github.com/stephanelee/robocode/pull/96#issuecomment-6624153988).
