# Melee semantic/evidence follow-up — 2026-11-04

Completed offline implementation and fixture acceptance only. Independent approval and actual three-bot transition validation remain pending for these changed bytes. Zero battles, datasets, tuning, external source, plugins, agents, publication or Gradle. The filtered OS boundary stayed in place. No temporary runtime path was added. Parent owns issue/PR integration, independent review, registered four-round own-three-bot diagnostic and final freeze; no approval transfers from earlier artifacts.

Definition and prospective protocol: [decisions](decisions.md), [protocol](../../docs/facet-melee-followup-protocol.md). The old scan-count denominator is replaced by distinct actual own decision turns with a live legally observed target no more than eight turns old. Repeated scans/decisions, gaps, future observations, terminal replay, death, round retention and cold games are checked. Eight uses the existing risk horizon; no formula or profile constants were tuned. Alignment explicitly uses enemy-to-destination versus own-to-destination bearings in native radians.

Population rows contain before/after displacement long/recent counts and versions, legal scan times, life state and beganMelee. They appear only on population changes, including cold round start. A2->1 transition clears unsubmitted multi-target plans and requires a survivor scan at or after the transition decision. Pending accepted requests remain frozen. Actual plan preparation/submission/confirmation carries the same plan identity, selected gun, target, mode, label convention, projected range/policy/power and store summaries. Combined-melee analytic fallback is labeled ANALYTIC rather than claiming displacement use. Additive trace schema2 fields stay under100000 lines; model schema2 byte layout is unchanged and the semantic definition is pinned by source hash. No full model is dumped per turn.

Deterministic telemetry records movement dispatch/candidate/destination counts, wall/bot rejections, eight-step certification, quota versus all-unsafe emergency with chosen control/cost, projection passes/samples/solve iterations and shared predictor steps. Terminal no-learning is distinct from learning disable/failure. Disabled melee gun now uses the existing analytic fallback instead of rerunning the learned projector. A gun failure in shot preparation cannot be erased by successful later work in the same decision or counted twice toward disable in that decision.

## Results and evidence

[Final Java17 offline build](evidence/followup-final-2.txt): **19 groups /10,169 assertions**, retaining the19/10,110 baseline and supplied866 physical transitions. All existing gates remain. Runnable [Facet.jar](Facet.jar), package smoke, deterministic duplicate packaging and **835-entry source/JAR payload parity** pass. Telemetry/lifecycle fixtures pass, including four unique terminal game-round rows. [Synthetic follow-up trace](evidence/FACET_followup-trace.jsonl) has44 rows and two matched fired callbacks, population2 and fresh-scan survivor population1, with frozen combinedmelee/recent DISPLACEMENT metadata. These are synthetic callbacks, not live shots.

Filled active-path fixture:4 targets,8 waves,16 full displacement stores,10,444,800 estimated payload bytes;437 shared predictor steps,1 projection pass,80 samples,197 solve iterations and8 wall/bot-safe selected steps. Full duel planner is not called. Separate forced-quota, all-unsafe, stopped/unknown/signed/stale/corner fixtures pass. All-unsafe chooses the safest imminent emergency among existing controls; collision avoidance is not promised where no safe control exists. Actual injected movement-history and gun-query failures disable their subsystem after three consecutive decisions, prevent reruns and reset per round. Model export digest is deterministic and changes with the normalizer; round learning persists and cold game counters start empty.

Runtime: Homebrew OpenJDK17.0.21.0.3+0, macOS27.2 arm64,256MiB tests; [metadata](evidence/build.json). Existing20-iteration/256-wave offline planner fixture p50=580666ns/max=1162625ns; this is not runner latency acceptance.

## Exact SHA256 pins

- JAR: `0c908121d8caba91cc3a872d5f8ad2793af42e7fc11f88537015018cc902c0de`
- [Source manifest](evidence/source-files.sha256) content: `4a8690815b07e425dc5b337a9782b6294a70434d115b2c9d1e00793fc1c5f0db`
- Official API1.4.0: `ab65c4d5cec1808adeb71375adae6d15341ae250def89a6c10fc9da879de0752`
- Export fixture model: `1f96f1225ea0a154f746cbb520420f6870499ddcc46589056cd01f42cfa11c31`
- Follow-up synthetic trace: `8db53b74456e78d5a9b8fb52b7b05edf2aae2f46a0951318213b826712eb43bf`
- Final raw build output: `43124372dd9512e88f5f733fea6342f9c3f8f6529d714398f9a02913261d1b65`

## Attempts and remaining limits

[Attempt1](evidence/followup-attempt-1.txt) failed the new gun fault-injection assertion: the test modified a cloned feature array and therefore did not inject failure. Corrected to a finite extreme feature that overflows query distance; no check was lowered. [Attempt2](evidence/followup-attempt-2.txt) passed10142; [attempt3](evidence/followup-attempt-3.txt) passed10145; [attempt4](evidence/followup-attempt-4.txt) added file-backed transition telemetry; [final1](evidence/followup-final.txt) passed10153; final2 adds explicit selected eight-step envelope assertions and recorded active work. Earlier pins are superseded. An initial multi-file patch validation failed atomically before correction. OS-denied system Python cache writes produced warnings during hash inspection, without relaxing the boundary or preventing successful hashes/builds.

Conservative freshness/envelope conventions, bounded12-iteration displacement solve, up to3 power-coherence stages of80 samples each, analytic/cycling fallback and emergency limitations remain approximations. No live required behavior is newly claimed: actual population progression, confirmed selected paths, cold process reset, latency/GC/skip/collision behavior and independent hit/terminal reconciliation require the parent's registered diagnostic. Unexercised conditions remain gaps without replacement rounds. No competitive-strength claim or independent approval.
