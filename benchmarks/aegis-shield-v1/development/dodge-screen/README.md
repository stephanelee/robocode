# Unregistered dev screen: interception plus dodging every shot

This screen follows the third "next direction" in [docs/aegis-shield.md](../../../../docs/aegis-shield.md). It is exploratory development only. No registered game was spent, no strength claim is made, and nothing here is pooled with registered data.

**Reported variant.** The author describes a local build of Aegis 0.1.0 modified by `dodge-all.patch`, with interceptors as backup. The retained diff raises the desired lateral clearance to 28 px and its cost weight to 200 against a replay weight of 10, and enables the dodge flag for every inferred shot. It is reported to have played 4 rounds (seed 3) against a **dev-only instrumented copy** of the pinned Diamond Port. The exact executed build and instrumentation provenance for this screen are unverified; the retained patch describes the proposed changes, not proof of the executed bytes.

**Decision: reject this tested exploratory variant after the four-round screen.** No registered acceptance or general conclusion about dodging follows.

| Measure | Four-round screen | Separate reference (not paired) |
| --- | --- | --- |
| Rounds won | 0 / 4 | 0 / 20 |
| Hits taken per round | 11, every round | 18.2 |
| Shots blocked | 5 of 80 inferred | 71.7% of inferred |
| Reported mean label \|GF\| (all waves / real shots) | 0.075 / 0.133 (unverified aggregate) | ≈0.01–0.015 in earlier development |
| Centre-candidate count in the reported aim summary | 25 of 326 (aggregate only) | ≈97% in earlier development |

**What the retained records support.** The four metric rows report 11 incoming hits each, five blocks among 80 inferred shots, and zero own skipped turns. The battle summary gives Aegis zero of four first places. The two-line instrumented summary reports a broader aggregate aim distribution than the separate earlier development summary. It has no per-round sequence or retained raw labels from which to reproduce the label means. These records do not isolate changes in opponent learning, real-shot label weighting, shield timing or other mechanisms. Other dodge policies remain untested.

**Provenance limits.** No exact executed Aegis source/JAR hash, executed opponent instrumentation patch/hash, launcher/runtime manifest, raw aim/label rows or raw-to-summary aggregation command is retained for this screen. Consequently the instrumented mechanism measurements and the claim that its modifications were observation-only cannot be independently verified here. The registered pilot and earlier development references use separate cohorts and configurations; their differences are descriptive, not a controlled effect estimate.

Files preserved unchanged: `dodge-all.patch`, `metrics.jsonl`, `battle.json`, and the reported two-line `instrumented-aim-bins.txt`. No missing evidence is reconstructed and no repeat game is proposed.
