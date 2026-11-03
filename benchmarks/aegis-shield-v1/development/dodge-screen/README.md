# Unregistered dev screen: interception plus dodging every shot

This screen follows the third "next direction" in [docs/aegis-shield.md](../../../../docs/aegis-shield.md). It is exploratory development only. No registered game was spent, no strength claim is made, and nothing here is pooled with registered data.

**Variant.** The frozen Aegis 0.1.0, modified by `dodge-all.patch`. Every inferred shot carries a dominant lateral-clearance cost: at least 28 px from the modelled centre-candidate line between contact−2 and the pass turn, with weight 200 against a replay weight of 10. Interceptors stay as backup. The variant was built locally and played 4 rounds (seed 3) against a **dev-only instrumented copy** of the pinned Diamond Port, which logs its labels and aim candidates and is never used at runtime.

**Result: rejected at screen.**

| Measure | Dodge-all screen | Registered 0.1.0 pilot |
| --- | --- | --- |
| Rounds won | 0 / 4 | 0 / 20 |
| Hits taken per round | 11, every round | 18.2 |
| Shots blocked | 5 of 80 inferred | 71.7% of inferred |
| Mean label \|GF\| (all waves / real shots) | 0.075 / 0.133 | ≈0.01–0.015 in development |
| Aims on the centre candidate | 25 of 326 | ≈97% in development |

**Measured finding.** Dodging moves exactly the real-shot (firing-wave) labels. Within a few rounds Diamond's aim spread to candidates ±1–4 from centre, so it was predicting our dodges. That destroyed the predictability that interception depends on. Its real-shot labels are therefore influential, despite the low 0.38 weight on its virtuality feature. Short-window AntiSurfer views plausibly explain this, but it was not isolated. Making dodging the main defence is not a viable direction against this opponent.

Files: `metrics.jsonl` and `battle.json` from the dev game, and `instrumented-aim-bins.txt` with the instrumented copy's aim and label summary.
