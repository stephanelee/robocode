# Aegis: bullet shielding against Diamond Port — report

Tracking: [issue #71](https://github.com/stephanelee/robocode/issues/71). PR: [#70](https://github.com/stephanelee/robocode/pull/70). Registration and outcome: [improvements/aegis-shield-v1.md](improvements/aegis-shield-v1.md). Bot: [experiments/Aegis](../experiments/Aegis/README.md).

## Result

**Rejected.** In the single registered fresh 20-round pilot against pinned Diamond Port 1.8.28, Aegis 0.1.0 scored **0.234%** of total score and won **0 of 20 rounds**. It had zero skipped turns and the evidence was valid. Confirmation was not run. This does not show that Aegis, or any idea here, beats Diamond.

## What was tried and why it differs from earlier challengers

Earlier challengers (pressure, context gun, wave defence, wave-horizon correction, guarded learned-ray selector) tried to out-aim or out-dodge Diamond. The best of them reached 8.3% share. Aegis instead tried to make Diamond's own learned gun predictable, then destroy its bullets:

1. **Shot inference** from legal energy drops, with one-scan-late corrections. The bullet's origin is the enemy scan one turn earlier, which development probes showed to be exact.
2. **Zero-label replay movement.** Diamond logs a wave every scan. A pinned-physics dynamic programme keeps Aegis laterally where it was one wave flight earlier, so the wave labels stay at GF 0 and Diamond aims its centre candidate.
3. **0.1-power interceptors** solved against the exact Tank Royale 1.4 bullet rules: pre-move origin, previous-turn gun direction, `[tick, tick+1]` segment intersection.

## What the evidence shows

| Finding | Evidence |
| --- | --- |
| A stationary target is hit by every Diamond shot, with lines within about 3 px | development probe |
| Our movement trains Diamond's aim: 12 px shifts produced ±7.5–9.5 px aim offsets | development probe |
| Replay movement made Diamond's aim predictable: mean label \|GF\| 0.17 → ~0.013, centre candidate in ~97% of aims | dev-only instrumented opponent copy; never used at runtime or in registered games |
| Interception works: 1,119 of 1,560 registered pilot shots destroyed (71.7%) | registered pilot |
| It is not enough: 363 hits taken (1,892.8 damage) against 7.6 dealt; every round lost | registered pilot |

Why it lost (mechanism, not a causal proof):
- **Each leak is expensive.** It costs us about 4p + 2(p−1) damage and refunds Diamond 3p energy. At p = 1.95 that is 9.75 damage and a 5.85 refund.
- **The low-energy endgame favours Diamond.** Below 63 energy Diamond's power falls cubically, and a fixed 0.1 interceptor then costs as much as its bullet. Our simulation put the break-even block rate near 90%.
- **There is no headroom for offence.** The 12-turn shield reload almost matches Diamond's fire interval, and Aegis has no gun offence.
- **Remaining failure sources** (development classification): Diamond occasionally choosing the adjacent ±1 candidate (≈12 px), residual self-prediction error up to ~3 px from Diamond's unobservable movement commands, low-coverage shields from limited lateral speed, and reload conflicts when contacts compress.

## Limitations

- It is a single 20-round game. The policy is deterministic, so variation comes only from the opponent, spawn positions and scheduling.
- Development games ran on a local container, not GitHub runners, and used evolving source versions. They are exploratory only.
- The full registered raw artifact still needs importing (see the evidence README).
- Diamond Port is a Python port, not the original Java Diamond.

## Possible next directions (not registered, not evidence)

- Make the endgame economic rather than purely defensive: for example, stop shielding sub-0.3 bullets only when an offence can use the freed reload. That requires a real gun.
- Keep range under 325 px so Diamond never powers down. It flees, and close range raises its power to 2.95.
- Combine interception with dodging for the adjacent candidate.

Any of these would need a new prospective registration and fresh games. Nothing here transfers to AIS.

**Follow-up screen (unregistered):** the dodging direction was screened first and rejected without spending registered games. When Aegis dodged every shot, Diamond's aim spread to candidates ±1–4 away from centre within four rounds: centre-candidate aims fell from ≈97% to 25 of 326. Every round was lost, with 11 hits taken each. Real-shot labels therefore drive its aim, so dodging destroys the predictability interception relies on. See [the screen record](../benchmarks/aegis-shield-v1/development/dodge-screen/README.md).
