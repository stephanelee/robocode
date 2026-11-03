# Breakwater vs AISurvivor

Run one fresh 20-round 1v1 game, `breakwater-ais-g01`, with the two unchanged committed bots at `642a3fa6d823cae6a69aece021ce65ba4df6957a`. Breakwater 0.1.0 seed 531821 and production AISurvivor 1.2.1 default train mode seed 531822 each get separate, initially absent metrics and model paths. Round learning stays within this one game. Use the pinned Tank Royale 1.4.0 battle runner, Java 17 parity builds, JDK 25 battle runtime, and an embedded headless server on an isolated Ubuntu 26.04 runner.

This is a descriptive matchup. Report both official point totals, score shares, round first places and recorded skips. Preserve both bots' round rows, final private endpoints, exact launched JAR hashes and runner log. Report incomplete evidence explicitly. No gameplay changes, replacement games, promotion criteria or reuse in the Diamond series.

## Measured result

AISurvivor took all 20 official first places. Both bots recorded zero skipped turns and zero body or wall collisions. Breakwater intercepted 27 enemy bullets with 679 shield shots. Another 91 shield shots hit AISurvivor, and one of eight deliberate attacks hit. AISurvivor hit with 375 of its 739 shots. [Official scoreboard](../../benchmarks/breakwater-vs-ais/battle.json) · [Single run](https://github.com/stephanelee/robocode/actions/runs/41709743219)

| Bot | Official first places | Points | Share |
| --- | ---: | ---: | ---: |
| AISurvivor 1.2.1 | 20/20 | 3,545 | 98.8842% |
| Breakwater 0.1.0 | 0/20 | 40 | 1.1158% |

Both private streams cover rounds 1–20 in order and both cold endpoints reached round 20. AISurvivor's own win field was false in rounds 2, 4, 6, 9 and 19, yielding only 15 win flags against 20 official first places. The cause is undetermined. Use the runner's scoreboard for wins. The bots' event-level totals for AISurvivor hits also differ (1,993.6 dealt and 2,004.0 taken by Breakwater); those are separate from official capped bullet-damage points. This one game supports a descriptive AISurvivor win. No promotion or cross-opponent strength decision was registered.

The [raw files](../../benchmarks/breakwater-vs-ais/) preserve both JSONL streams, the AIS binary checkpoint and Breakwater JSON endpoint, exact participant hashes, captured runtime and runner log. The [executed workflow](https://github.com/stephanelee/robocode/blob/7c10bb6abe8e3dd81633b90b7a8ae36cc5e4ca24/.github/workflows/breakwater-ais.yml) contains the complete reproduction commands. It verified both committed artifacts against their Java 17 builds before the isolated match.
