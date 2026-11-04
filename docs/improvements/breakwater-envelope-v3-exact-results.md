# Breakwater 0.3 scored zero against Diamond and swept the AISurvivor transfer

**Reject against Diamond Port.** Breakwater fired three shots and missed all three in the 20-round pilot. Its official score was 0 to Diamond's 3,596. It missed both the >20% share and one-win gates, so no confirmation ran. Against production AISurvivor, the unchanged JAR won 20/20 rounds and scored 1,296 to 240 (84.375% share). Breakwater recorded zero skips in all 45 physical rounds, including the separate five-round diagnostic. [Registered rules](breakwater-envelope-v3-exact.md) · [Original machine decision](../../benchmarks/breakwater-envelope-v3-exact/experiment.json) · [PR #69](https://github.com/stephanelee/robocode/pull/69)

| Competitive result | Diamond Port 1.8.28 | AISurvivor 1.2.1 |
| --- | ---: | ---: |
| Breakwater first places | 0/20 | 20/20 |
| Breakwater score share | 0% | 84.375% |
| Fresh, cool, affordable opportunities | 31,392 | 23,078 |
| Confirmed shots / hits | 3 / 0 | 16 / 13 |
| Energy spent firing | 3.45 | 24.20 |
| Damage dealt / taken in Breakwater events | 0 / 2,023.54 | 83.20 / 240.00 |
| Breakwater wall / bot collisions | 0 / 0 | 0 / 0 |
| Earlier v0.2 first places / share | 0/20 / 2.05% | 5/20 / 24.98% |

Breakwater held fire on 31,389 of the 31,392 pilot opportunities. Diamond took every survival point. In the transfer, Breakwater earned all 1,000 survival points, won 17 rounds without firing, and recorded 12 of its 13 hits in round 8. AISurvivor fired 1,314 shots and landed 45. The round summaries show the survival outcome but do not establish why each round ended. Breakwater's raw win callback missed transfer rounds 3, 6 and 20; its 20 official round-end increments reconciled with the final scoreboard. Each competitive matchup used one fresh game. The earlier 0.2 games are historical references; they do not identify a causal or repeatable gain. [Diamond scoreboard](../../benchmarks/breakwater-envelope-v3-exact/breakwater-envelope-v3-pilot/battle.json) · [AISurvivor scoreboard](../../benchmarks/breakwater-envelope-v3-exact/breakwater-envelope-v3-ais-g01/battle.json) · [v0.2 results](breakwater-mobile-v2-exact-results.md)

## What the paired shadow comparison showed

The legacy gun controlled all 157 real shots in the five diagnostic rounds and landed five. Breakwater logged 4,810 legal scans and 2,626 common opportunities. Of those, 2,556 resolved against contiguous later scans; 70 were still pending at round end. There were no gap or horizon masks. The table counts hypothetical rays intersecting the observed target footprint. Every comparison within a row uses the same resolved opportunities.

| Shadow cohort | Opportunities | Jointly resolved | v0.2 ray hits | v0.3 ray hits |
| --- | ---: | ---: | ---: | ---: |
| All fresh, cool, affordable turns | 2,626 | 2,556 | 945 (36.97%) | 1,263 (49.41%) |
| v0.3 proposed angle covered at least two paths | 4 | 4 | 2 | 0 |
| v0.3 eligible with the actual gun heading | 2 | 2 | 0 | 0 |
| v0.3 would hold | 2,624 | 2,554 | 945 | 1,263 |
| v0.2 eligible with the actual gun heading | 157 | 157 | 6 | 6 |

Across all resolved opportunities, 723 favored only the candidate ray, 405 favored only the legacy ray, 540 hit with both, and 888 missed with both. The +12.44 percentage-point difference in virtual containment arose on turns the candidate would decline. The two-path agreement rule selected only four opportunities and none of their candidate rays hit. Current-heading eligibility is conditional on the legacy gun's actual slew; a separately controlling candidate can produce a different sequence. The live pilot measured that sequence and confirmed that the gun almost never fired. Correlated opportunities and hypothetical ray hits do not measure actual bullet damage or competitive strength. [Raw opportunity and scan streams](../../benchmarks/breakwater-envelope-v3-exact/breakwater-envelope-v3-shadow/)

## Reproduction and provenance

The [authorized launcher](https://github.com/stephanelee/robocode/blob/e5780d81ebf6ae78bead3f719242c20380dbf84d/.github/workflows/breakwater-envelope-v3-exact.yml) ran once, head `e5780d81ebf6ae78bead3f719242c20380dbf84d`, using frozen payload `59e27e172eb6010877c41fa06dcc7817c997d407`, source `4bedfd5025133dda34a1037a19ff0215a43d8e58`, and JAR SHA-256 `e6c98feef83e083149df782d643496fb6b6fe3fab33c95356ea2634981c6baf0`. Source/JAR parity passed under Java 17.0.21+9. The battle runner used Temurin 25.0.5+8, Python 3.13.17 and all six frozen runtime hashes on ubuntu-26.04. Each game used a fresh process and private endpoint. The seeds were 533000 for shadow, 533001 for the Diamond pilot and 533021/533022 for the transfer. The prior zero-game attempt remains separate. [Full raw files, hashes and offline replay](../../benchmarks/breakwater-envelope-v3-exact/README.md) · [Sole authorized run](https://github.com/stephanelee/robocode/actions/runs/41721148640) · [Original invalid attempt](breakwater-envelope-v3-results.md)
