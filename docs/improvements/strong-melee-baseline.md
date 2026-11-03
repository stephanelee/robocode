# Frozen AISurvivor versus a more sophisticated melee roster

The user requested stronger melee opponents including Retcon and explicitly authorized **"Include Python opponents"**. This is a descriptive60-round baseline, not a candidate promotion or automatic optimization. Freeze accepted master `1e31cceca5cf7f0ffe59dcbdc5f71015a4945871` (production946), JAR SHA256 `9679fc5ee27b68877f0b6d89a2104ffff115d6251ec44d3036fdd1f1d64988a6`; unmerged external PR47/48 and live GUI/server7654 are excluded/untouched.

## Exact roster and provenance

- AISurvivor: actual default TRAIN/controller, private missing checkpoint/metrics per game, all20 online rounds counted. Three games `strong946-g01/g02/g03`, AIS seeds20262601/20262602/20262603, **60 total rounds**, no model reuse. Normal transition to duel behavior after eliminations is part of the product; record action/eligible/forced counters without attributing every turn to one component.
- Retcon0.2.0, unchanged `robocode-dev/rumble-bots` commit `28981c7a469ac064194b41001cd7183ec5ecf189`, MIT; existing source/API pins in [retcon.lock.json](../../tools/opponents/retcon.lock.json). It explicitly advertises melee, with wave surfing/virtual head-on and linear guns; the earlier 1v1 results are not a melee ranking.
- Adaptive Prime1.0 and Diamond Port1.8.28: native Python Tank Royale implementations from `fduch-stranger/robocode-bot` commit `67b68c22e41c18281a3fce9ef638e3c23a0b47cf`, explicit melee descriptors. Adaptive uses minimum-risk melee movement and multiple adaptive guns; Diamond Port implements sophisticated GF/displacement-style targeting/movement. Their source-based complexity motivates this sparring roster; small upstream self-reported trials do **not** establish championship strength. Both share a repository/core, and the Python Diamond port is not assumed equivalent to the classic Java bot.
- [Immutable selected-source/runtime lock](../../tools/opponents/strong-melee.lock.json):110 verified files (selected two bots + shared `bot_core` + requirements/license), canonical SHA256 `a9bf0b5d62b510ce3a405db9dcbe11cf2c1bce3f982e2a5d12b9fdc76b8fbc92`. Adaptive/shared core Apache-2.0; Diamond retains its Zlib notice. No Tomcat, developer scripts, `.env` or `.env.guns` files are packaged.
- Python3.13 and Bot API1.3.1 with the complete12-wheel dependency closure pinned by exact URL/size/SHA256 in the lock; install offline/no-deps from verified wheels into a private venv, no ambient/latest resolution. Original native launchers and all opponent logic remain unchanged. Explicit interpreter, no inherited ROBOCODE/PYTHONPATH overrides, nonexistent developer config paths, fresh per-game copied shared core and bot folders. Record runtime versions/installed packages and preserve notices.

## Fixed setup and reporting

Reuse existing `benchmark.battle(policy='train')` with four opponents total: AISurvivor, Retcon, Adaptive Prime, Diamond Port. Official Tank Royale1.4.0 runner,1000×1000 melee, min/max4 participants,20 rounds/game,100000µs turn timeout, isolated embedded ports. Ubuntu26.04, Java17 fresh build/tests/JAR parity then JDK25 Retcon/runtime plus Python3.13. Only AIS RNG is explicitly seeded; arena starts and other bots retain native defaults.

There are **no strength/skip acceptance thresholds**: preserve every round, skip, collision, failure and opponent result. No tuning, policy suite, replacement seed/game or favorable retry. Acquisition/config/schema failure stops and retains partial evidence for independent review. Native Python/olderAPI compatibility is measured, not assumed. Require exact official names/versions/unique IDs and three finite cold AIS endpoints with counters1..20; native Retcon export is collected as offline diagnostics, never controller input. Other opponents' timing/skip data may be unavailable and must not be invented.

Before outcomes, independently clear the exact source/pin/setup/workflow. Report per-game official rank, score/share and first-place counts for all four bots, AIS shooting/damage/collisions/skips/selection counts, cold checkpoint hashes and Retcon native diagnostics with scope limits. Keep the earlier sample60 and Retcon1v1 cohorts separate. Preserve immutable registration/layout/source/launch/runtime/license/API hashes, all actual raw results/logs/models and failures; document outcome/PR/evidence in `docs/improvements.md`. No gameplay/instrumentation changes or automatic optimization loop follows this baseline.

## Actual result: completed negative baseline

[PR #52](https://github.com/stephanelee/robocode/pull/52), [completed scored run41648769599](https://github.com/stephanelee/robocode/actions/runs/41648769599), frozen execution `4411f6bdd8459af73d85bce81c6b49d4c71ea8fd`. All60 rounds, three cold finite20-round AIS endpoints, official four-bot identities and complete native Retcon exports validated. **Retain the complete descriptive evidence: AISurvivor finished fourth in every game.** There is no strength acceptance gate or gameplay promotion; losses, collisions and all opponent results are retained.

| Bot | Official ranks across three games | Equal-game mean share | Official first places /60 |
| --- | --- | --- | --- |
| Diamond Port | 1,1,1 | 46.505442% | 51 |
| Adaptive Prime | 2,2,2 | 24.396867% | 4 |
| Retcon | 3,3,3 | 16.885499% | 4 |
| AISurvivor | 4,4,4 | **12.212192%** | **1** |

| Game | Diamond score /share /first places | Adaptive score /share /first places | Retcon score /share /first places | AIS score /share /first places |
| --- | --- | --- | --- | --- |
| strong946-g01 | 6380 /46.168319% /18 | 3372 /24.401187% /2 | 2257 /16.332586% /0 | 1810 /13.097909% /0 |
| strong946-g02 | 6314 /46.256410% /16 | 3283 /24.051282% /1 | 2142 /15.692308% /2 | 1911 /14.000000% /1 |
| strong946-g03 | 6339 /47.091598% /17 | 3330 /24.738132% /1 | 2508 /18.631602% /2 | 1284 /9.538667% /0 |

AIS telemetry totals:1372 confirmed shots,347 hits (**25.291545%**), damage dealt2659.078721/taken5468.135071,549 wall collisions,309 bot collisions and zero skipped turns. The `turns` field sums83232 round-scope values; action counts sum24222. **Do not interpret83232 as precise AIS time alive or use it to infer the skipped-turn/latency cause.** Strategy counts0..5 are `[6298,3262,3375,3617,3912,3758]`;2652 selections were duel-eligible/forced (0/1397/1255 across games). The default product includes a multi-enemy phase and later duel transitions; these counters are not a full component timeline.

Native Retcon exports cover20 rounds each, with zero self-reported skips/over-budget turns and compute maxima9507/11245/11863µs. Those timings describe **Retcon only**, not AIS or Python timing; equivalent Python skip/timing data was not collected. The native Python runtime was CPython3.13.17, the private12-wheel installation passed `pip check`, and all109 staged Python/core/license file hashes per game matched the immutable source lock. No developer environment/config files or private opponent input entered AIS.

This is evidence of a serious weakness against this specific more sophisticated roster. The earlier60 sample rounds and Retcon1v1 confirmation are separate populations, not pooled or controlled replacements. Three games do not establish a universal ranking, isolate movement/gun/learning causes or prove the native port equivalent to a classic champion. No score-driven tuning, replacements or additional performance attempts occurred.

### Acquisition chronology and final integration

The initial [run41647764010](https://github.com/stephanelee/robocode/actions/runs/41647764010) failed **before any battle**, because absent pinned zero-byte package files were mistaken for existing empty bytes. Its original artifact13234364075, source/report and failure log are preserved. The [immutable prospective continuation](strong-melee-baseline-continuation.md) and minimal1131851 file-existence regression were independently cleared before the corrected acquisition. The original60-round IDs/seeds were unspent, not favorable replacements. Only the corrected run contains scored games; both executions remain auditable.

The **measured** AIS snapshot remains pre-warm-up1e31/production946/JAR9679. Accepted master later merged warm-up PR47 and the separate negative1v1 report PR51; final documentation integration preserves those accepted changes and the root README rather than reverting production. This report does **not** claim to benchmark the newer warm-up JAR. The one-shot workflow was removed before integration/publication. [All original60 records, three models, official/native results, licenses, pins and zero-game failure](../../benchmarks/strong-melee-baseline/README.md).
