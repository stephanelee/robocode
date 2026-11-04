# Independent Facet revision4 review

2026-11-04. Reviewed stable new Facet implementation in `experiments/Facet/`, supplied specification, official pinned API/engine contracts, independent fixtures and acquisition infrastructure only. No prohibited opponent implementation/model/private telemetry, unrelated gameplay experiment, source archive or history was inspected. No gameplay files edited. No competitive matches run.

Reviewed committed JAR SHA256: `73be04d7386a60474743a78fcebf3fd0c143e37120b410b18aa668bd5ed50c16`. Root identifies publication commit `124a49d62b5a8a4badda252bbca1d5bbfff07c60` in draft PR103; this report does not assert required CI or merge status beyond independently supplied CI evidence.

## Decision

**Approve the bounded live-adapter diagnostic below. Performance collection is conditional on its successful validation, an exact prospective freeze of the requested300-round acquisition, and the acquisition safeguards below. No performance or merge approval is granted by this report alone.**

The reproducible revision2 blockers have been corrected. The bot can progress to live integration testing; there is no justification for treating synthetic success as observed adapter/engine integration. This remains a duel milestone: melee is gated, existing actual traces do not exercise every skipped/rejected/multicontact callback, and the requested300 rounds cannot establish original Diamond equivalence or general tournament strength.

## Independent validation

- Recompiled current test sources against the frozen committed JAR and ran under Java17.0.21.0.3,256MiB: **18 groups/10,034 assertions passed**, including866 actual motion/shot transitions. Log `revision4/independent-suite.txt`.
- Reran independent prior counterexamples against the current JAR: origin features remain invariant after future appends; broken reserve and updated kill-cap plans reject; late initial heat is[0,1.6]; approaching513-turn threat remains; learned counter changes alter model bytes; collision example chooses STOP/gap37; unequal corner predicts(782,581). Log `revision4/previous-reproductions.txt`.
- Checked the actual implementation's `Waves.shadowSegment` against the official compiled server line-intersection and endpoint broad-phase predicates on **327,701 deterministic angular/edge samples**, with **zero false safety credit**. Cases: partial annulus, disjoint annulus, angular wrap, wrong tick, overlong/stale-segment broad-phase rejection. Source/output `revision4/Revision4ShadowCheck.java` and `revision4/shadow-engine-check.txt`. This independently tests the implemented solver, not merely an independently proposed solver. It complements the broader independent720,917-sample fixture and the implementation's uncertainty/death/pending/quota fixtures; neither is a universal formal proof or live lifecycle acquisition.
- Source-reviewed terminal observation-only drain, duplicate completion suppression, fresh game identity/writers/cold Core, schema2 export, common confirmed gun-rating cohorts, prefix kinematics and per-shooter own orbit history, affordable forecast clocks, full wall horizons and survivor recency stores. These corrections address R2-1–R2-8 and original terminal finding22 at the synthetic/source level.
- Acquisition unit tests passed3 tests; no source/launcher/opponent process was executed by those tests. Infrastructure is still subject to the exact safeguards below.

## Bounded prospective live-adapter diagnostic

Register in docs and issue101 before launch. This is integration evidence, separate from all competitive cohorts.

**Budget:** exactly two fresh cold games of two rounds each (four scheduled diagnostic rounds), frozen Facet JAR above versus a newly written independent stationary ContractPartner. No AISurvivor, Retcon, Diamond or reserved experiment participates. Seeds49667 and49668; reverse launch order in the second game. Official pinned embedded runner,800x600,gunCooling.1,turnTimeout100000us,readyTimeout10000000us,default maxInactivity450,unlimitedTPS. Bound each game to240seconds and the entire acquisition to600seconds. Empty private working/model/trace directories per game. Stop on acquisition/validation failure; retain all attempts and uncollected schedule entries. No replacement games or tuning.

**Partner behavior:** remain stopped; use legal radar scans only; acquire/aim through current-heading timing; request minimum power.1 only when legal/cool/aligned and no more than once per60turns. Otherwise passive, allowing Facet to finish. No future/spectator state, reserved code or private opponent data. Record own callbacks with game/round/eventTurn/currentOwnStateTurn/callbackSequence, bullet IDs/owner/power/position/heading, hit/damage/energy, death, round/game completion, walls and skips. This probe is an independently constructed integration fixture, not a performance opponent.

**Accept only if both scheduled games complete:**

1. Official results show exactly2 rounds and exact Facet/probe identities for each game; launch/JAR/API/runner/probe/runtime hashes and commands are retained.
2. Each Facet file has exactly2 unique `(gameId,round)` rows; round order is1,2; game IDs differ across cold games. New process/input directories prove cold starts; model policy has no loader. Learning may persist between its two rounds.
3. Confirmed Facet shots are nonzero and match unique official `BulletFiredEvent` callbacks cumulatively, with no unmatched confirmation, ghost request or duplicate physical hit accounting. Own shot origins/headings follow the already validated current-heading contract.
4. Probe received-bullet IDs/damage and Facet outgoing hit/damage counters agree, including the fatal terminal hit. Terminal batches are drained once; no duplicate round/game metrics rows and no commands appear after round finalization. Round/model exports are complete and valid.
5. No nonfinite metrics/commands, uncaught/boundary exceptions, learned subsystem failures, unexplained skips or quota violations; all fallback/skip counters retained and reviewed. No claim of a required event family passing when the run did not exercise it.

Do not modify the measured Facet JAR or inject artificial sleeps for this minimal diagnostic. Actual skipped-turn/rejected-request/multicontact acquisition can remain explicit fixture-only limits; existing source and synthetic checks cover their defensive paths. A separate bounded diagnostic would be needed to claim those actual events were acquired. Diagnostic score/wins have no strength interpretation.

## Acquisition safeguards for the requested300 rounds

User's latest scope is five independent cold20-round games each against AISurvivor,Retcon,Diamond Port:300 total. This supersedes the original proposed200-round roster/budget only; keep the same independent strategy parameters and no retuning. Treat this as a descriptive registered screen, not a fresh10-game-per-opponent confirmation or production transfer.

Before adding the trigger freeze file:

- The Java17 workflow rebuilds AISurvivor into its launch directory. After entrywise parity succeeds, restore the saved committed AIS JAR bytes before validating/launching the frozen byte hash, or explicitly freeze the verified rebuilt artifact's actual bytes. ZIP compression/toolchain differences can change whole-JAR digest while payload parity still passes. Preserve the production parity gate.
- Freeze and verify the Facet/AIS JAR and descriptors/config, source manifest, official runner/server/API identities, both opponent lock-file hashes, harness/adapter/launcher hashes, exact runtime versions/OS image and cold model policy. Runtime provenance must be observed, not merely asserted from a freeze. Record actual full launch argv and whitelisted state/seed configuration per game; do not include secrets.
- Verify result versions as well as names/round counts: Facet0.1.0,Retcon0.2.0,Diamond Port1.8.28 and the frozen AIS descriptor version. Validate actual staged opponent bytes each game without reading them into the decision system. Keep private Retcon/opponent telemetry disabled.
- Retain complete official results, private Facet legal trace/metrics/model hashes, runner logs, raw command lines, all preparation/invalid attempts and the exact frozen registration. No retry/resume replaces a failed measured game. Alternating launch order is not controlled engine seats/seeds; state this limitation prospectively.
- Quantitative summary must include per-game score shares and official first places, damage/confirmed shots/hits, collision/skip rates with turn denominators, range/scan freshness, numeric failures/fallbacks and latency/GC evidence. Five game clusters limit precision; dependent rounds must not be labeled independent observations. Do not claim broader strength or confirmation from a favorable subset.

`tools/facet_benchmark.py` already preserves empty isolated cold directories, fixed game structure, artifact checks, private state paths, exact-round metric rows, finite JSON parsing, official runner use and invalid-attempt status. Freeze validation and workflow byte restoration need final verification before performance approval. The trigger file currently being absent correctly prevents accidental acquisition.

## Remaining limits

Full melee activation/review and separate screen remain outstanding. Unsupported continuous shadow origin/time/power uncertainty receives zero reduction, which is conservative and explicit. Forecasts are low-confidence hypotheses and never labels or physical shots; current-origin error8 with a multi-turn heat window should be documented as a forecast approximation rather than claimed to enclose all future firing locations. Exact source/JAR payload parity, CI, live diagnostic and premeasurement freeze must remain visible gates. No merge is authorized by this reviewer yet.

## Live diagnostic disposition addendum

The registered acquisition completed all4 scheduled integration rounds in two cold games, elapsed8.29seconds, without retries. Reviewer independently validated raw files using `revision4/validate_adapter.py`; full derived evidence and hashes are in `revision4/adapter-independent-validation.json`.

**PASS required integration checks with declared probe logging limitation.** Each game has22 unique confirmed Facet shots matching22 own official fired callbacks,20 outgoing hit callbacks matching20 probe receipts, and exactly2 unique completion rows. All4 fatal terminal hits are retained; no post-finalization commands, unmatched confirmations, cancellations, numerical/subsystem failures, skips or gaps were found. Every confirmed origin and heading exactly matches the request state's current position and gun heading. Both model exports have valid end markers and recorded hashes. Raw event damage209.6/200 differs from capped official scoring as expected from overkill; the probe and Facet event sums agree exactly.

The probe's roundStart rows are absent: its startup logger requests state before the first tick. This is a logging limitation, not a passing claim for acquired round-start state. Required round identities, nonzero real shots, cross-hit receipts and final completion behavior were captured. `learning.disabled` increments once per terminal observation-only drain by source control flow; it is not an exception-induced subsystem disable. Actual rejected requests, forced skips and multicontact remain fixture-only limitations and were not silently marked acquired.

**Independent premeasurement decision:** approve proceeding to the user's requested descriptive300-round cold-game acquisition once the final prospective freeze verifies exact reviewed Facet JAR/source/config/descriptors, opponent locks, official runner/API/harness/runtime identities, committed artifact byte restoration and no-retry structure. No additional gameplay fix or integration rerun is required by this review. Final freeze documentary validation remains to be completed before the trigger file is published. This approval is for bounded measurement, not promotion, full-specification completion, merge, Diamond equivalence or confirmed competitive superiority.
