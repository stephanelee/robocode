# KeelHelm v1

[Issue107](https://github.com/stephanelee/robocode/issues/107), [PR108](https://github.com/stephanelee/robocode/pull/108), [prospective protocol](../../docs/improvements/keel-helm-v1.md), [fixed game registry](games.json), [unchanged source/JAR/tool freeze](freeze.json).

**Registered screen PASS: all14cold games/120rounds valid,0own skips, no invalid or unplayed IDs.** Independent replay reconciled2,630raw-decision fields and1,341authority/binding checks. Candidate logged integer round maxMs reached13, baseline12; all remain below75. This opens fresh300round confirmation eligibility only. **No confirmation games have run and the improvement is not confirmed.**

| Opponent | Baseline mean share | Candidate mean share | Candidate minus baseline | Official firstPlaces baseline→candidate |
| --- | ---: | ---: | ---: | ---: |
| Diamond Port |48.774673%|58.562467%|+9.787794pp|22/30→27/30|
| Retcon |96.755268%|96.072547%|−0.682721pp|20/20→20/20|
| Walls |94.659091%|93.799324%|−0.859767pp|5/5→5/5|
| AIS TRAIN |92.735703%|89.984825%|−2.750878pp|5/5→5/5|

All prospectively fixed screen guards pass. Primary shares are equal-game means; adaptive rounds are correlated and starts/opponent randomness are uncontrolled. Candidate Diamond shares ranged48.826979–69.261637%; one game below50% remains in the registered population. No favorable subset was selected.

The candidate's pooled Diamond incoming hit frequency was10.566663%, versus baseline10.269182%; both arms still lost the bullet-damage exchange. This does not establish aiming superiority or a lower incoming hit frequency. Diamond ram events were2baseline/0candidate, so the known unchanged Keel ram-bookkeeping limitation cannot be assumed inactive or used to discard games. Logged maxMs is not a complete latency guarantee. No general strength or isolated causal mechanism claim follows from this finite screen.

Full [result/limitations](../../docs/improvements/keel-helm-v1-results.md), [independent outcome report](screen-evidence-41874753112/provider-and-audits/keel-helm-screen-outcome-review.md), [all14game measurements](screen-evidence-41874753112/provider-and-audits/keel-helm-screen-41874753112-games.csv), [all120round measurements](screen-evidence-41874753112/provider-and-audits/keel-helm-screen-41874753112-rounds.csv), [raw evidence manifest](screen-evidence-41874753112/manifest.json), [sole screen run](https://github.com/stephanelee/robocode/actions/runs/41874753112).

The exact original final provider ZIP is117,899,908bytes/SHA2568c49b360f4a442b3f241734ce0e4b662faabb8e44bd986da2a080cc7cbbbc14b, retained in57verbatimparts. All1,349logical original files remain lossless and **all14whole original `.battle.gz` recordings** are direct regular files. Raw decision.json remains byte-identical (SHA256bb4a2e40d438c25ff6f82083de26a6c90954325f2789d4c63ce4ee457ce8bb0c), retaining original screen head368b49a9bae42a0cefd4fd9aa033a6bd576e566d. Original [pregame provenance](screen-provenance-41874753112/manifest.json) and separate actual-worker approval are preserved. The permanent screen receipt5add0bbade3239772c0350a8b0bf40a6b06e0e41 remains unchanged.

Verify retained bytes only (no metric reduction or gameplay):

```sh
python3 -I -B -S benchmarks/keel-helm-v1/screen-evidence-41874753112/verify.py
python3 -I -B -S benchmarks/keel-helm-v1/screen-provenance-41874753112/verify.py
```

Conditional confirmation requires publishing full unchanged evidence, a proper independently reviewed evidence gate, new exact-head review, an actively assigned separate reviewer before dispatch and fresh actual same-worker runtime admission. Screen and confirmation may not be pooled. Maximum registered budget remains420physical rounds:120screen consumed,300fresh conditional rounds unplayed, no replacements. Original Keel baseline, production AISurvivor, candidate source/JAR, protocol, registry and source freeze remain unchanged.

Candidate Java17:15tests and clean788entryparity, including independent exact rebuild; harness21puretests and reviewed integrity corrections. Initial failed pure fixtures/fixes remain preserved in [reviews](reviews/), [harness checks](harness-validation/) and [candidate validation](../../experiments/KeelHelm/validation/java17-local/); fixtures are not competitive evidence.
