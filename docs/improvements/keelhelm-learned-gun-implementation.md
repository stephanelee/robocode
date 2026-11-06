# Learned melee displacement candidate 0.5.0

Prospective protocol: [learned-gun protocol](keelhelm-learned-gun-protocol.md), issue129. The accepted project remains exactly0.2.1. The isolated candidate lives in `benchmarks/keelhelm-learned-gun/candidate-project`; it defaults to the new gun and accepts `KEEL_HELM_LEARNED_GUN=false` for contract diagnostics. It is not a tournament entrant/default.

`MeleeDisplacementGun` bounds per-ID history, actual endpoint labels, nearest distinct-origin selection, uncertainty fallback and query caching as registered. It retains models only within a game. Legal BulletFired events confirm pending shot metadata; actual changed directions and target IDs are traced. The outcome ledger survives the melee-to-duel transition and reports unresolved/evicted bullets as censored. Model observations and pending requests clear at the transition. Damage feedback never affects phase1 aim or power.

Only the candidate's melee aiming endpoint changes. The inherited circular endpoint is captured before asking the model, and is the exact fallback. Movement, radar, target selection, selected power, energy/tolerance gates and duel code remain inherited. Normal opponent/API pins and the100000us deadline are unchanged.

Java17 offline build:29tests,0failures/errors; committed candidate JAR matches the reproducible build across835archive entries. Candidate SHA256 `69b1dec56655f76f8af8a87155806bd0c6bc2e53c19ca3a2742421f9b18d853c`. Six added contracts cover actual endpoint learning, sparse/invalid fallback, model lifecycle, real-shot confirmation, censoring/overflow and body-relative features. These fixtures demonstrate contracts only, not competitive strength.

`tools/keelhelm_displacement.py` prepares/fixes40cold serial games/520rounds with two melee slot rotations and separate duels, freezes all inputs, stages every roster, records resource/cleanup evidence, validates actual changed-shot directions against the observer recording, and uses exact fractions for registered gates. Preparation/review precedes launch; any invalid game stops the comparison without retry/resume. All original artifacts are retained outside Git worktrees.
