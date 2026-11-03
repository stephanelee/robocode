# Coherent duel v2: restore adaptive aiming, preserve control

Baseline is the shipped [PR #42 composition](improvements/coherent-duel-v1.md), source `728f5a68c2494881f5658d1a80934ef2c97f98d1`, JAR `9daf9cf0e24f319900fee1c8869d50df4ffb004719b1380a8db40fb4961551f6`. Its measured default performance improved, but gun accuracy regressed. Those bundled measurements do not establish which component caused that tradeoff.

An initial suspected command/fire-gate mismatch was falsified: v1 supplies no legacy aiming hint, so its `current` candidate and its gate both use linear prediction. No patch or matches were run for that unreachable hypothesis.

This coherent gun revision instead makes a real fourth forecast available:

- Compute the existing per-enemy/per-class GuessFactor profile and circular `TargetingGeometry` forecast at the **same proposed power** as `DuelGun`. Pass the GF-rotated aim angle, predicted range and power together; range comes from the unperturbed forecast, while the candidate point is rotated by the actual GF angle.
- The gun validates that hint and gates the **current pre-turn heading** against the same candidate footprint. Its virtual cohort freezes that same GF angle at actual confirmed origin/power. Head-on, linear and circular candidates retain their existing rules; no new neural model or optimizer is introduced.
- Each accepted fire request prepares residual-GF passage tracking; only `BulletFiredEvent` supplies the actual origin, speed and creation turn. A fresh observation near wave passage updates per-enemy/class GF evidence against the **unperturbed** forecast reference, regardless of which gun was selected. This is counterfactual passage evidence, not an actual-hit label or calibrated hit probability. It is kept separate from the gun's resolved virtual-ray cohorts.
- Local GF profiles retain the existing round decay; class profiles use the existing numeric checkpoint format. Round-local pending shots, waves and histories reset normally. The fixture covers actual request/confirmation separation, non-central GF geometry, passage updates without a hit event and class-profile checkpoint round-trip.

The same gun revision also corrects a verified finishing edge case: zero energy is disabled but still alive in Tank Royale1.4.0, so the bot may fire the minimum legal power at a zero-energy opponent. A small fixed damage margin avoids intentionally leaving an opponent at exactly zero. Existing own-energy reserves and power caps remain; no threshold search is performed.

Default selection/credit, accepted movement, radar and energy-wave timing stay unchanged. Legacy policy, FIXED/EVAL and melee retain their existing paths. Build the complete bot/JAR and run component fixtures first, then collect **one** prospectively reviewed whole-product comparison against frozen PR #42 AISurvivor, pinned Retcon and samples. No component matches or causal aiming-gain claim precedes that result.
