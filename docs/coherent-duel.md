# Coherent default duel candidate

This is one integrated bot candidate, built before a whole-product comparison. It combines the accepted engine-conformant DuelSurf controller with explicit default duel selection, radar, legal energy inference, and a coordinated gun. Component fixtures prove contracts, not competitive strength. The earlier [48-round hierarchy screen](../benchmarks/default-duel-policy-screen/README.md) supports only its original narrow candidate and is not pooled into the integrated result.

## Composition and scope

- **Selection and credit:** default TRAIN selects action zero before sampling when exactly one live enemy and one cached opponent remain after stale pruning. The returned, executed and replay-credited action agree. Forced next-state bootstrap uses Q0; normal six-action selection/bootstrap remains outside that gate. Network dimensions and serialized checkpoints are unchanged.
- **Movement:** retain the accepted `DuelSurf` controller and its literal 1.4.0 speed, old-heading movement, limited turn and wall simulation. No new movement optimizer is introduced.
- **Radar:** `DuelRadar` crosses the latest legal target bearing with bounded overshoot. Scans aged at most two turns support locking; older or missing state triggers a full search. Radar is independent of both body and gun turn in the new duel path. Existing 40-turn cache expiry remains; stale observations are not silently presented as current positions.
- **Energy and waves:** bullet damage and shooter-hit bonus from callback turn k correct scan energy at k+1, matching the [recorded pinned-engine trace](engine-energy-trace.md). Firing costs and wall/ram timing are not shifted. Only consecutive scans infer a shot; gaps retain uncertainty. Enemy waves and own motion history reset each round.
- **Gun:** a standalone `DuelGun` consumes legal scans and actual confirmed own shots. It coordinates physical forecasts, current-heading alignment, firepower and low-energy finishing; its own virtual-shot evidence is separate from the legacy GF histogram. Its exact implementation and deterministic tests are reviewed with this candidate. No private opponent exports, observer state or learned offline model enters live decisions.

These new radar, gun and timing paths apply only to the default TRAIN controller policy in a one-opponent game. `AISURVIVOR_DUEL_POLICY=learned`, EVAL, explicit FIXED and melee retain their existing targeting/radar/timing paths. Explicit fixed action overrides remain honored. TRAIN still updates the DQN from the actual executed action; this composition does not claim that a trained neural policy discovered the controller or caused any measured gain.

## Whole-product validation

Build and unit-test with Java 17, rebuild the tracked JAR, and verify normalized ZIP entry parity. Independent source review precedes performance collection. The bounded prospective roster will compare actual default TRAIN behavior against frozen accepted AISurvivor, pinned Retcon and official Target/SpinBot/Walls. Each bot/game has independent model, metrics, seed and output paths; frozen AIS identities are descriptor-only labels, with original JAR hashes preserved. Seeds initialize AIS RNG, not engine arena starts; swapped order is not a paired-start experiment.

The exact immutable round budget, gates, provenance and workflow must be registered and independently cleared before matches. No component-by-component performance series, extra favorable runs, or claimed win over the roster precedes that whole-candidate result. All official arrays, metrics, model endpoints and partial failures are retained. PR #41's separate narrow confirmation was never run.
