# Aegis 0.1.0 — isolated 1v1 bullet-shield experiment

Aegis is a separate experimental Tank Royale bot for the [pinned Diamond Port 1.8.28 opponent](../../docs/opponents/diamond-port.md). It is not AISurvivor or DiamondChallenger and shares no source, checkpoint or metrics with them. Production AIS source and JAR are unchanged. The registered protocol and outcome are in [docs/improvements/aegis-shield-v1.md](../../docs/improvements/aegis-shield-v1.md).

## Strategy

Prior challengers lost the damage exchange: they failed to out-aim a strong surfer and failed to dodge a strong learning gun. Aegis does neither. It makes the opponent's own learning gun **predictable**, then shoots its bullets down.

1. **Legal shot inference.** Each enemy shot is inferred from consecutive radar-locked scan energy drops, with one-scan-late corrections for our bullet damage, the shooter's hit bonus and shared inactivity damage (`Aegis.inferredPower`). The bullet's true origin is the enemy's scanned position one turn before the drop (the pinned engine creates bullets from the pre-move position).
2. **Head-on line model.** The bullet is modelled as aimed at our position predicted one turn ahead from the scan two turns before creation (position + speed along heading). Its angle is taken from the shooter's classic-order self-prediction, inferred from its observed speeds (`Aegis.classicSpeed`).
3. **Zero-label replay movement.** A learning gun that logs a wave every turn learns guess factor 0 if, for every wave, we are laterally back where we were when it was launched as it breaks. An exact dynamic programme (`LateralPlanner`, pinned 1.4 speed law, half-pixel lattice) moves along one fixed axis and minimises the lateral distance to our own position one wave-flight ago (`REPLAY`). It also asks for full speed in the four turns before each predicted enemy shot (`FAST_*`). The shield's crossing geometry is fixed by that motion, because the replay repeats it at contact.
4. **0.1-power interceptors.** `ShieldSolver` searches fire turns and directions using the engine's exact bullet contract (pre-move origin, previous-turn gun direction, `[tick, tick+1]` segments, segment–segment and segment–circle tests). It maximises coverage of ±4 px lateral errors around the modelled line, with the centre weighted most. A chosen plan locks the trajectory until intercept. The scheduler leaves reload time for the next shot. Weakly covered or unshieldable shots get a lateral dodge cost.

`DiamondLabel` replicates the opponent's footprint label geometry. Its constants were checked offline against labels produced by the pinned opponent's own code in a private dev-only instrumented copy. At runtime it is used only for the `meanLabelError` metric; no opponent state, file, RNG or timing is read.

## Why it could work, and its known limits

Development probes (unregistered, see the report) found that the pinned opponent aims a stationary or zero-label target almost exactly: its centre candidate angle wins about 97% of aims once labels are near zero. Its gun ignores bullet-hit-bullet events. The energy arithmetic, however, is unforgiving. Each blocked 1.95 shot costs it 1.95 and us 0.1, but each leak costs us 9.75 and refunds it 5.85. Below 63 energy the opponent powers down cubically, and at about 0.1–0.3 power our fixed 0.1 interceptor costs as much as its bullet. Simulation says roughly 90% blocking is needed to win most rounds. Unregistered development games reached about 74–80%. Aegis has no gun offence and no cross-round learning or persisted model.

## Build, test and run

```sh
./gradlew -p experiments/Aegis --no-daemon clean build   # Java 17; 11 JUnit tests
```

The runnable directory is `experiments/Aegis/build/bot/Aegis/`. The tracked `Aegis.jar` must match a fresh Java 17 build (`tools/verify_jar.py`). Environment: `AEGIS_SEED` (recorded only; the policy is deterministic), `AEGIS_METRICS_PATH` (must be absent, one JSONL row per round) and optional default-off `AEGIS_TRACE_PATH` (diagnostic event trace). There is no model path, because Aegis persists nothing.
