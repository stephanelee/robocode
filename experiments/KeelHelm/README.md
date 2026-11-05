# KeelHelm 0.3.0 follow-up candidates

Adds a dedicated melee policy to the KeelHelm duelist: per-ID legal scans, continuous radar coverage, minimum-risk movement, target selection and bounded circular intercept fire. At one remaining opponent it clears transient tracks and waves, then resumes the original duel policy. From-start 1v1 retains the original surfing, learning gun and energy policy.

The melee module keeps opponents separate and does not train duel waves from mixed identities. It searches 64 directions with engine-native prediction, opponent risk, wall clearance and recent-position avoidance. It does not use observer data or opponent internals.

The completed tournament uses the frozen 0.1.0 bundle; historical results remain unchanged. The 0.2.1 benchmark and decision are documented separately in docs/improvements/keelhelm-melee-v2-protocol.md and issue #123. Contract fixtures alone establish no competitive strength.

Build/test with Java 17: ./gradlew -p experiments/KeelHelm clean build. Run the generated build/bot/KeelHelm bundle or committed JAR with bot/ launch files. Optional telemetry: KEEL_HELM_METRICS; initial-side seed: KEEL_HELM_SEED.

Measured top-four melee comparison: score share 13.19% → 26.98%, first places 7/80 → 27/80; Retcon/sample guards pass. See [full report](../../docs/improvements/keelhelm-melee-v2-results.md). The earlier 0.2.0 lifecycle failure is retained separately. This limited cohort result establishes no general-strength claim.

Four prospective melee interventions are controlled by KEEL_HELM_MELEE_FEATURES: bit1 adaptive head-on/linear/circular guns; bit2 accuracy-aware firepower; bit4 shot-opportunity targeting; bit8 predicted crossfire risk. Zero retains0.2.1 policy decisions. The follow-up study tests one new bit per stage against the last independently accepted mask, with parallel melee and1v1 games on minimind. Only accepted bits become launcher defaults after the study; inactive/rejected bits remain optional experiments. See docs/improvements/keelhelm-melee-followups-protocol.md and issue125. No follow-up result is claimed before measurement.
