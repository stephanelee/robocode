# KeelHelm 0.2.0

Adds a dedicated melee policy to the KeelHelm duelist: per-ID legal scans, continuous radar coverage, minimum-risk movement, target selection and bounded circular intercept fire. At one remaining opponent it clears transient tracks and waves, then resumes the original duel policy. From-start 1v1 retains the original surfing, learning gun and energy policy.

The melee module keeps opponents separate and does not train duel waves from mixed identities. It searches 64 directions with engine-native prediction, opponent risk, wall clearance and recent-position avoidance. It does not use observer data or opponent internals.

The completed tournament uses the frozen 0.1.0 bundle; historical results remain unchanged. The 0.2.0 benchmark and decision are documented separately in docs/improvements/keelhelm-melee-v1-protocol.md and issue #123. Contract fixtures alone establish no competitive strength.

Build/test with Java 17: ./gradlew -p experiments/KeelHelm clean build. Run the generated build/bot/KeelHelm bundle or committed JAR with bot/ launch files. Optional telemetry: KEEL_HELM_METRICS; initial-side seed: KEEL_HELM_SEED.
