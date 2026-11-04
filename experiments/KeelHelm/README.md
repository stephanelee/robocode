# KeelHelm 0.1.0

Isolated experimental fork of confirmed Keel0.1.0. The first-wave stopped suffix preserves the body turn requested by the unchanged orbit policy, using the unchanged engine-native Phys.step at target speed0. Its endpoint heading therefore feeds the second-wave escape consistently. The terminal second-wave STOP branch remains unchanged.

Only the first-leg stopped rollout changes gameplay. Guns, fire power, learning, motion helpers, options, constants and horizons retain baseline bytes. Candidate-only per-round aggregate exposure counters do not feed decisions. Baseline Keel and production AISurvivor are untouched.

Protocol: docs/improvements/keel-helm-v1.md, issue107 / PR108. Pure semantic tests establish modeled command consistency, not competitive strength. No benchmark result is claimed.

Run bundle: KeelHelm.json, KeelHelm.sh / KeelHelm.cmd, KeelHelm.jar. Optional harness variables: KEEL_HELM_METRICS and KEEL_HELM_SEED. All other runtime behavior follows Keel.
