# Prism implementer prompt

Give the following prompt and the [behavioral specification](diamond-port-clean-room-specification.md) to a fresh implementation agent. Provide a filtered checkout that excludes the reference source and restricted artifacts described below.

---

Build a new standalone Java 17 Tank Royale bot named Prism under `experiments/Prism/`.

Use `docs/diamond-port-clean-room-specification.md` as the behavioral specification. Wherever it proposes the name Facet or `FACET_*` paths, substitute Prism and `PRISM_*`. Keep Prism's code, descriptor, JAR, configuration, models and telemetry separate from every other bot.

Your task is to implement and validate the bot, not merely propose a plan. Read `AGENTS.md` first, inspect git status, and preserve all unrelated work. Leave production AISurvivor unchanged.

Maintain the clean-room implementation boundary: work from the specification, official engine/API contracts and independently constructed tests. Do not inspect Diamond's implementation, archived source copies, evolved startup dataset, private models or source-derived expected decisions. If you have already seen that code, disclose the exposure before implementing.

Implement in stages:

1. Official API integration, lifecycle, native coordinates and event chronology.
2. Engine-native movement prediction, swept circular geometry, histories, bounded nearest-neighbor storage and wave tracking.
3. Independent analytic startup targeting, contextual long/recent guns, virtual-gun selection, power management and radar.
4. Enemy-fire inference, learned danger, two-wave surfing, wall safety and confirmed-bullet shadows.
5. Melee targeting, minimum-risk movement and transitions to the final duel.

Use the specification's independent initial profile. Document assumptions and deviations. Do not copy Diamond's startup data or replace missing requirements with undocumented tuning. Distinguish confirmed bullets, inferred shots, forecast waves and virtual learning observations.

Verify engine timing, stopping, reversal, first-step wall safety, firing confirmation, scan-energy visibility, learning chronology, round resets and numerical bounds before performance testing. Produce a reproducible build, standalone runnable JAR, meaningful tests and a README.

Before any gameplay experiment, register the hypothesis, exact source/JAR/configuration/opponent/runtime pins, fixed budget and decision rules in `docs/` and the relevant issue. Obtain the required independent premeasurement review. Use the specification's proposed screen as the starting protocol; freeze it before observing results. Preserve every failure and invalid attempt. Do not automatically retune or launch confirmation.

Use isolated headless runners. Do not connect to the interactive server on port 7654 or consume unrelated reserved diagnostics. Compare against pinned Diamond Port, Retcon, Adaptive Prime and existing pinned official samples. Report complete results without claiming strength from fixtures or isolated wins.

All GitHub Actions jobs must use `ubuntu-26.04`. Build/test with Java 17 and preserve source/committed-JAR parity. Finish Gradle builds before using JDK 25 for Retcon's source launcher. Preserve Stéphane Lee's author identity and published history; add no AI authorship trailers.

Create a PR with implementation, tests, deviations and evidence. Do not merge your own PR; a separate agent must independently review and merge after required CI passes. Update `docs/improvements.md` after every experiment with budget, outcomes, accept/reject decision, PR and raw-evidence links.

Finish with artifact and PR links, build/test results, source/JAR hashes, measured results if any, and explicit remaining limitations. A duel-only implementation is a milestone, not completion of the full specification.
