# DiamondChallenger: isolated 1v1 experiment

The user requested a separate bot to fight Diamond Port before applying successful ideas to AISurvivor. This task is tracked under [issue53](https://github.com/stephanelee/robocode/issues/53), alongside the [overall priorities](advanced-opponents.md).

## Objective and isolation

Build a coherent experimental 1v1 challenger and demonstrate repeatable wins against the pinned Python Diamond Port1.8.28. The AIS baseline lost0/60 official first places and averaged3.886%share; preserve that negative evidence. A modest score gain is useful progress, not a claim that the challenger beats Diamond.

Give the challenger its own descriptor/name, runnable artifact, experimental source or reproducible source patch, checkpoint and metrics paths. Build it in an isolated directory. It may reuse reviewed components, but the experimental strategy and its source/JAR hashes must be explicit. Merely changing AIS's display name is not a strategy improvement. Do not replace or modify production AIS source/JAR when publishing challenger work, and do not share live or benchmark checkpoints between the bots.

Keep the implementation small: reuse the existing battle harness and verified native-opponent loader. No new engine, broad plugin framework, model-training stack or melee work is needed. All deployed inputs must come from the challenger's legal scans, own state and combat events. Diamond's source and private telemetry may inform offline research, never live hidden inputs.

## Execution

1. Movement agent develops the Diamond-focused counterstrategy and isolated prototype; benchmark agent handles packaging and collection. Agree a minimal interface directly.
2. Before any games, publish the exact hypothesis, source/JAR/opponent/runtime pins, cold-state policy, budgets, seeds, metrics and decision rules. Independent reviewer clears exact execution; no additional root wait is needed afterward.
3. Use bounded screens to reject weak whole-bot ideas, followed by genuinely fresh confirmation for promising candidates. Report all score shares, official first places, shooting, damage, collisions, skips and game variation. No favorable reruns, outcome-selected seeds, hidden failures or post-outcome threshold relaxation.
4. To claim the challenger beats Diamond, require a prospectively registered fresh confirmation showing mean score share above50% and majority official first places across multiple games, with explicit validity and uncertainty limits. The exact confirmation budget and other guards are frozen before outcomes.
5. Publish raw evidence, actual endpoint models, logs, executed tooling and hashes. Update docs/improvements.md after each experiment with measured budget, results, accept/reject reasons and PR/report/raw links. A different agent reviews and merges; implementers never self-merge.

Diamond Port is an experimental Python adaptation, not the original championship-rated Java Diamond. Keep its pin, source and license notices unchanged. GitHub runners stay exactly Ubuntu26.04, Java17 build/test/JAR parity before JDK25/native Python runtime. Local XPSDESTKOP1 testing is authorized if useful, without changing a registered environment or touching the interactive server on7654.

## Transfer back to AIS

Retcon preservation is not a gate for the experimental challenger itself. Once an idea works, propose a separate AIS integration with a newly frozen source and fresh A/B against Diamond and Adaptive, absolute and relative Retcon-win guards, and sample regression checks. Challenger results cannot be reused as proof of the integrated AIS version. Keep the two identities, states, results and claims separate. Melee follows successful 1v1 work.

PR55's already-running AIS experiment finishes unchanged and remains separately documented. No new AIS optimization run or automatic gameplay promotion follows it under this new direction.
