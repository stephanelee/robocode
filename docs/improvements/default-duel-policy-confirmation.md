# Default TRAIN duel policy: fresh confirmation

Prospectively registered after the 48-round screen, before collecting confirmation outcomes.

## Frozen comparison

- Accepted baseline: `7104b65d03298caceb010a5bfd085bda65092d17`, JAR SHA256 `d047bf99325d2ad73aedd87eac6a4baa1677170afff0c86b948f03ea93390e1d`.
- Candidate production source: `7894af4f8c07db7274538bf8c2c82c338c44cbb8`, JAR SHA256 `0b7154de17124136866edc8eb10116ee132547cc5ba65b09acd3ffbe8cb56645`. Production Java, Java tests and bot assets remain byte-identical during confirmation. Later commits may add only this registration, runner, workflow, fixtures and evidence.
- Both arms use actual default TRAIN mode, with `AISURVIVOR_DUEL_POLICY` unset. No fixed-action stage, changed gun, DuelSurf, model architecture or private observations.
- Official Retcon 0.2.0 source commit `28981c7a469ac064194b41001cd7183ec5ecf189`, server/runner and sample release 1.4.0, Ubuntu 26.04 and Java 25 runtime; Java 17 build and tracked-JAR parity first.

## Exact collection budget

Six independent paired Retcon blocks, seeds `20261801` through `20261806`, 20 scored rounds per arm per block: **240 Retcon rounds**. Alternate previous/candidate, candidate/previous by block. Each arm/game starts a fresh process and absent independent model, with its own seeded initialization; score all 20 online-training rounds. Metrics must show no loaded checkpoint and sequential round counters 1..20; preserve final checkpoint bytes, SHA256 and header counter 20.

Fresh Target, SpinBot and Walls games use seed `20261807`, three rounds per arm per opponent: **18 sample rounds**. Alternate arm order across sample opponents. Preserve fresh model endpoints with header counter three. Total budget **258 scored rounds**. No performance early stop, replacement seeds, favorable reruns or discarded failures. Invalid collection preserves all partial evidence and requires independent review before any continuation.

Candidate telemetry must prove controller default, every eligible duel decision forced to selected/executed action zero, action counts consistent, and at least one forced decision in each game. Existing endpoint and cold-model validators apply to both arms. All decisions below use unrounded values.

## Locked retention gates

Let each block delta be candidate score share minus baseline share, calculated from each whole 20-round official game. All six blocks must complete.

1. Equal-block mean delta is at least **+5 percentage points**.
2. One-sided 95% Student-t lower confidence bound is strictly positive: `mean(delta) - 2.0150483733330233 * sample_sd(delta) / sqrt(6) > 0`, df=5. This descriptive small-block interval does not guarantee power or generalization.
3. At least four of six deltas are strictly positive; worst delta is at least -5 percentage points.
4. Total candidate wins are at least baseline wins minus three.
5. Both arms have zero AISurvivor skipped turns.
6. Candidate aggregate wall collisions per 1000 turns are at most 1.2 times baseline. If baseline is zero, candidate must also be zero.
7. Each sample game loses at most five score-share percentage points, completes all three rounds, and both arms have zero AISurvivor skipped turns.

All gates must pass. A pass supports retention of this default-policy change in the tested setting, subject to independent code/raw-evidence review and standard CI. It is not a learned-policy benefit, proof of beating every opponent, or a universal win claim. The original screen is supporting evidence only and is not pooled into this fresh decision. A failure restores accepted gameplay/JAR while retaining the candidate patch, immutable plans, complete raw results and endpoints. No outcome-dependent changes to these gates.

## Evidence

Preserve both bot bundles and hashes, exact candidate workflow head and frozen production source identity, this immutable registration, runtime provenance, all 258 metric rows, 18 official result arrays, 12 Retcon native exports, every model endpoint, partial failures, per-block/sample results, raw hashes and an executable gate report. The source snapshot and model format remain unchanged.
