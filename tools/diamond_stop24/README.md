# Diamond STOP24 core: preparation is not acquisition

Registered protocol: `docs/improvements/diamond-stop-contract24.md` and the immutable24 cards in `benchmarks/diamond-stop-contract24/cases.json` (issue99/PR100). This directory implements only that one-step source-composition characterization. No bot entry point, server, socket, game, policy fix or competitive claim is implemented.

## Interfaces

Zero-case preparation (standard-library Python only):

```text
python -I -B -S /absolute/tools/diamond_stop24/prepare.py \
  --java-home ABS --server-jar ABS --port-root ABS --sdk-site ABS --output ABS/core
```

The core output must be absent/empty except that the wrapper may already have placed its pinned `server.jar` there. This compiles with Java17, runs `javap` metadata inspection and hashes/AST-parses original source bytes. It never imports the tested SDK/port or runs the Java main/helper. It emits deterministic `bridge.jar`, `classes/`, portable `core.json`, a compile log and separate actual compile provenance. `core.json` freezes the tool/source/card/bridge/class/signature bytes without machine paths; host-level provenance binds the actual paths and runtime. Failed preparation output remains evidence, never implicitly deleted.

Acquisition only after exact-payload review/receipt, under the wrapper's ONE300-second process-group cap:

```text
private-venv/bin/python -I -B -S /absolute/tools/diamond_stop24/evaluate.py \
  --prepared ABS/core --locations ABS/locations.json --activation ABS/activation.json \
  --output ABS/acquisition
```

Locations required fields are `javaHome`, `serverJar`, `portRoot`, `sdkSite`, `python`, `coreDir` (absolute); wrapper metadata may add fields. `python` identifies the actual private interpreter path including its venv basename, not only the resolved base-binary symlink. The wrapper independently binds all those locations/runtime/package bytes, exact-head review and durable receipt. The evaluator verifies the wrapper receipt/profile/package/core/run/boot binding before tested imports and creates the previously absent fixed sibling `acquisition` output root; it never resumes one. No evaluator or bridge main may be invoked as a smoke test during preparation.

The bridge classpath is exactly its own frozen JAR plus the pinned nested server JAR. Its main prints loaded-origin/hash/signature provenance, accepts24 ordered ID/hex-TSV requests, and calls actual `MathKt.calcNewBotSpeed` and `limitTurnRate` once per card each. It invokes no server main and refuses extra/incomplete input. The source-derived old-heading translation, normalized heading and two positional constraint calls retain literal candidate-expression order. Only exact first-pass constraint sets final speed0; the second pass keeps original old position and does not update speed/heading again.

## Import boundary

The runtime loader executes the exact hashed originals of six SDK modules (BotException, constants, MathUtil, schema Color, graphics Color, IntentValidator) and two port modules (dia_utils, movement_predictor). It supplies empty namespace shells instead of executing SDK/package initializers. The only forwarded package exports are the real original Color classes; no methods/constants are replaced. Source is compiled from verified bytes directly, avoiding stale/ambient `.pyc`. The SDK graphics Color module's own declarative color initialization remains unmodified; no motion-controller invocation is made by the loader.

`-S` supplements `-I -B` to disable site/pth bootstrap. A meta-path guard rejects unreviewed SDK/port modules, network/entry-point modules and non-stdlib third-party imports. Every loaded module is inventoried with resolved source/extension path/hash (and existing stdlib cached-bytecode path/hash where applicable); built-in/frozen modules and explicit namespace shells are marked. Java injection options and Python import/injection variables must be absent. The host wrapper owns stronger exact runtime/package checks and an actual loaded-origin review is still required.

## Output and interpretation

`attempted.jsonl` is fsynced before each registered call. `intermediates.jsonl` preserves original+SDK outputs before the Java reference; `java-inputs.jsonl`/`java-output.jsonl` retain the exact wire exchange. `raw.jsonl` contains complete ordered rows, original classic output and converted Tank heading, SDK intermediate/final intent, reference unconstrained/first/second/final states, exact float hex/raw bits and errors. Adapter back-as-front normalization is recorded separately; the explicit frozen command pair remains the reference input. Original wall involvement is separately marked as a reconstruction from its returned heading/velocity, not a modified predictor trace.

`summary.json` reduces all24 and the predetermined eight STOP cards with the fixed1e-9 tolerance; discrepancies are characterization results. Invalid metadata/output time, nonfinite values or exceptions produce permanent `failure.json` with phase/attempted ID/partial row/traceback and no resumed/replacement cases. Raw outputs and failures stay intact. The wrapper must always preserve available evidence and spend the fixed receipt even after a failure. There are no wins/damage/firing/hit-rate or strength results.

## Checks deliberately NOT executed before exact-payload clearance

- No import of IntentValidator, MathUtil, SDK constants/Color or port dia_utils/MovementPredictor.
- No SDK controller/clamp invocation, original predictor construction/call or fixture/card evaluation.
- No Java bridge main, MathKt numeric invocation, hand-computed numeric sanity call or alternate reference.
- No game, server, WebSocket, Bot construction or opponent process.
- No expected numeric output generated/inspected, no tolerance tuning and no extra cases.

Only syntax compilation, actual Java17 compilation and source/ZIP/class-signature/hash metadata inspection are permitted preparation checks. Those cannot establish importer feasibility or numerical agreement; exact independent source/artifact/runtime review must clear the sole acquisition first.
