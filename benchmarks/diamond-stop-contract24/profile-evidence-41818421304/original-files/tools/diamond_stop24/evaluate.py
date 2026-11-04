"""ACQUISITION ONLY. Exactly24 frozen cards; never run during zero-case preparation."""
import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path
import struct
import subprocess
import sys
import traceback
import types

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
JAVA_INJECTION = ('CLASSPATH', 'JAVA_TOOL_OPTIONS', '_JAVA_OPTIONS', 'JDK_JAVA_OPTIONS')
PYTHON_INJECTION = ('PYTHONPATH', 'PYTHONHOME', 'PYTHONSTARTUP', 'PYTHONUSERBASE', 'PYTHONINSPECT')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')


def number(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError('Nonfinite numeric output')
    return dict(value=value, hex=value.hex(), bits=struct.pack('>d', value).hex())


def validate_number_records(value):
    if isinstance(value, dict):
        if set(value) == {'value', 'hex', 'bits'}:
            decoded = float.fromhex(value['hex'])
            if (not math.isfinite(decoded) or struct.pack('>d', decoded).hex() != value['bits']
                    or struct.pack('>d', value['value']).hex() != value['bits']):
                raise ValueError('Numeric JSON/hex/raw-bit mismatch')
        else:
            for item in value.values():
                validate_number_records(item)
    elif isinstance(value, list):
        for item in value:
            validate_number_records(item)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError('Nonfinite raw numeric value')


def append(path, value):
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def own_source_module(name, path, expected):
    path = Path(path).resolve(strict=True)
    if digest(path) != expected:
        raise ValueError('Owned source bytes differ from frozen prepared closure')
    if name in sys.modules:
        raise ValueError('Owned source module already loaded')
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = ''
    sys.modules[name] = module
    exec(compile(path.read_bytes(), str(path), 'exec', dont_inherit=True), module.__dict__)
    return module


def adapter_intermediates(case, dia):
    # Source-derived adapter normalization only; never import/construct the bot entry point.
    # The explicit frozen turn/distance pair remains the numeric reference input.
    desired = case['classicHeadingRadians'] + case['requestedClassicTurnRadians'] + (math.pi if case['distance'] < 0 else 0)
    relative = dia.normal_relative_angle(desired - case['classicHeadingRadians'])
    if abs(relative) > dia.HALF_PI:
        actual_distance = -100.0
        actual_turn = -(math.degrees(math.pi + relative)) if relative < 0 else math.degrees(math.pi - relative)
    else:
        actual_distance = 100.0
        actual_turn = math.degrees(-relative) if relative < 0 else -math.degrees(relative)
    return dict(desiredTravelAngleRadians=number(desired), normalizedRelativeRadians=number(relative),
                backAsFrontDistance=number(actual_distance), backAsFrontRequestedTankTurnDegrees=number(actual_turn),
                explicitRequestedTankTurnDegrees=number(-math.degrees(case['requestedClassicTurnRadians'])))


def reduction(rows):
    def group(selected):
        names = ('x', 'y', 'speed', 'headingDegrees')
        return dict(count=len(selected), ids=[r['id'] for r in selected],
            agreements=sum(r['agreement'] for r in selected),
            discrepancies=[r['id'] for r in selected if not r['agreement']],
            maximumAbsoluteError={key: max((r['errors'][key] for r in selected), default=None) for key in names},
            timeMismatches=[r['id'] for r in selected if not r['timeAgrees']],
            referenceWallCases=[r['id'] for r in selected if r['reference']['firstPassConstrained'] or r['reference']['secondPassConstrained']])
    return dict(schemaVersion=1, status='valid', physicalGames=0,
        interpretation='One-step source composition characterization only; no competitive inference or live budget.',
        tolerance=1e-9, all=group(rows), STOP=group([row for row in rows if row['input']['command'] == 'STOP']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', required=True, type=Path)
    parser.add_argument('--locations', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--activation', required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if output != args.prepared.resolve().parent / 'acquisition':
        raise ValueError('Only the fixed prepared-root acquisition destination is permitted')
    output.mkdir(parents=True, exist_ok=False)  # spent partial roots cannot resume
    rows, bridge, phase, active, row, loader = [], None, 'validate-prepared', None, None, None
    try:
        if not (sys.flags.isolated and sys.flags.dont_write_bytecode and sys.flags.no_site):
            raise RuntimeError('Use the pinned private Python with -I -B -S')
        if any(key in os.environ for key in JAVA_INJECTION + PYTHON_INJECTION) or any(key.startswith(('PYTHON', 'LD_', 'DYLD_')) for key in os.environ):
            raise RuntimeError('Unsanitized interpreter injection environment')
        core_dir = args.prepared.resolve(strict=True)
        core = json.loads((core_dir / 'core.json').read_text())
        activation = json.loads(args.activation.read_text())
        home = core_dir.parent
        if args.activation.resolve() != home / 'activation.json':
            raise RuntimeError('Wrong host receipt path')
        if (activation.get('receiptRef') != 'refs/tags/experiment-receipts/diamond-stop-contract24-v1'
                or not re.fullmatch('[0-9a-f]{40}', activation.get('receiptObject', ''))
                or not re.fullmatch('[0-9a-f]{40}', activation.get('payloadSHA', ''))
                or activation['payloadSHA'] != os.environ.get('PAYLOAD_SHA')):
            raise RuntimeError('Missing/mismatched spent host receipt')
        identity = activation['hostIdentity']
        if (identity['runAttempt'] != 1 or os.environ.get('GITHUB_RUN_ATTEMPT') != '1'
                or identity['runID'] != os.environ.get('GITHUB_RUN_ID')
                or identity['workspace'] != str(ROOT.resolve())
                or identity['bootID'] != Path('/proc/sys/kernel/random/boot_id').read_text().strip()):
            raise RuntimeError('Wrong active host/run/workspace identity')
        if (digest(home / 'profile.json') != activation['profileSHA256']
                or digest(home / 'host.json') != activation['hostDocumentSHA256']
                or digest(ROOT / 'benchmarks/diamond-stop-contract24/freeze.json') != activation['freezeSHA256']
                or digest(ROOT / 'benchmarks/meridian-gun-plan/package-contents.json') != activation['packageSHA256']):
            raise RuntimeError('Frozen receipt/profile/package binding differs')
        profile = json.loads((home / 'profile.json').read_text())
        if profile['core']['core.json']['sha256'] != digest(core_dir / 'core.json'):
            raise RuntimeError('Core differs from independently frozen profile')
        locations = json.loads(args.locations.read_text())
        required = ('javaHome', 'serverJar', 'portRoot', 'sdkSite', 'python', 'coreDir')
        paths = {key: Path(locations[key]).absolute() for key in required}
        if paths['coreDir'].resolve() != core_dir or Path(sys.executable).absolute() != paths['python']:
            raise RuntimeError('Wrong private interpreter/core directory')
        for name, expected in core['toolSources'].items():
            if digest(HERE / name) != expected:
                raise RuntimeError('Prepared tool bytes differ: ' + name)
        pins = core['pins']
        if digest(paths['serverJar']) != pins['serverSHA256'] or digest(core_dir / 'bridge.jar') != core['bridgeSHA256']:
            raise RuntimeError('Pinned bridge/server bytes differ')
        for category, base in (('sdk', paths['sdkSite']), ('port', paths['portRoot'])):
            for relative, expected in pins[category].items():
                path = (base / relative).resolve(strict=True)
                if not path.is_relative_to(base.resolve()) or digest(path) != expected:
                    raise RuntimeError('Original staged source changed: ' + relative)
        for relative, expected in pins['engineReferenceSources'].items():
            if digest(ROOT / relative) != expected:
                raise RuntimeError('Frozen source-derived engine reference changed: ' + relative)
        case_path = ROOT / core['cases']
        if digest(case_path) != pins['casesSHA256']:
            raise RuntimeError('Frozen card bytes differ')
        case_data = json.loads(case_path.read_text())
        cards = case_data['cases']
        if len(cards) != 24 or [c['id'] for c in cards] != [f'{i:02d}' for i in range(1, 25)]:
            raise RuntimeError('Expected precisely24 ordered frozen cards')
        for case in cards:
            if any(float(case[key]).hex() != value for key, value in case['inputHex'].items()):
                raise RuntimeError('Input decimal/hex disagreement')
            if case['inputTime'] != 13 or case['expectedOutputTime'] != 14 or case['classicHeadingRadians'] != math.pi / 2 or case['tankHeadingDegrees'] != 0.0:
                raise RuntimeError('Wrong registered time/heading')
        save(output / 'inputs.json', case_data)
        save(output / 'prepared-binding.json', dict(coreSHA256=digest(core_dir / 'core.json'),
            locations=locations, locationsSHA256=digest(args.locations), evaluatorArguments=sys.argv,
            activation=activation, activationSHA256=digest(args.activation), sourcePins=pins, forbiddenPaths='No Bot/bootstrap/server/socket/entry-point construction or execution'))
        phase = 'verify-frozen-base-python-runtime'
        verifier_metadata = core['pythonRuntimeVerifier']
        if verifier_metadata['proofFilename'] != 'python-runtime.json':
            raise RuntimeError('Unexpected base-runtime proof location')
        runtime_proof_path = home / verifier_metadata['proofFilename']
        runtime_proof_sha = digest(runtime_proof_path)
        runtime_proof = json.loads(runtime_proof_path.read_text())
        if (runtime_proof_sha != identity['pythonRuntimeSHA256']
                or runtime_proof != profile['pythonRuntime']):
            raise RuntimeError('Base Python dependency proof differs from frozen profile/receipt')
        verifier_path = ROOT / verifier_metadata['source']
        if verifier_path.resolve() != (ROOT / 'tools/diamond_stop24_python_proof.py').resolve():
            raise RuntimeError('Unexpected runtime-proof verifier source')
        verifier = own_source_module('diamond_stop24_runtime_verifier', verifier_path, verifier_metadata['sha256'])
        runtime_files = verifier.verify(runtime_proof)  # metadata-only rehash, no tested import/helper
        save(output / 'python-runtime-verified.json', dict(proofPath=str(runtime_proof_path),
            proofSHA256=runtime_proof_sha, verifierSource=verifier_metadata, filesVerified=len(runtime_files),
            builtinFrozenBinding=runtime_proof['builtinFrozenBinding'], policy=runtime_proof['policy'],
            boundary='host verifies before child startup; child rehashes before any tested import'))
        phase = 'load-reviewed-closure'
        loader = own_source_module('diamond_stop24_loader', HERE / 'loader.py', core['toolSources']['loader.py'])
        own_files = {HERE / name: value for name, value in core['toolSources'].items()}
        own_files[verifier_path] = verifier_metadata['sha256']
        validator, dia, predictor, guard = loader.load(paths['sdkSite'], paths['portRoot'], pins,
            runtime_files, runtime_proof['builtinFrozenBinding'], runtime_proof_sha, own_files)
        save(output / 'imports-before-cases.json', loader.inventory(guard, own_files, pins, paths['sdkSite'], paths['portRoot']))
        bindings = {}
        for name in ('get_new_target_speed', 'validate_target_speed', 'validate_turn_rate'):
            function = getattr(validator, name)
            source = Path(function.__code__.co_filename).resolve(strict=True)
            expected_source = (paths['sdkSite'] / 'robocode_tank_royale/bot_api/internal/intent_validator.py').resolve()
            if source != expected_source:
                raise RuntimeError('Unexpected actual SDK method body origin')
            bindings['IntentValidator.' + name] = dict(module=function.__module__, path=str(source),
                sha256=digest(source), firstLine=function.__code__.co_firstlineno)
        function = predictor.predict
        source = Path(function.__code__.co_filename).resolve(strict=True)
        if source != (paths['portRoot'] / 'diamond_port/movement_predictor.py').resolve():
            raise RuntimeError('Unexpected actual predictor method origin')
        bindings['MovementPredictor.predict'] = dict(module=function.__module__, path=str(source),
            sha256=digest(source), firstLine=function.__code__.co_firstlineno)
        save(output / 'tested-bindings.json', bindings)
        phase = 'start-pinned-bridge'
        java = (paths['javaHome'] / 'bin/java').absolute()
        jar = (core_dir / 'bridge.jar').resolve(strict=True)
        server = paths['serverJar'].resolve(strict=True)
        command = [str(java), '-cp', str(jar) + os.pathsep + str(server), 'DiamondStopBridge',
                   str(server), str(jar), core['bridgeSHA256'], pins['mathClassSHA256']]
        save(output / 'commands.json', dict(java=command, python=sys.argv, outerProcessCapSeconds=300,
            capOwner='host wrapper; one cap for whole24-case process, never per card'))
        environment = os.environ.copy()
        for key in JAVA_INJECTION + PYTHON_INJECTION:
            environment.pop(key, None)
        with (output / 'java-stderr.log').open('w') as java_error:
            bridge = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=java_error, text=True, encoding='utf-8', bufsize=1, env=environment)
            header_line = bridge.stdout.readline()
            append(output / 'java-output.jsonl', json.loads(header_line))
            header = json.loads(header_line)
            if header.get('type') != 'java-provenance' or header.get('serverSHA256') != pins['serverSHA256'] or header.get('mathClassSHA256') != pins['mathClassSHA256']:
                raise RuntimeError('Missing/wrong actually loaded Java provenance')
            for case in cards:
                active = case['id']
                row = dict(id=active, input=case, inputBits={key: number(case[key]) for key in case['inputHex']})
                append(output / 'attempted.jsonl', dict(id=active, input=case))
                phase = 'original-predictor'
                # Exactly the registered direct forecast once. No extra helper smoke tests.
                original = predictor(dia.BattleField(800.0, 600.0)).predict(
                    dia.RobotState(dia.Point(case['x'], case['y']), math.pi / 2, case['speed'], 13),
                    case['distance'], case['requestedClassicTurnRadians'], case['maxVelocity'], 1, False)
                original_degrees = (90 - math.degrees(original.heading)) % 360
                original_raw_x = case['x'] + math.sin(original.heading) * original.velocity
                original_raw_y = case['y'] + math.cos(original.heading) * original.velocity
                row['original'] = dict(x=number(original.location.x), y=number(original.location.y),
                    classicHeadingRadians=number(original.heading), headingDegrees=number(original_degrees),
                    speed=number(original.velocity), time=original.time,
                    unconstrainedReconstruction=dict(x=number(original_raw_x), y=number(original_raw_y)),
                    wallConstrainedByReconstruction=original_raw_x != original.location.x or original_raw_y != original.location.y)
                phase = 'adapter-sdk-controller'
                row['adapter'] = adapter_intermediates(case, dia)
                controller = validator.get_new_target_speed(case['speed'], case['distance'], case['maxVelocity'])
                target = validator.validate_target_speed(controller, case['maxVelocity'])
                requested_turn = validator.validate_turn_rate(-math.degrees(case['requestedClassicTurnRadians']), 10.0)
                row['sdk'] = dict(controllerTargetSpeed=number(controller), finalTargetSpeed=number(target),
                                  finalRequestedTurnDegrees=number(requested_turn))
                # Persist original+SDK outputs before invoking the engine reference for this card.
                append(output / 'intermediates.jsonl', row)
                phase = 'actual-engine-reference'
                line = '\t'.join([active] + [float(v).hex() for v in (case['x'], case['y'],
                    case['tankHeadingDegrees'], case['speed'], target, requested_turn)] + [str(case['inputTime'])])
                append(output / 'java-inputs.jsonl', dict(id=active, stdinHexTSV=line))
                bridge.stdin.write(line + '\n')
                bridge.stdin.flush()
                reference_line = bridge.stdout.readline()
                reference = json.loads(reference_line)
                append(output / 'java-output.jsonl', reference)
                if reference.get('type') != 'reference' or reference.get('id') != active:
                    raise RuntimeError('Wrong Java row identity')
                validate_number_records(reference)
                row['reference'] = reference
                final = reference['final']
                errors = {key: abs(row['original'][key]['value'] - final[key]['value']) for key in ('x', 'y', 'speed')}
                errors['headingDegrees'] = abs(math.remainder(original_degrees - final['headingDegrees']['value'], 360.0))
                if any(not math.isfinite(v) for v in errors.values()):
                    raise RuntimeError('Nonfinite error')
                row.update(errors=errors, timeAgrees=original.time == final['time'] == 14)
                row['agreement'] = row['timeAgrees'] and all(v <= 1e-9 for v in errors.values())
                loader.inventory(guard, own_files, pins, paths['sdkSite'], paths['portRoot'])
                append(output / 'raw.jsonl', row)
                rows.append(row)
                if not row['timeAgrees']:
                    raise RuntimeError('Exact registered output time14 required')
            phase = 'close-complete-bridge'
            bridge.stdin.close()
            extra = bridge.stdout.read()
            status = bridge.wait()
            if status != 0 or extra:
                raise RuntimeError('Java bridge failed or emitted extra output: ' + str(status))
        if len(rows) != 24 or [r['id'] for r in rows] != [c['id'] for c in cards]:
            raise RuntimeError('Incomplete24-case result')
        save(output / 'imports-final.json', loader.inventory(guard, own_files, pins, paths['sdkSite'], paths['portRoot']))
        save(output / 'summary.json', reduction(rows))
        save(output / 'files.json', {p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file()})
    except BaseException:
        failure = dict(schemaVersion=1, status='invalid', phase=phase, attemptedID=active,
            completeRows=len(rows), completeIDs=[r['id'] for r in rows], physicalGames=0,
            traceback=traceback.format_exc(), receipt='Spent by wrapper; no resume/replacement/retry')
        if row is not None:
            failure['latestPartialRow'] = row
        if loader is not None:
            try:
                save(output / 'imports-on-failure.json', loader.capture_modules_on_failure())
            except Exception as capture_error:
                failure['importProvenanceCaptureError'] = repr(capture_error)
        save(output / 'failure.json', failure)
        if bridge is not None and bridge.poll() is None:
            bridge.terminate()
            try:
                bridge.wait(timeout=2)
            except subprocess.TimeoutExpired:
                bridge.kill()
                bridge.wait()
        raise


if __name__ == '__main__':
    main()
