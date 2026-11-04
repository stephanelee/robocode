"""Compile/inspect only: never import SDK/predictor or invoke a numeric helper."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CASE_REL = 'benchmarks/diamond-stop-contract24/cases.json'
RUNTIME_VERIFIER_REL = 'tools/diamond_stop24_python_proof.py'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('java-home', 'server-jar', 'port-root', 'sdk-site', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    pins = json.loads((HERE / 'source_pins.json').read_text())
    verifier = ROOT / RUNTIME_VERIFIER_REL
    ast.parse(verifier.read_bytes(), filename=str(verifier))  # source metadata only; never import
    verifier_metadata = dict(source=RUNTIME_VERIFIER_REL, sha256=digest(verifier), proofFilename='python-runtime.json')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(path.name != 'server.jar' for path in output.iterdir()):
        raise ValueError('Core preparation requires a fresh directory (only wrapper server.jar may exist)')
    server = args.server_jar.resolve(strict=True)
    if digest(server) != pins['serverSHA256']:
        raise ValueError('Pinned nested server JAR mismatch')
    for category, base in (('sdk', args.sdk_site), ('port', args.port_root)):
        for relative, expected in pins[category].items():
            path = (base / relative).resolve(strict=True)
            if not path.is_relative_to(base.resolve()) or digest(path) != expected:
                raise ValueError('Original source identity mismatch: ' + relative)
            ast.parse(path.read_bytes(), filename=str(path))  # syntax metadata only; never execute
    for relative, expected in pins['engineReferenceSources'].items():
        if digest(ROOT / relative) != expected:
            raise ValueError('Pinned source-derived engine reference changed: ' + relative)
    if digest(ROOT / CASE_REL) != pins['casesSHA256']:
        raise ValueError('Registered input cards changed')
    with zipfile.ZipFile(server) as jar:
        name = 'dev/robocode/tankroyale/server/rules/MathKt.class'
        if jar.namelist().count(name) != 1:
            raise ValueError('Ambiguous MathKt class entry')
        class_bytes = jar.read(name)
        if hashlib.sha256(class_bytes).hexdigest() != pins['mathClassSHA256']:
            raise ValueError('MathKt bytecode differs')
    java_home = args.java_home.resolve(strict=True)
    javac, javap = java_home / 'bin/javac', java_home / 'bin/javap'
    environment = __import__('os').environ.copy()
    for key in ('CLASSPATH', 'JAVA_TOOL_OPTIONS', '_JAVA_OPTIONS', 'JDK_JAVA_OPTIONS'):
        environment.pop(key, None)
    version = subprocess.run([str(javac), '-version'], capture_output=True, text=True,
                             check=True, env=environment)
    version_text = version.stdout + version.stderr
    if not version_text.strip().startswith('javac 17.'):
        raise ValueError('Compilation requires actual Java17')
    signatures = subprocess.run([str(javap), '-classpath', str(server),
        'dev.robocode.tankroyale.server.rules.MathKt'], check=True, capture_output=True,
        text=True, env=environment).stdout
    required = ('public static final double calcNewBotSpeed(double, double);',
                'public static final double limitTurnRate(double, double);')
    if any(signature not in signatures for signature in required):
        raise ValueError('Required true public static double signatures unavailable')
    classes = output / 'classes'
    classes.mkdir()
    command = [str(javac), '--release', '17', '-g:none', '-encoding', 'UTF-8',
               '-classpath', str(server), '-d', str(classes), str(HERE / 'DiamondStopBridge.java')]
    compiled = subprocess.run(command, capture_output=True, text=True, env=environment)
    (output / 'compile.log').write_text(compiled.stdout + compiled.stderr)
    compiled.check_returncode()
    with zipfile.ZipFile(output / 'bridge.jar', 'x', compression=zipfile.ZIP_STORED) as jar:
        def put(name, contents):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_STORED
            jar.writestr(info, contents)
        put('META-INF/MANIFEST.MF', b'Manifest-Version: 1.0\r\nMain-Class: DiamondStopBridge\r\n\r\n')
        for path in sorted(classes.rglob('*.class')):
            put(path.relative_to(classes).as_posix(), path.read_bytes())
    metadata = dict(schemaVersion=1, role='ZERO CASE COMPILE AND METADATA ONLY',
        pins=pins, cases=CASE_REL, pythonRuntimeVerifier=verifier_metadata, bridge='bridge.jar', bridgeSHA256=digest(output / 'bridge.jar'),
        bridgeClasses={p.relative_to(classes).as_posix(): digest(p) for p in sorted(classes.rglob('*.class'))},
        toolSources={p.name: digest(p) for p in sorted(HERE.iterdir()) if p.is_file()},
        signatures=list(required), engineClassMajor=int.from_bytes(class_bytes[6:8], 'big'))
    (output / 'core.json').write_text(json.dumps(metadata, sort_keys=True, indent=2) + '\n')
    # This file binds local compilation metadata; portable core.json has no machine paths.
    (output / 'compile-provenance.json').write_text(json.dumps(dict(command=command,
        javacVersion=version_text, javacSHA256=digest(javac), javapSHA256=digest(javap),
        signatures=signatures, casesExecuted=0), sort_keys=True, indent=2) + '\n')


if __name__ == '__main__':
    main()
