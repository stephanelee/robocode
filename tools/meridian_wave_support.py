"""Freeze or acquire the single independently cleared issue #93 zero-game diagnostic."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import sysconfig
import traceback
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REG = ROOT / 'benchmarks/meridian-wave-support'
BOT = ROOT / 'experiments/Meridian'
ARTIFACT = BOT / 'build/wave-support'
FREEZE = REG / 'freeze.json'
ATTEMPT = ROOT / 'build/meridian-wave-support-attempt'
JDK = (Path('/opt/homebrew/opt/openjdk@17') / 'bin/java').resolve().parents[1]
JAVA = JDK / 'bin/java'
JAVAC = JDK / 'bin/javac'
SOURCE = '4a02dfde940356946f35c25a4f3b02640f7bdeb8'
BASE = '809b880cdcee98d7946d05e6bcd30f623f604348'
JAR_SHA = '7667ef48974e67fe1e42ecee15005905bfe915c4c4afe7cba8e52c146ec2cafe'
JAVA_SHA = '1e82f9fb8ac162eea33c3e5b36896458f0b58785e93effc20b78ed0d32f5e9da'
JAVAC_SHA = '2868d8e548eec1a68c8caaf88e0a203992f002b47d2af260fd97326a44e054f7'
MODULES_SHA = '87215c58e1b671f093bd097b4a641b74d919091a77fc4d269e6ff7a6e065b31a'


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def write(path, value):
    with path.open('x') as fd:
        json.dump(value, fd, indent=2, allow_nan=False, sort_keys=True)
        fd.write('\n')


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def route(text):
    start = text.index('    private double route(')
    return text[start:text.index('    public void run()', start)]


def class_entries(path):
    with zipfile.ZipFile(path) as archive:
        return {name: hashlib.sha256(archive.read(name)).hexdigest()
                for name in sorted(archive.namelist()) if name.endswith('.class')}


def source_inventory():
    paths = [ROOT / 'AGENTS.md', ROOT / 'docs/improvements/meridian-wave-support.md',
             ROOT / 'docs/improvements/meridian-platform.md', BOT / 'build.gradle', BOT / 'settings.gradle',
             ROOT / 'tools/meridian_wave_support.py', ROOT / 'tools/verify_jar.py',
             ROOT / 'gradle/wrapper/gradle-wrapper.properties', ROOT / 'gradle/wrapper/gradle-wrapper.jar',
             ROOT / 'gradlew', REG / 'engine-lock.json', ARTIFACT / 'classpath.txt', BOT / 'Meridian.jar']
    paths += sorted((BOT / 'src').glob('*.java'))
    paths += sorted((BOT / 'bot').iterdir())
    paths += sorted((BOT / 'diagnostic').glob('*.java'))
    paths += sorted((BOT / 'build/classes/java/diagnostic').rglob('*.class'))
    paths += sorted((REG / 'engine-source').rglob('*'))
    return {str(path.relative_to(ROOT)): digest(path) for path in paths if path.is_file()}


def verify_source_and_engine():
    current = (BOT / 'src/Meridian.java').read_text()
    frozen = git('show', SOURCE + ':experiments/Meridian/src/Meridian.java').decode()
    baseline = git('show', BASE + ':experiments/Meridian/src/Meridian.java').decode()
    if current != frozen or route(current) != route(baseline):
        raise ValueError('Source differs from frozen held code or v3 route seam')
    for filename in ('Flight.java', 'HitAngles.java', 'CrossingGun.java'):
        if (BOT / 'src' / filename).read_bytes() != git('show', SOURCE + ':experiments/Meridian/src/' + filename):
            raise ValueError('Shared unchanged source differs: ' + filename)
    if digest(BOT / 'Meridian.jar') != JAR_SHA:
        raise ValueError('Unchanged baseline JAR differs')
    if class_entries(BOT / 'Meridian.jar') != class_entries(BOT / 'build/bot/Meridian/Meridian.jar'):
        raise ValueError('Java 17 source/committed JAR class parity failed')
    engine = json.loads((REG / 'engine-lock.json').read_text())
    if engine['repository'] != 'robocode-dev/tank-royale' or engine['commit'] != 'c8ad3a8d19a843f6258d6f6f9db7f29229963903':
        raise ValueError('Wrong pinned engine')
    for name, expected in engine['files'].items():
        path = REG / 'engine-source' / name
        raw = path.read_bytes()
        blob = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
        if digest(path) != expected['sha256'] or blob != expected['gitBlob']:
            raise ValueError('Engine oracle source differs: ' + name)
    return engine


def runtime():
    if sys.version_info[:3] != (3, 15, 1) or platform.system() != 'Darwin' or platform.release() != '27.2.0' or platform.machine() != 'arm64':
        raise ValueError('Runtime differs from prospectively registered Mac/Python profile')
    if digest(JAVA) != JAVA_SHA or digest(JAVAC) != JAVAC_SHA or digest(JDK / 'lib/modules') != MODULES_SHA:
        raise ValueError('JDK binary differs from prospectively registered exact Java 17')
    jdk_files = {str(p.relative_to(JDK)): digest(p) for folder in ('bin', 'lib', 'conf')
                 for p in sorted((JDK / folder).rglob('*')) if p.is_file()}
    jdk_files['release'] = digest(JDK / 'release')
    python_files = {str(Path(sys.executable).resolve()): digest(Path(sys.executable).resolve())}
    framework_binary = Path(sys.base_prefix) / 'Python'
    if framework_binary.is_file():
        python_files[str(framework_binary.resolve())] = digest(framework_binary)
    for name in ('_hashlib', '_json', '_posixsubprocess'):
        module = __import__(name)
        if getattr(module, '__file__', None):
            python_files[str(Path(module.__file__).resolve())] = digest(module.__file__)
    return dict(os=platform.platform(), uname=list(platform.uname()), python=sys.version,
                pythonBuild=list(platform.python_build()), pythonFiles=python_files,
                javaVersion=subprocess.check_output([str(JAVA), '-version'], stderr=subprocess.STDOUT, text=True),
                javacVersion=subprocess.check_output([str(JAVAC), '-version'], stderr=subprocess.STDOUT, text=True),
                jdkRoot=str(JDK), jdkFiles=jdk_files)


def manifest():
    engine = verify_source_and_engine()
    classpath = [str(Path(line).resolve()) for line in (ARTIFACT / 'classpath.txt').read_text().splitlines() if line]
    if not classpath or any(not Path(path).is_file() or not path.endswith('.jar') for path in classpath):
        raise ValueError('Incomplete runnable diagnostic classpath')
    agent, = [path for path in classpath if Path(path).name == 'mockito-core-5.14.2.jar']
    command = [str(JAVA), '-javaagent:' + agent, '-cp', os.pathsep.join(classpath), 'WaveSupportDiagnostic', str(ATTEMPT)]
    return dict(schemaVersion=1, issue='https://github.com/stephanelee/robocode/issues/93',
                workspace=str(ROOT.resolve()), sourceCommit=SOURCE, prospectiveBaseline=BASE,
                protocolSHA256=digest(ROOT / 'docs/improvements/meridian-wave-support.md'),
                sourceFiles=source_inventory(), engine=engine, runtime=runtime(),
                classpath=[dict(path=path, sha256=digest(path), classes=class_entries(path)) for path in classpath],
                diagnosticClasses=class_entries(ARTIFACT / 'wave-support-diagnostic.jar'),
                buildCommand=['./gradlew', '--offline', '--no-daemon', '-p', 'experiments/Meridian', 'waveSupportArtifact', 'jar'],
                sourceJarParityCommand=[str(Path(sys.executable).resolve()), 'tools/verify_jar.py',
                    'experiments/Meridian/Meridian.jar', 'experiments/Meridian/build/bot/Meridian/Meridian.jar'],
                freezeCommand=[str(Path(sys.executable).resolve()), str(ROOT / 'tools/meridian_wave_support.py'), '--freeze'],
                command=command, cwd=str(BOT / 'bot'), processTimeoutSeconds=600,
                environment={k: os.environ[k] for k in ('PATH', 'HOME', 'LANG') if k in os.environ},
                budget=dict(competitiveRounds=0, states=2941, contexts=5882, trajectories=5882,
                            positionRows=294100, maximumFutureSegmentChecks=1470100, acquisitionAttempts=1))


def main(freeze=False, clearance=None):
    if freeze:
        REG.mkdir(parents=True, exist_ok=True)
        write(FREEZE, manifest())
        print('Frozen artifact/runtime/protocol; no diagnostic cases executed. SHA256=' + digest(FREEZE))
        return
    if clearance is None:
        raise ValueError('Acquisition requires an independent clearance document')
    approval = json.loads(clearance.read_text())
    if approval.get('decision') != 'approved' or approval.get('freezeSHA256') != digest(FREEZE) or not approval.get('reviewer'):
        raise ValueError('Independent exact-payload clearance is missing or mismatched')
    ATTEMPT.mkdir(parents=True)  # One attempt; existing complete OR failed root cannot resume.
    write(ATTEMPT / 'clearance.json', approval)
    write(ATTEMPT / 'attempt.json', dict(freezeSHA256=digest(FREEZE), startedUTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                      head=git('rev-parse', 'HEAD').decode().strip()))
    try:
        frozen = json.loads(FREEZE.read_text())
        current = manifest()
        write(ATTEMPT / 'actual-manifest.json', current)
        if frozen != current:
            raise ValueError('Source, artifact, command, engine pin or full runtime changed after review')
        (ATTEMPT / 'freeze.json').write_bytes(FREEZE.read_bytes())
        with (ATTEMPT / 'process.log').open('xb') as log:
            subprocess.run(frozen['command'], cwd=frozen['cwd'], env=frozen['environment'],
                           stdout=log, stderr=subprocess.STDOUT, timeout=frozen['processTimeoutSeconds'], check=True)
        summary = json.loads((ATTEMPT / 'summary.json').read_text())
        if not summary['countsValid'] or not summary['diagnosticValid']:
            raise ValueError('Acquisition row counts or oracle conformance invalid')
    except BaseException as error:
        write(ATTEMPT / 'wrapper-failure.json', dict(error=str(error), traceback=traceback.format_exc()))
        raise
    finally:
        write(ATTEMPT / 'artifact-files.json', {str(p.relative_to(ATTEMPT)): digest(p)
             for p in sorted(ATTEMPT.rglob('*')) if p.is_file()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', action='store_true', help='Metadata/hash collection only; no matrix execution')
    parser.add_argument('--clearance', type=Path, help='Independent approved exact-freeze JSON for the sole acquisition')
    args = parser.parse_args()
    main(args.freeze, args.clearance)
