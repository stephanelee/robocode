#!/usr/bin/env python3
"""Prepare this frozen invitational without starting any battles."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

COMMIT = '1f6e6a38582838fe891330a0b3e7a7d9f1668279'
EXPERIMENTS = ['Aegis', 'Breakwater', 'DiamondChallenger', 'Keel', 'KeelHelm', 'Meridian', 'Prism']
KNOWN_REPLAY_FAILURES = {
    ('MasterParityTest', 'masterMakesIdenticalDecisionsOnEveryGoldenTrace(Path)'),
    ('TraceReplayTest', 'goldenTracesReplayIdentically(Path)'),
    ('TraceReplayTest', 'signedZeroIsNotEqualToZero(Path)'),
    ('TraceReplayTest', 'incompleteTracesAreNotSuccessful(Path)'),
    ('TraceReplayTest', 'alteredRecordedCommandIsDetected(Path)'),
}


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def known_macos_failures(directory):
    failures = set()
    for path in directory.glob('TEST-*.xml'):
        suite = ET.parse(path).getroot()
        if int(suite.get('errors', '0')):
            return False
        for case in suite.findall('testcase'):
            failure = case.find('failure')
            if failure is not None:
                message = failure.get('message', '')
                if 'gunTurnRate' not in message or '2.086986457658' not in message:
                    return False
                failures.add((case.get('classname'), case.get('name')))
    return failures == KNOWN_REPLAY_FAILURES


def main(args):
    source = args.source.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Preparation output must be empty; retained evidence is never overwritten')
    actual = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != COMMIT:
        raise ValueError(f'Use a separate frozen source checkout at {COMMIT}; observed {actual}')
    subprocess.run(['git', '-C', str(source), 'diff', '--exit-code', 'HEAD', '--', 'src', 'AISurvivor', 'experiments', 'build.gradle', 'tools'], check=True, stdout=subprocess.DEVNULL)
    java17 = args.java17_home.absolute() / 'bin/java'
    version17 = subprocess.check_output([str(java17), '-version'], stderr=subprocess.STDOUT, text=True)
    if 'version "17.' not in version17:
        raise ValueError('The Gradle build requires JDK17')
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(source / 'tools'))
    import benchmark
    import retcon
    lock = json.loads((source / 'tools/opponents/strong-melee.lock.json').read_text())
    cache = source / '.benchmark-cache'
    assets = {name: benchmark.download(cache, name, sha) for name, sha in benchmark.ASSETS.items()}
    retcon_source, api_cache, retcon_provenance = retcon.prepare(cache)
    guest = source / 'guest-source'
    if guest.exists() or (source / 'samples').exists():
        raise ValueError('Use a fresh source checkout: guest-source and samples must be absent')
    guest.mkdir()

    def fetch(entry):
        target = guest / retcon.checked_relative(entry['path'])
        data = retcon.read_url(f"https://raw.githubusercontent.com/{lock['repository']}/{lock['commit']}/{entry['path']}", entry['size'])
        if len(data) != entry['size'] or hashlib.sha256(data).hexdigest() != entry['sha256']:
            raise ValueError('Guest source pin mismatch: ' + entry['path'])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(fetch, lock['files']))
    for path in guest.rglob('*.sh'):
        path.chmod(0o755)
    benchmark.extract(assets['sample-bots-Java-1.4.0.zip'], source / 'samples')
    env = os.environ.copy()
    env['JAVA_HOME'] = str(args.java17_home.absolute())
    env['PATH'] = str(java17.parent) + os.pathsep + env.get('PATH', '')
    gradle = args.gradle.absolute() if args.gradle else source / 'gradlew'
    original = output / 'AISurvivor-committed.jar'
    shutil.copy2(source / 'AISurvivor/AISurvivor.jar', original)
    builds = []
    for name in ['AISurvivor'] + EXPERIMENTS:
        project = source if name == 'AISurvivor' else source / 'experiments' / name
        log = output / (name + '-build.log')
        with log.open('w') as stream:
            result = subprocess.run([str(gradle), '-p', str(project), '--no-daemon', 'clean', 'build'], env=env, cwd=source, stdout=stream, stderr=subprocess.STDOUT)
        allowed = (name == 'AISurvivor' and args.allow_known_macos_replay_differences
                   and platform.system() == 'Darwin' and platform.machine() == 'arm64'
                   and known_macos_failures(source / 'build/test-results/test'))
        if result.returncode and not allowed:
            raise RuntimeError(f'{name} build/tests failed; retained {log}')
        committed = original if name == 'AISurvivor' else project / (name + '.jar')
        rebuilt = source / 'AISurvivor/AISurvivor.jar' if name == 'AISurvivor' else project / 'build/bot' / name / (name + '.jar')
        parity = subprocess.check_output([sys.executable, str(source / 'tools/verify_jar.py'), str(committed), str(rebuilt)], text=True)
        builds.append(dict(name=name, buildExitCode=result.returncode, knownReplayDifferencesAcknowledged=bool(result.returncode and allowed), parityExitCode=0, parity=parity, jarSHA256=digest(committed), rebuiltJarSHA256=digest(rebuilt)))
        shutil.copytree(project / 'build/test-results', output / (name + '-test-results'), dirs_exist_ok=True)
    # Only select the battle JDK after every Java17 Gradle build is finished.
    version25 = subprocess.check_output([str(args.java25.absolute()), '-version'], stderr=subprocess.STDOUT, text=True)
    if 'version "25.' not in version25:
        raise ValueError('Retcon battle execution requires JDK25')
    python = str(args.python.absolute())  # Keep the venv path; resolving its symlink loses site-packages.
    runtime = json.loads(subprocess.check_output([python, '-c', 'import sys,json,importlib.metadata as m; assert sys.version_info >= (3,10); assert m.version("robocode-tank-royale")=="1.3.1"; print(json.dumps(dict(version=sys.version,packages=sorted((d.metadata["Name"],d.version) for d in m.distributions()))))'], text=True))
    subprocess.run([python, '-m', 'pip', 'check'], check=True)
    import tournament
    bases = {name: source / 'AISurvivor' if name == 'AISurvivor' else source / 'experiments' / name / 'bot' for name in tournament.NAMES[:8]}
    bases.update({'Retcon': retcon_source, 'Diamond Port': guest / 'bots/ports/diamond-port', 'Adaptive Prime': guest / 'bots/adaptive-prime'})
    bases.update({name: source / 'samples' / name for name in tournament.NAMES[-3:]})
    tournament.save(output / 'prepared-entry-manifests.json', tournament.collect_entry_manifests(source, bases))
    tournament.save(output / 'prepared-shared-inputs.json', tournament.collect_shared_inputs(source, args.java25.absolute(), args.python.absolute()))
    provenance = dict(mainCommit=actual, java17=version17, java25=version25, python=runtime, platform=platform.platform(), machine=platform.machine(), host=platform.node(), assets=benchmark.ASSETS, retcon=retcon_provenance, pythonGuests=dict(commit=lock['commit'], sourceSHA256=lock['sourceSHA256'], opponents=lock['opponents']), builds=builds)
    (output / 'builds.json').write_text(json.dumps(builds, indent=2) + '\n')
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print('Prepared frozen entries; no games started. Evidence: ' + str(output))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--java17-home', type=Path, required=True)
    parser.add_argument('--java25', type=Path, required=True)
    parser.add_argument('--python', type=Path, required=True)
    parser.add_argument('--gradle', type=Path)
    parser.add_argument('--allow-known-macos-replay-differences', action='store_true')
    main(parser.parse_args())
