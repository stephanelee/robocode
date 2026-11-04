"""Zero-game acquisition and exact-payload/runtime verification for Peak v1."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import zipfile

import benchmark
import meridian_profile as original_profile
from meridian_gun_plan import package_contents
from meridian_peak_common import ROOT, REG, OUT, CACHE, save

BASE = '809b880cdcee98d7946d05e6bcd30f623f604348'
CONTROL_JAR = '5875565a13b5759940ccb080c0c7f237caec0bae525a68e68643a35890a1a8a9'
PACKAGE_SHA = '52b7bfcf10f8226711798f27784be64583ea4c8fc676452d27f6e16b906bf191'
PROVENANCE = OUT / 'provenance'
ASSET_NAMES = ('AISurvivor', 'experiments/Meridian', 'experiments/MeridianPeak')
TOOL_NAMES = ('benchmark.py', 'compare.py', 'retcon.py', 'retcon_stats.py', 'strong_melee_opponents.py',
              'meridian_profile.py', 'meridian_gun_plan.py', 'meridian_duels.py', 'verify_jar.py',
              'meridian_gun_plan_fixture.py')


def participant_assets():
    paths = list((ROOT / 'AISurvivor').glob('*'))
    for name in ('Meridian', 'MeridianPeak'):
        folder = ROOT / 'experiments' / name
        paths += sorted((folder / 'src').glob('*.java')) + sorted((folder / 'bot').glob('*')) + [folder / (name + '.jar')]
    paths += [ROOT / 'tools/opponents/strong-melee.lock.json', ROOT / 'tools/opponents/retcon.lock.json']
    return {str(path.relative_to(ROOT)): benchmark.digest(path) for path in paths if path.is_file()}


def closure():
    paths = list((ROOT / 'tools').glob('meridian_peak*.py')) + list((ROOT / 'tools').glob('test_meridian_peak*.py'))
    paths += [ROOT / 'tools' / name for name in TOOL_NAMES]
    paths += [ROOT / '.github/workflows/meridian-peak-v1.yml', REG / 'games.json', REG / 'MeridianPeakBattle.java',
              ROOT / 'benchmarks/meridian-aim/profile.json', ROOT / 'benchmarks/meridian-aim/profile-20261026.json',
              ROOT / 'benchmarks/meridian-gun-plan/package-contents.json', ROOT / 'build.gradle', ROOT / 'settings.gradle',
              ROOT / 'gradlew', ROOT / 'gradle/wrapper/gradle-wrapper.jar', ROOT / 'gradle/wrapper/gradle-wrapper.properties']
    paths += list((ROOT / 'src').rglob('*.java'))
    for name in ('Meridian', 'MeridianPeak'):
        folder = ROOT / 'experiments' / name
        paths += [folder / 'build.gradle', folder / 'settings.gradle'] + list((folder / 'test').rglob('*.java'))
    return {str(path.relative_to(ROOT)): benchmark.digest(path) for path in sorted(set(paths)) if path.is_file()}


def allowed_runtime():
    profiles = []
    for name in ('profile.json', 'profile-20261026.json'):
        value = json.loads((ROOT / 'benchmarks/meridian-aim' / name).read_text())
        del value['assets']
        profiles.append(value)
    return profiles


def walls_manifest(archive):
    if benchmark.digest(archive) != benchmark.ASSETS['sample-bots-Java-1.4.0.zip']:
        raise ValueError('Official sample ZIP mismatch')
    with zipfile.ZipFile(archive) as z:
        descriptors = [name for name in z.namelist() if name.endswith('/Walls/Walls.json') or name == 'Walls/Walls.json']
        if len(descriptors) != 1:
            raise ValueError('Expected one official Walls descriptor')
        descriptor = descriptors[0]
        bot_root = str(Path(descriptor).parent)
        shared_root = str(Path(bot_root).parent / 'lib')
        files = {name: hashlib.sha256(z.read(name)).hexdigest() for name in sorted(z.namelist())
                 if not name.endswith('/') and (name.startswith(bot_root + '/') or name.startswith(shared_root + '/'))}
        data = json.loads(z.read(descriptor))
        if data['name'] != 'Walls' or not data.get('version') or not files:
            raise ValueError('Wrong official Walls identity')
    return dict(archive='sample-bots-Java-1.4.0.zip', archiveSHA256=benchmark.digest(archive),
                descriptor=descriptor, botRoot=bot_root, sharedRoot=shared_root,
                name=data['name'], version=data['version'], files=files)


def collect():
    PROVENANCE.mkdir(parents=True, exist_ok=True)
    original_profile.main()  # No game; retains exactly the two registered runtime schemas.
    original = json.loads((ROOT / 'build/meridian/profile.json').read_text())
    runtime = {key: value for key, value in original.items() if key != 'assets'}
    if runtime not in allowed_runtime():
        save(PROVENANCE / 'unregistered-runtime.json', runtime)
        raise ValueError('Runtime differs from both preallowed exact Ubuntu profiles')
    package = package_contents(original)
    expected = REG.parent / 'meridian-gun-plan/package-contents.json'
    if benchmark.digest(expected) != PACKAGE_SHA or package != json.loads(expected.read_text()):
        save(PROVENANCE / 'unregistered-package-contents.json', package)
        raise ValueError('Installed native packages differ from existing frozen contents')
    save(PROVENANCE / 'package-contents.json', package)
    if benchmark.digest(PROVENANCE / 'package-contents.json') != PACKAGE_SHA:
        raise ValueError('Package manifest canonical bytes differ')
    archive = benchmark.download(CACHE, 'sample-bots-Java-1.4.0.zip', benchmark.ASSETS['sample-bots-Java-1.4.0.zip'])
    walls = walls_manifest(archive)
    save(PROVENANCE / 'walls.json', walls)
    data = dict(runtime=runtime, assets=participant_assets(), walls=walls)
    save(PROVENANCE / 'profile.json', data)
    return data


def validate_payload(profile):
    frozen = json.loads((REG / 'freeze.json').read_text())
    if (type(frozen.get('schemaVersion')) is not int or frozen['schemaVersion'] != 1
            or frozen.get('controlCommit') != BASE or frozen.get('controlJarSHA256') != CONTROL_JAR
            or frozen.get('protocolSHA256') != benchmark.digest(ROOT / 'docs/improvements/meridian-peak-v1.md')
            or frozen.get('allowedRuntime') != allowed_runtime() or frozen.get('packageSHA256') != PACKAGE_SHA):
        raise ValueError('Preregistered protocol/runtime family/package/control freeze metadata changed')
    if profile['runtime'] not in allowed_runtime() or profile['assets'] != frozen['assets'] or profile['walls'] != frozen['walls']:
        raise ValueError('Actual participant/runtime/Walls assets differ from reviewed freeze')
    if participant_assets() != frozen['assets'] or closure() != frozen['tooling']:
        raise ValueError('Candidate, source or execution tooling changed')
    if benchmark.digest(REG / 'games.json') != frozen['gamesSHA256']:
        raise ValueError('Schedule changed')
    return frozen


def host_identity(stage):
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    if not boot or not os.environ.get('GITHUB_RUN_ID') or not os.environ.get('GITHUB_RUN_ATTEMPT'):
        raise ValueError('Missing boot/run identity')
    return dict(stage=stage, bootID=boot, workspace=str(ROOT.resolve()), runID=os.environ['GITHUB_RUN_ID'],
                runAttempt=os.environ['GITHUB_RUN_ATTEMPT'], profileSHA256=benchmark.digest(PROVENANCE / 'profile.json'),
                packageSHA256=benchmark.digest(PROVENANCE / 'package-contents.json'),
                freezeSHA256=benchmark.digest(REG / 'freeze.json') if (REG / 'freeze.json').exists() else None,
                toolingSHA256=hashlib.sha256(json.dumps(closure(), sort_keys=True).encode()).hexdigest())


def check_host(stage, receipt=None):
    path = PROVENANCE / (stage + '-host.json')
    host = json.loads(path.read_text())
    if host != host_identity(stage):
        raise ValueError('Stage boot/workspace/run/profile/package/tooling binding changed')
    if receipt is not None and (receipt.get('hostIdentity') != host or receipt.get('hostDocumentSHA256') != benchmark.digest(path)):
        raise ValueError('Stage receipt host document differs')
    return host


def preflight(stage, require_frozen=False):
    profile = collect()
    if require_frozen:
        validate_payload(profile)
    host = host_identity(stage)
    path = PROVENANCE / (stage + '-host.json')
    if path.exists() and json.loads(path.read_text()) != host:
        raise ValueError('Existing host binding differs; no workspace reuse')
    if not path.exists():
        save(path, host)
    return profile


def freeze(walls):
    target = REG / 'freeze.json'
    if target.exists():
        raise FileExistsError(target)
    assets = participant_assets()
    if not (ROOT / 'experiments/MeridianPeak/MeridianPeak.jar').is_file():
        raise ValueError('Build/test and freeze the candidate JAR before source freeze')
    data = dict(schemaVersion=1, controlCommit=BASE, controlJarSHA256=CONTROL_JAR,
                protocolSHA256=benchmark.digest(ROOT / 'docs/improvements/meridian-peak-v1.md'),
                assets=assets, tooling=closure(), walls=json.loads(walls.read_text()),
                gamesSHA256=benchmark.digest(REG / 'games.json'), allowedRuntime=allowed_runtime(), packageSHA256=PACKAGE_SHA)
    with target.open('x') as fd:
        json.dump(data, fd, indent=2, sort_keys=True, allow_nan=False)
        fd.write('\n')


def stage_walls(destination, expected):
    archive = CACHE / expected['archive']
    if walls_manifest(archive) != expected:
        raise ValueError('Walls ZIP/staged source manifest differs')
    # Unmodified source and shared library layout from the pinned archive only.
    root = destination / 'official-samples'
    root.mkdir()
    with zipfile.ZipFile(archive) as z:
        for name, digest in expected['files'].items():
            target = root / name
            if not target.resolve().is_relative_to(root.resolve()):
                raise ValueError('Unsafe sample path')
            target.parent.mkdir(parents=True, exist_ok=True)
            raw = z.read(name)
            if hashlib.sha256(raw).hexdigest() != digest:
                raise ValueError('Walls staged content differs')
            target.write_bytes(raw)
            if target.suffix == '.sh':
                target.chmod(0o755)
    return root / expected['botRoot']


def prepare_walls():
    """Validate the unchanged official sample source on Java 17; never start it."""
    archive = benchmark.download(CACHE, 'sample-bots-Java-1.4.0.zip', benchmark.ASSETS['sample-bots-Java-1.4.0.zip'])
    expected = walls_manifest(archive)
    destination = OUT / 'walls-preparation'
    destination.mkdir(parents=True)
    source = stage_walls(destination, expected)
    libraries = sorted((destination / 'official-samples' / expected['sharedRoot']).rglob('*.jar'))
    output = destination / 'classes'
    output.mkdir()
    command = [str(Path(os.environ['JAVA_HOME']) / 'bin/javac'), '-cp', os.pathsep.join(str(p) for p in libraries),
               '-d', str(output), *[str(p) for p in sorted(source.rglob('*.java'))]]
    save(destination / 'compile-command.json', dict(command=command, walls=expected))
    with (destination / 'compile.log').open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=120)
    save(destination / 'walls.json', expected)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('profile', 'screen', 'confirmation'), default='profile')
    parser.add_argument('--check-frozen', action='store_true')
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--walls-manifest', type=Path)
    parser.add_argument('--prepare-walls', action='store_true')
    args = parser.parse_args()
    if args.prepare_walls:
        prepare_walls()
    elif args.freeze:
        if args.walls_manifest is None:
            parser.error('--freeze requires --walls-manifest from the zero-game preflight')
        freeze(args.walls_manifest)
    else:
        preflight(args.stage, args.check_frozen)
