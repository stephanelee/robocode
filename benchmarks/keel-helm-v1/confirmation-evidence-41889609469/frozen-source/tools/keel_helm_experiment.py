"""Registered KeelHelm stages. Preparation/approval is zero-game; no retries.

Source freeze and actual same-worker runtime approval are separate gates. Only
run() can start games, after complete admission and a create-only durable receipt.
"""
import argparse
import datetime as dt
import gzip
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import stat
import statistics
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request

# Explicit reviewed local helper closure; launch with -I -B -S to bypass site/pth.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import benchmark
import retcon
import strong_melee_opponents as native

ROOT = benchmark.ROOT
REG = ROOT / 'benchmarks/keel-helm-v1'
BUILD = ROOT / 'build/keel-helm'
CACHE = ROOT / '.benchmark-cache/keel-helm'
ISSUE = 107
REPOSITORY = 'stephanelee/robocode'
RUNNER = 'robocode-tankroyale-runner-1.4.0.jar'
IMAGES = {'20261026.180.1', '20261101.186.1'}
BOT = {'baseline': ('Keel', 'KEEL_METRICS', 'KEEL_SEED'),
       'candidate': ('KeelHelm', 'KEEL_HELM_METRICS', 'KEEL_HELM_SEED')}
ENEMY = {'diamond': ('Diamond Port', '1.8.28'), 'retcon': ('Retcon', '0.2.0'),
         'walls': ('Walls', '1.0'), 'ais': ('AISurvivor', '1.0')}
EXPOSURE = ('stoppedRolloutSteps', 'headingChangedSteps', 'secondWaveEvaluations', 'executedStopNow')
APPROVAL_PREFIX = 'KEEL_HELM_RUNTIME_APPROVAL_V1'
DISPATCH_PREFIX = 'KEEL_HELM_DISPATCH_ASSIGNMENT_V1'
REPORT_PREFIX = 'KEEL_HELM_RUNTIME_REVIEW_V1'
REVOCATION_PREFIX = 'KEEL_HELM_RUNTIME_REVOCATION_V1'
APPROVAL_KEYS = {'schemaVersion', 'verdict', 'repository', 'issue', 'stage', 'headSHA',
    'runId', 'runAttempt', 'bootId', 'workspace', 'artifactId', 'artifactDigestSHA256',
    'profileSHA256', 'freezeSHA256', 'gamesSHA256', 'participantManifestSHA256',
    'opponentManifestSHA256', 'commandsSHA256', 'reviewerTask', 'dispatchAssignmentSHA256'}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def load(path):
    return json.loads(path.read_bytes())


def utc(value):
    return dt.datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp()


def now():
    return time.time()


def api(path, method='GET', value=None, deadline=None):
    token = os.environ.get('GH_TOKEN')
    if not token:
        raise ValueError('Explicit wrapper-only GH_TOKEN is required')
    request = urllib.request.Request('https://api.github.com/repos/' + REPOSITORY + '/' + path,
        data=canonical(value) if value is not None else None, method=method,
        headers={'Authorization': 'Bearer ' + token, 'Accept': 'application/vnd.github+json',
                 'X-GitHub-Api-Version': '2022-11-28', 'User-Agent': 'KeelHelm-registered-experiment',
                 'Content-Type': 'application/json'})
    timeout = 20 if deadline is None else min(20, deadline - now())
    if timeout <= 0:
        raise TimeoutError('Bounded GitHub approval deadline exceeded')
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def comments(deadline=None):
    result = []
    for page in range(1, 21):
        rows = api(f'issues/{ISSUE}/comments?per_page=100&page={page}', deadline=deadline)
        result.extend(rows)
        if len(rows) < 100:
            return result
    raise ValueError('Issue comment history exceeds bounded 2000-comment approval scan')


def prefixed(body, prefix):
    if not body.startswith(prefix + '\n'):
        return None
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate JSON key')
            result[key] = value
        return result
    value = json.loads(body[len(prefix) + 1:], object_pairs_hook=unique,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON value')))
    if not isinstance(value, dict):
        raise ValueError('Expected one JSON object after exact prefix')
    return value


def clean_env():
    # No ambient Python/Java injection, repository credentials or developer state.
    result = {k: os.environ[k] for k in ('PATH', 'LANG', 'JAVA_HOME') if k in os.environ}
    result.update(PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1', LC_ALL='C.UTF-8')
    return result


def clean_parent_environment(environment):
    # Pinned package/venv preparation also receives no repository/runtime credentials.
    return {k: environment[k] for k in ('PATH', 'HOME', 'LANG', 'JAVA_HOME', 'TMPDIR') if k in environment} | {
        'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONNOUSERSITE': '1', 'LC_ALL': 'C.UTF-8'}


def files(directory):
    result = {}
    for p in sorted(directory.rglob('*')):
        if p.is_symlink():
            result[p.relative_to(directory).as_posix()] = {'symlink': os.readlink(p),
                'sha256': benchmark.digest(p) if p.is_file() else None, 'mode': stat.S_IMODE(p.lstat().st_mode)}
        elif p.is_file():
            result[p.relative_to(directory).as_posix()] = {'sha256': benchmark.digest(p), 'mode': stat.S_IMODE(p.stat().st_mode)}
    return result


def head():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()


def stage_plan(stage):
    data = load(REG / 'games.json')
    if set(data) != {'screen', 'confirmation'}:
        raise ValueError('Unexpected population stages')
    for s, expected_count, expected_budget in (('screen', 14, 120), ('confirmation', 18, 300)):
        rows = data[s]
        if len(rows) != expected_count or sum(g['rounds'] for g in rows) != expected_budget:
            raise ValueError('Registered population budget differs')
        if len({g['id'] for g in rows}) != len(rows):
            raise ValueError('Duplicate game ID')
    return data[stage]


def source_assets():
    paths = []
    for d in ('experiments/Keel', 'experiments/KeelHelm'):
        base = ROOT / d
        paths += [p for sub in ('src', 'test', 'bot') for p in (base / sub).rglob('*') if p.is_file()]
        paths += [base / n for n in ('build.gradle', 'settings.gradle', base.name + '.jar')]
    paths += [p for folder in ('AISurvivor', 'src') for p in (ROOT / folder).rglob('*') if p.is_file()]
    paths += [ROOT / p for p in ('build.gradle', 'settings.gradle', 'gradlew', 'tools/benchmark.py',
        'tools/retcon.py', 'tools/retcon_stats.py', 'tools/strong_melee_opponents.py', 'tools/verify_jar.py',
        'tools/keel_helm_experiment.py', 'tools/keel_helm_python_proof.py', 'tools/KeelHelmBattle.java',
        'tools/test_keel_helm_experiment.py', 'tools/keel_helm_harness.md', '.github/workflows/keel-helm-v1.yml',
        'tools/opponents/strong-melee.lock.json', 'tools/opponents/retcon.lock.json',
        'docs/improvements/keel-helm-v1.md', 'benchmarks/keel-helm-v1/games.json',
        'benchmarks/keel-improvement/baseline.json')]
    paths += [p for p in (ROOT / 'gradle').rglob('*') if p.is_file()]
    return {p.relative_to(ROOT).as_posix(): benchmark.digest(p) for p in sorted(set(paths))}


def validate_baseline():
    baseline = load(ROOT / 'benchmarks/keel-improvement/baseline.json')
    for row in baseline['files'] + baseline['productionAISurvivor']:
        if benchmark.digest(ROOT / row['path']) != row['sha256']:
            raise ValueError('Immutable baseline/production file changed: ' + row['path'])


def freeze():
    target = REG / 'freeze.json'
    if target.exists():
        raise FileExistsError('Source freeze is write-once')
    validate_baseline()
    stage_plan('screen')
    save(target, dict(schemaVersion=1, protocol='keel-helm-v1', assets=source_assets(),
        allowedImages=sorted(IMAGES), javaBuild='17.0.21+9', javaBattle='25.0.5+8', python='3.13.17',
        screenRounds=120, confirmationRounds=300, approvalKeys=sorted(APPROVAL_KEYS),
        gameTimeoutSeconds=1800, stageTimeoutMinutes=340, reviewTimeoutSeconds=1800,
        minimumRemainingSeconds=2100, maxRoundMsExclusive=75))


def verify_freeze():
    frozen = load(REG / 'freeze.json')
    if frozen['assets'] != source_assets():
        raise ValueError('Source/JAR/command/criteria freeze changed')
    validate_baseline()
    return frozen


def java_identity(home, expected):
    home = Path(home).resolve(strict=True)
    release = (home / 'release').read_text()
    runtime = expected + ('-LTS' if expected.startswith('25.') else '')
    if 'IMPLEMENTOR="Eclipse Adoptium"' not in release or f'JAVA_RUNTIME_VERSION="{runtime}"' not in release:
        raise ValueError('Wrong Temurin runtime build: ' + str(home))
    return dict(home=str(home), version=subprocess.check_output([str(home / 'bin/java'), '-version'],
        stderr=subprocess.STDOUT, text=True, timeout=15), files=files(home))


def capture_build():
    # Run AFTER Java17 source builds/parity/tests, BEFORE setup-java25.
    target = ROOT / 'build/keel-helm-build.json'
    if target.exists():
        raise FileExistsError(target)
    save(target, dict(java17=java_identity(os.environ['JAVA_HOME'], '17.0.21+9'),
        buildLogSHA256=benchmark.digest(ROOT / 'build/keel-helm-build.log'),
        casesOrGames=0, gradleVersion=subprocess.check_output([str(ROOT / 'gradlew'), '--version'], cwd=ROOT, text=True, timeout=120),
        commandCWD=str(ROOT), commands=[
            ['./gradlew', '--no-daemon', 'clean', 'build'],
            ['python', 'tools/verify_jar.py', str(Path(os.environ['RUNNER_TEMP']) / 'KeelHelm-AISurvivor-committed.jar'), 'AISurvivor/AISurvivor.jar'],
            ['./gradlew', '-p', 'experiments/Keel', '--no-daemon', 'clean', 'build'],
            ['python', 'tools/verify_jar.py', 'experiments/Keel/Keel.jar', 'experiments/Keel/build/bot/Keel/Keel.jar'],
            ['./gradlew', '-p', 'experiments/KeelHelm', '--no-daemon', 'clean', 'build'],
            ['python', 'tools/verify_jar.py', 'experiments/KeelHelm/KeelHelm.jar', 'experiments/KeelHelm/build/bot/KeelHelm/KeelHelm.jar'],
            ['python', '-I', '-B', '-S', '-m', 'unittest', 'discover', '-s', 'tools', '-p', 'test_keel_helm_experiment.py', '-v']]))



def identity():
    if os.environ.get('GITHUB_REPOSITORY') != REPOSITORY or os.environ.get('GITHUB_RUN_ATTEMPT') != '1':
        raise ValueError('Hosted repository/run_attempt1 required')
    if platform.machine() != 'x86_64' or os.environ.get('ImageVersion') not in IMAGES:
        raise ValueError('Unregistered architecture/image')
    os_release = Path('/etc/os-release').read_text()
    if 'VERSION_ID="26.04"' not in os_release:
        raise ValueError('Only Ubuntu26.04 is allowed')
    if sys.version_info[:3] != (3, 13, 17):
        raise ValueError('Only CPython3.13.17 is allowed')
    return dict(repository=REPOSITORY, headSHA=head(), runId=int(os.environ['GITHUB_RUN_ID']),
        runAttempt=1, pullNumber=int(os.environ['PULL_NUMBER']), bootId=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        workspace=str(ROOT.resolve()), image=os.environ['ImageVersion'], os=os_release,
        architecture=platform.machine(), pythonVersion=sys.version,
        pythonExecutable=str(Path(sys.executable).resolve()), pythonSHA256=benchmark.digest(Path(sys.executable).resolve()))


def dispatch_assignment(stage, ident, rows=None):
    selected = []
    for row in comments() if rows is None else rows:
        obj = prefixed(row['body'], DISPATCH_PREFIX)
        if obj is None or obj.get('stage') != stage or obj.get('headSHA') != ident['headSHA']:
            continue
        required = {'schemaVersion', 'repository', 'issue', 'stage', 'headSHA', 'reviewerTask',
            'reviewerAccount', 'implementationTasks', 'assignmentToolRecord', 'assignmentToolRecordSHA256'}
        if set(obj) != required or obj['schemaVersion'] != 1 or obj['repository'] != REPOSITORY or obj['issue'] != ISSUE:
            raise ValueError('Invalid dispatch assignment schema')
        if row['user']['login'] != 'stephanelee' or obj['reviewerAccount'] != 'stephanelee':
            raise ValueError('Unassigned dispatch/reviewer GitHub account')
        if not obj['reviewerTask'] or obj['reviewerTask'] in obj['implementationTasks']:
            raise ValueError('Self-review/implementation task cannot review its own worker')
        tool = obj['assignmentToolRecord']
        if not isinstance(tool, dict) or sha(canonical(tool)) != obj['assignmentToolRecordSHA256']:
            raise ValueError('Missing/invalid separate agent dispatch provenance')
        if tool.get('reviewerTask') != obj['reviewerTask'] or tool.get('activeBeforeDispatch') is not True:
            raise ValueError('Reviewer was not actively assigned before dispatch')
        selected.append((row, obj))
    if len(selected) != 1:
        raise ValueError('Need exactly one prospective dispatch assignment for exact head/stage')
    record = selected[0]
    run = api('actions/runs/' + str(ident['runId']))
    if utc(record[0]['created_at']) >= utc(run['created_at']):
        raise ValueError('Dispatch assignment was not published before this workflow dispatch')
    return record


def native_verify(source, python):
    lock = native.manifest()
    actual = {p.relative_to(source).as_posix(): benchmark.digest(p) for p in source.rglob('*') if p.is_file()}
    expected = {p['path']: p['sha256'] for p in lock['files']}
    if actual != expected:
        raise ValueError('Executed native opponent source inventory changed')
    for row in lock['runtime']['wheels']:
        p = source.parent / 'wheels' / row['filename']
        if benchmark.digest(p) != row['sha256']:
            raise ValueError('Pinned wheel changed')
    return dict(source=actual, venv=files(python.parent.parent), wheels=files(source.parent / 'wheels'))


def runtime_metadata(out, collect=False):
    location = load(out / 'locations.json')
    py = Path(location['python'])
    proof_file = out / 'python-runtime.json'
    proof_command = [str(py), '-I', '-B', '-S', str(ROOT / 'tools/keel_helm_python_proof.py')]
    env = clean_env()
    if collect:
        with proof_file.open('wb') as f:
            subprocess.run(proof_command + ['--collect'], stdout=f, env=env, check=True, timeout=180)
    else:
        subprocess.run(proof_command + ['--verify', str(proof_file)], env=env, check=True, timeout=180)
    build = load(ROOT / 'build/keel-helm-build.json')
    if files(Path(build['java17']['home'])) != build['java17']['files']:
        raise ValueError('Java17 build toolchain mutated')
    return dict(identity=identity(), java17=build['java17'], java25=java_identity(location['java25'], '25.0.5+8'),
        pythonRuntime=load(proof_file), pythonRuntimeSHA256=benchmark.digest(proof_file),
        native=native_verify(Path(location['diamondSource']), py),
        retcon=dict(source=files(Path(location['retconSource'])), api=files(Path(location['retconAPI']))),
        runner=benchmark.digest(Path(location['runner'])), samples=files(Path(location['samples'])),
        tools={n: benchmark.digest(Path(shutil.which(n)).resolve()) for n in ('git', 'timeout')},
        sourceAssets=source_assets(), sourceModes={p: stat.S_IMODE((ROOT / p).stat().st_mode) for p in source_assets()}, buildLogSHA256=benchmark.digest(ROOT / 'build/keel-helm-build.log'))


def prepare_game(game, out, locations):
    name, marker, seedvar = BOT[game['bot']]
    data = out / 'games' / game['id']
    work = out / 'work' / game['id']
    data.mkdir(parents=True)
    work.mkdir(parents=True)
    home = work / 'empty-home'
    home.mkdir()
    own = work / name
    bundle = ROOT / 'experiments' / name
    shutil.copytree(bundle / 'bot', own)
    shutil.copy2(bundle / (name + '.jar'), own / (name + '.jar'))
    descriptor = load(own / (name + '.json'))
    if descriptor['name'] != name or descriptor['version'] != '0.1.0':
        raise ValueError('Participant descriptor differs')
    env = clean_env()
    env.update(HOME=str(home), JAVA_HOME=locations['java25'], PATH=locations['java25'] + '/bin:/usr/bin:/bin',
        **{marker: str(data / 'markers.txt'), seedvar: str(game['seed'])})
    enemy = game['opponent']
    if enemy == 'diamond':
        source = Path(locations['diamondSource'])
        env.update(native.stage(source, work, Path(locations['python'])))
        other = work / 'opponents/diamond-port'
        benchmark.isolate_opponent(source / 'bots/ports/diamond-port', other)
        native.verify_staged(work, ['diamond-port'])
    elif enemy == 'retcon':
        other = work / 'Retcon'
        benchmark.isolate_opponent(Path(locations['retconSource']), other)
        api_cache = work / 'retcon-api'
        shutil.copytree(Path(locations['retconAPI']), api_cache)
        env['RUMBLE_JAVA_CACHE'] = str(api_cache)
    elif enemy == 'walls':
        choices = list(Path(locations['samples']).rglob('Walls.json'))
        if len(choices) != 1:
            raise ValueError('Expected one pinned Walls descriptor')
        other = work / 'Walls'
        benchmark.isolate_opponent(choices[0].parent, other)
    elif enemy == 'ais':
        other = work / 'AISurvivor'
        shutil.copytree(ROOT / 'AISurvivor', other)
        env.update(AISURVIVOR_MODEL_PATH=str(data / 'ais-model.dat'), AISURVIVOR_METRICS_PATH=str(data / 'ais.jsonl'),
            AISURVIVOR_SEED=str(game['seed']), AISURVIVOR_MODE='train')
    else:
        raise ValueError('Unregistered opponent')
    enemy_configs = list(other.glob('*.json'))
    if len(enemy_configs) != 1:
        raise ValueError('Ambiguous opponent descriptor')
    config = load(enemy_configs[0])
    expected_name, expected_version = ENEMY[enemy]
    if config['name'] != expected_name:
        raise ValueError('Opponent identity differs')
    # Descriptor versions are pinned by source/sample/JAR hashes, not an invented sample version.
    version = config['version']
    if enemy in ('diamond', 'retcon') and version != expected_version:
        raise ValueError('Opponent version differs')
    command = [str(Path(locations['java25']) / 'bin/java'), '-cp', locations['runner'],
        str(ROOT / 'tools/KeelHelmBattle.java'), str(game['port']), str(game['rounds']), str(data), str(own), str(other)]
    inputs = {str(p.relative_to(ROOT)): dict(sha256=benchmark.digest(p), mode=stat.S_IMODE(p.stat().st_mode)) for p in work.rglob('*') if p.is_file()}
    if list(home.iterdir()) or (data / 'ais-model.dat').exists():
        raise ValueError('Cold process/home/model state is not empty')
    result = dict(game=game, name=name, version=descriptor['version'], opponentName=config['name'],
        opponentVersion=version, command=command, cwd=str(work), environment=env, inputFiles=inputs,
        ownWritableHome=str(home), persistentModelBefore='N/A: both Keel arms learn only in memory',
        candidateExposureOnly=game['bot'] == 'candidate', modelFilesBefore=[])
    save(data / 'prepared-game.json', result)
    return result


def prepare(stage):
    verify_freeze()
    out = BUILD / stage
    if out.exists():
        raise FileExistsError('Stage output must be previously absent; no retry/resume')
    out.mkdir(parents=True)
    ident = identity()
    raw_assignment, assignment = dispatch_assignment(stage, ident)
    save(out / 'dispatch-assignment-comment.json', raw_assignment)
    save(out / 'dispatch-assignment.json', assignment)
    if stage == 'confirmation':
        verify_confirmation_gate(out)
    # Collection is once. No prepare/repair call is made after approval.
    ambient = dict(os.environ)
    try:
        os.environ.clear(); os.environ.update(clean_parent_environment(ambient))
        source, python, py = native.prepare(CACHE, sys.executable)
    finally:
        os.environ.clear(); os.environ.update(ambient)
    ret_source, ret_api, rc = retcon.prepare(CACHE)
    runner = benchmark.download(CACHE, RUNNER, benchmark.ASSETS[RUNNER])
    samples_zip = benchmark.download(CACHE, 'sample-bots-Java-1.4.0.zip', benchmark.ASSETS['sample-bots-Java-1.4.0.zip'])
    samples = CACHE / 'samples'
    if samples.exists():
        raise FileExistsError('Fresh sample staging required')
    benchmark.extract(samples_zip, samples)
    location = dict(diamondSource=str(source), python=str(python), retconSource=str(ret_source),
        retconAPI=str(ret_api), runner=str(runner), samples=str(samples), java25=str(Path(os.environ['JAVA_HOME']).resolve()))
    save(out / 'locations.json', location)
    metadata = runtime_metadata(out, collect=True)
    plans = [prepare_game(g, out, location) for g in stage_plan(stage)]
    participants = {arm: dict(name=BOT[arm][0], version='0.1.0', jarSHA256=benchmark.digest(ROOT / 'experiments' / BOT[arm][0] / (BOT[arm][0] + '.jar')),
        source=files(ROOT / 'experiments' / BOT[arm][0] / 'src'), modelHashes='N/A: in-memory only') for arm in BOT}
    opponents = dict(diamond=py, retcon=rc, samplesArchiveSHA256=benchmark.digest(samples_zip),
        sampleFiles=files(samples), productionAISurvivor=load(ROOT / 'benchmarks/keel-improvement/baseline.json')['productionAISurvivor'])
    commands = [dict(id=p['game']['id'], preparedGameFile=str((out / 'games' / p['game']['id'] / 'prepared-game.json').relative_to(ROOT)),
        preparedGameSHA256=benchmark.digest(out / 'games' / p['game']['id'] / 'prepared-game.json'),
        preparedGame=p) for p in plans]
    for filename, data in [('participant-manifest.json', participants), ('opponent-manifest.json', opponents),
                           ('commands.json', commands), ('profile.json', metadata)]:
        save(out / filename, data)
    # Archive all reviewed source/JAR bytes and exact build log; actual runtime files are hashed by profile.
    provenance = out / 'provenance'
    provenance.mkdir()
    for rel in verify_freeze()['assets']:
        dst = provenance / 'source' / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, dst)
    for name in ('profile.json', 'participant-manifest.json', 'opponent-manifest.json', 'commands.json',
                 'python-runtime.json', 'dispatch-assignment-comment.json', 'dispatch-assignment.json', 'locations.json'):
        shutil.copy2(out / name, provenance / name)
    shutil.copy2(REG / 'freeze.json', provenance / 'freeze.json')
    shutil.copy2(ROOT / 'build/keel-helm-build.log', provenance / 'build.log')
    shutil.copy2(ROOT / 'build/keel-helm-build.json', provenance / 'build.json')
    save(provenance / 'manifest.json', files(provenance))
    save(out / 'state.json', dict(stage=stage, status='awaiting finalized provenance artifact', gamesStarted=0))


def expected_approval(stage, out, artifact):
    ident = load(out / 'profile.json')['identity']
    assignment = load(out / 'dispatch-assignment.json')
    return dict(schemaVersion=1, verdict='CLEAR', repository=REPOSITORY, issue=ISSUE, stage=stage,
        headSHA=ident['headSHA'], runId=ident['runId'], runAttempt=1, bootId=ident['bootId'], workspace=ident['workspace'],
        artifactId=artifact['id'], artifactDigestSHA256=artifact['digest'].removeprefix('sha256:'),
        profileSHA256=benchmark.digest(out / 'profile.json'), freezeSHA256=benchmark.digest(REG / 'freeze.json'),
        gamesSHA256=benchmark.digest(REG / 'games.json'), participantManifestSHA256=benchmark.digest(out / 'participant-manifest.json'),
        opponentManifestSHA256=benchmark.digest(out / 'opponent-manifest.json'), commandsSHA256=benchmark.digest(out / 'commands.json'),
        reviewerTask=assignment['reviewerTask'],
        dispatchAssignmentSHA256=sha(load(out / 'dispatch-assignment-comment.json')['body'].encode()))


def validate_approval(row, expected, account, finalized):
    obj = prefixed(row['body'], APPROVAL_PREFIX)
    if obj is None:
        return False
    if set(obj) != APPROVAL_KEYS or obj != expected:
        return False
    if row['user']['login'] != account or utc(row['created_at']) <= finalized:
        return False
    # Reject bool/int coercion and malformed full SHA/digest fields despite Python equality.
    for key in ('schemaVersion', 'issue', 'runId', 'runAttempt', 'artifactId'):
        if type(obj[key]) is not int:
            return False
    for key in ('headSHA',):
        if not re.fullmatch('[0-9a-f]{40}', obj[key]):
            return False
    for key in APPROVAL_KEYS:
        if key.endswith('SHA256') and not re.fullmatch('[0-9a-f]{64}', obj[key]):
            return False
    return True


def artifact_identity(artifact_id, digest, ident):
    a = api('actions/artifacts/' + str(artifact_id))
    if a['id'] != artifact_id or a.get('expired') or a.get('digest') != 'sha256:' + digest:
        raise ValueError('Finalized artifact identity/digest differs')
    if a.get('workflow_run', {}).get('id') != ident['runId'] or a['workflow_run'].get('head_sha') != ident['headSHA']:
        raise ValueError('Artifact belongs to another run/head')
    return a


def review_report(rows, approval, expected, account, finalized):
    result = []
    for row in rows:
        obj = prefixed(row['body'], REPORT_PREFIX)
        if obj is None:
            continue
        if row['user']['login'] != account or utc(row['created_at']) <= finalized or row['id'] >= approval['id']:
            continue
        if (obj.get('approvalPayloadSHA256') == sha(canonical(expected))
                and obj.get('reviewerTask') == expected['reviewerTask']
                and isinstance(obj.get('reviewReport'), str) and obj['reviewReport'].strip()
                and isinstance(obj.get('agentToolProvenance'), dict)
                and obj['agentToolProvenance'].get('reviewerTask') == expected['reviewerTask']
                and obj['agentToolProvenance'].get('reportGeneratedBeforeGames') is True):
            result.append(row)
    if len(result) != 1:
        raise ValueError('Missing/ambiguous independent reasoned report and agent tool provenance')
    return result[0]


def comment_pin(row):
    """Stable authority fields; mutable avatar/reaction metadata is not authority."""
    required = ('id', 'node_id', 'url', 'issue_url', 'html_url', 'created_at', 'updated_at', 'body')
    if any(name not in row for name in required) or any(name not in row.get('user', {}) for name in ('id', 'login', 'type')):
        raise ValueError('Missing provider comment authority identity')
    return {name: row[name] for name in required} | dict(
        author={name: row['user'][name] for name in ('id', 'login', 'type')},
        bodySHA256=sha(row['body'].encode()))


def artifact_pin(row):
    required = ('id', 'name', 'digest', 'size_in_bytes', 'created_at', 'updated_at', 'expired')
    if any(name not in row for name in required):
        raise ValueError('Missing finalized artifact authority identity')
    run = row.get('workflow_run', {})
    if any(name not in run for name in ('id', 'head_sha')):
        raise ValueError('Missing finalized artifact run identity')
    return {name: row[name] for name in required} | dict(workflow_run={name: run[name] for name in ('id', 'head_sha')})


def accepted_authority(out, artifact, expected):
    names = ('approval-comment.json', 'review-report-comment.json', 'dispatch-assignment-comment.json')
    return dict(schemaVersion=1, approvalPayload=expected, approvalPayloadSHA256=sha(canonical(expected)),
        comments={name: comment_pin(load(out / name)) for name in names},
        originalAcceptedCopiesSHA256={name: benchmark.digest(out / name) for name in names},
        artifact=artifact_pin(artifact))


def approved_game(out, game_id):
    """The complete executable record and its exact stored bytes are approved inputs."""
    if (out / 'accepted-authority.json').exists():
        authority = load(out / 'accepted-authority.json')
        original_comment = load(out / 'approval-comment.json')
        if comment_pin(original_comment) != authority['comments']['approval-comment.json']:
            raise ValueError('Original API approval copy/authority chain changed')
        original_payload = prefixed(original_comment['body'], APPROVAL_PREFIX)
        if original_payload != authority['approvalPayload'] or benchmark.digest(out / 'commands.json') != original_payload['commandsSHA256']:
            raise ValueError('Current commands manifest is not bound to original accepted API approval')
    bindings = [row for row in load(out / 'commands.json') if row['id'] == game_id]
    if len(bindings) != 1:
        raise ValueError('Missing/ambiguous approved per-game command record')
    binding = bindings[0]
    path = (ROOT / binding['preparedGameFile']).resolve(strict=True)
    required = (out / 'games' / game_id / 'prepared-game.json').resolve(strict=True)
    if path != required or not path.is_relative_to(ROOT):
        raise ValueError('Approved per-game metadata path changed')
    if benchmark.digest(path) != binding['preparedGameSHA256'] or load(path) != binding['preparedGame']:
        raise ValueError('Full approved per-game command/environment/identity metadata changed')
    if binding['preparedGame']['game']['id'] != game_id:
        raise ValueError('Approved command belongs to another game')
    return binding['preparedGame']


def revocation(rows, expected, approval, account):
    """A matching authorized revocation only removes authority; it never grants it."""
    required = {'schemaVersion', 'verdict', 'repository', 'issue', 'stage', 'headSHA', 'runId',
        'runAttempt', 'bootId', 'artifactId', 'reviewerTask', 'approvalCommentId', 'reason'}
    for row in rows:
        if row.get('user', {}).get('login') != account:
            continue
        obj = prefixed(row['body'], REVOCATION_PREFIX)
        if obj is None:
            continue
        # Wrong-run revocations do not govern this worker; malformed targeted ones fail closed.
        if obj.get('approvalCommentId') != approval['id']:
            continue
        if set(obj) != required or obj.get('schemaVersion') != 1 or obj.get('verdict') != 'REVOKE':
            raise ValueError('Malformed targeted runtime revocation')
        if any(obj.get(name) != expected[name] for name in
                ('repository', 'issue', 'stage', 'headSHA', 'runId', 'runAttempt', 'bootId', 'artifactId', 'reviewerTask')):
            raise ValueError('Targeted runtime revocation has inconsistent identity')
        if not isinstance(obj['reason'], str) or not obj['reason'].strip() or utc(row['created_at']) <= utc(approval['created_at']):
            raise ValueError('Invalid targeted runtime revocation provenance')
        raise ValueError('Independent runtime approval was explicitly revoked')


def revalidate_authority(stage, out, phase):
    """Fetch pinned live authority before receipt and each child; never redefine originals."""
    evidence = out / 'approval-revalidations' / (phase + '.json')
    if evidence.exists():
        raise FileExistsError('Authority revalidation cannot replace prior phase evidence')
    state = dict(schemaVersion=1, stage=stage, phase=phase,
        checkedAtUTC=dt.datetime.now(dt.timezone.utc).isoformat(), currentComments={}, status='checking')
    save(evidence, state)
    try:
        ident = load(out / 'profile.json')['identity']
        original_artifact = load(out / 'artifact.json')
        original_approval = load(out / 'approval.json')
        authority = load(out / 'accepted-authority.json')
        if authority != accepted_authority(out, original_artifact, original_approval):
            raise ValueError('Original accepted authority/copy hash chain changed')
        expected = expected_approval(stage, out, original_artifact)
        if expected != original_approval or sha(canonical(expected)) != authority['approvalPayloadSHA256']:
            raise ValueError('Current profile/participant/opponent/commands/freeze/population anchor changed')
        if identity() != ident:
            raise ValueError('Approved worker identity changed')
        current_artifact = artifact_identity(original_artifact['id'], original_artifact['digest'].removeprefix('sha256:'), ident)
        state['currentArtifact'] = current_artifact
        save(evidence, state)
        if artifact_pin(current_artifact) != authority['artifact']:
            raise ValueError('Finalized provenance artifact was mutated')
        current = {}
        for name, pin in authority['comments'].items():
            row = api('issues/comments/' + str(pin['id']))
            current[name] = row
            state['currentComments'][name] = row
            save(evidence, state)
            if comment_pin(row) != pin:
                raise ValueError('Accepted comment deleted/edited/author/timestamp/body changed: ' + name)
        finalized = utc(original_artifact['updated_at'])
        assignment = load(out / 'dispatch-assignment.json')
        approval = current['approval-comment.json']
        if not validate_approval(approval, expected, assignment['reviewerAccount'], finalized):
            raise ValueError('Current accepted approval no longer clears the exact worker')
        rows = comments()
        state['scannedComments'] = rows
        save(evidence, state)
        selected = [row for row in rows if validate_approval(row, expected, assignment['reviewerAccount'], finalized)]
        if len(selected) != 1 or comment_pin(selected[0]) != authority['comments']['approval-comment.json']:
            raise ValueError('Current runtime approval is missing/ambiguous/replaced')
        current_report = review_report(rows, approval, expected, assignment['reviewerAccount'], finalized)
        if comment_pin(current_report) != authority['comments']['review-report-comment.json']:
            raise ValueError('Current independent report/provenance is ambiguous/replaced')
        raw_assignment, current_assignment = dispatch_assignment(stage, ident, rows=rows)
        if (comment_pin(raw_assignment) != authority['comments']['dispatch-assignment-comment.json']
                or current_assignment != assignment):
            raise ValueError('Current independent dispatch assignment changed')
        revocation(rows, expected, approval, assignment['reviewerAccount'])
        pr = api('pulls/' + str(ident['pullNumber']))
        state['currentPullRequest'] = pr
        save(evidence, state)
        if (pr['state'] != 'open' or pr['head']['sha'] != expected['headSHA']
                or ('KEEL-HELM-SOURCE-CLEARED ' + expected['headSHA']) not in pr['body']):
            raise ValueError('Current exact PR head/source clearance changed')
        # Once consumed, the immutable remote annotated receipt is the authority anchor.
        if (out / 'receipt.json').exists():
            receipt = load(out / 'receipt.json')
            ref = api('git/ref/tags/experiment-receipts/keel-helm-v1-' + stage)
            state['currentReceiptRef'] = ref
            if ref['object']['type'] != 'tag' or ref['object']['sha'] != receipt['annotatedTag']['sha']:
                raise ValueError('Permanent receipt identity changed')
            tag = api('git/tags/' + ref['object']['sha'])
            state['currentReceiptTag'] = tag
            save(evidence, state)
            payload = json.loads(tag['message'])
            if (payload != receipt['payload'] or payload['acceptedAuthority'] != authority
                    or payload['acceptedAuthoritySHA256'] != sha(canonical(authority))
                    or payload['approval'] != original_approval):
                raise ValueError('Original receipt-bound accepted authority changed')
        for binding in load(out / 'commands.json'):
            approved_game(out, binding['id'])
        state.update(status='valid', acceptedAuthoritySHA256=sha(canonical(authority)))
        save(evidence, state)
        return authority
    except Exception as exc:
        state.update(status='invalid', error=type(exc).__name__ + ': ' + str(exc), traceback=traceback.format_exc())
        save(evidence, state)
        raise


def wait_approval(stage, artifact_id, digest):
    out = BUILD / stage
    ident = load(out / 'profile.json')['identity']
    artifact = artifact_identity(artifact_id, digest, ident)
    # upload-artifact returns only after finalization. Use provider updated_at as conservative finalized timestamp.
    finalized = utc(artifact['updated_at'])
    deadline = finalized + 1800
    save(out / 'artifact.json', artifact)
    expected = expected_approval(stage, out, artifact)
    if set(expected) != APPROVAL_KEYS:
        raise ValueError('Internal strict approval schema differs')
    save(out / 'expected-approval.json', expected)
    assignment = load(out / 'dispatch-assignment.json')
    while now() < deadline:
        rows = comments(deadline=deadline)
        valid = [r for r in rows if validate_approval(r, expected, assignment['reviewerAccount'], finalized)]
        if len(valid) > 1:
            raise ValueError('Duplicate runtime approvals')
        if valid:
            report = review_report(rows, valid[0], expected, assignment['reviewerAccount'], finalized)
            artifact_identity(artifact_id, digest, ident)
            save(out / 'approval-comment.json', valid[0])
            save(out / 'review-report-comment.json', report)
            save(out / 'approval.json', expected)
            save(out / 'accepted-authority.json', accepted_authority(out, artifact, expected))
            return
        time.sleep(min(10, max(0, deadline - now())))
    raise TimeoutError('No exact independent runtime approval within30minutes;0games')


def parse_markers(lines, rounds, candidate, complete=True):
    starts, ends, stats, skips = [], [], {}, []
    for line in lines:
        if not line:
            continue
        m = re.fullmatch(r'(\d+) (.*)', line)
        if not m:
            raise ValueError('Unparseable private marker')
        r, payload = int(m[1]), m[2]
        if not 1 <= r <= rounds:
            raise ValueError('Out-of-population private round')
        if payload == 'start':
            if r != len(starts) + 1 or (r > 1 and r - 1 not in ends):
                raise ValueError('Out-of-order start marker')
            if r in starts:
                raise ValueError('Duplicate start marker')
            starts.append(r)
        elif payload == 'end':
            if r in ends or r not in stats:
                raise ValueError('Duplicate end or missing timing record')
            ends.append(r)
        elif payload.startswith('skip '):
            if not re.fullmatch(r'skip \d+', payload):
                raise ValueError('Unparseable skip marker')
            skips.append((r, payload))
            raise ValueError('Own skipped turn; timing guard rejects whole stage')
        elif payload.startswith('stats '):
            if r in stats or r not in starts or r in ends:
                raise ValueError('Duplicate/out-of-order stats/timing record')
            tokens = payload.split()[1:]
            fields = {}
            for token in tokens:
                if token.count('=') != 1:
                    raise ValueError('Unparseable stats record')
                key, value = token.split('=')
                if key in fields:
                    raise ValueError('Duplicate stats field')
                fields[key] = value
            if 'maxMs' not in fields or not re.fullmatch(r'\d+', fields['maxMs']):
                raise ValueError('Missing/unparseable integer maxMs')
            max_ms = int(fields['maxMs'])
            if max_ms >= 75:
                raise ValueError('Own round maxMs>=75; timing guard rejects whole stage')
            exposure = None
            if candidate:
                if not set(EXPOSURE) <= fields.keys() or any(not re.fullmatch(r'\d+', fields[k]) for k in EXPOSURE):
                    raise ValueError('Missing/unparseable candidate exposure counters')
                exposure = {k: int(fields[k]) for k in EXPOSURE}
            elif set(EXPOSURE) & fields.keys():
                raise ValueError('Baseline exposure must remain N/A and byte-identical')
            stats[r] = dict(maxMs=max_ms, exposure=exposure, rawFields=fields)
        else:
            raise ValueError('Unknown private marker record')
    if complete:
        expected = list(range(1, rounds + 1))
        if starts != expected or ends != expected or sorted(stats) != expected:
            raise ValueError('Missing private start/end/stats record')
    return stats


def partial_observation(data):
    paths = list(data.glob('*.battle.gz'))
    seen, completed, ticks, error = set(), set(), {}, None
    for path in paths:
        try:
            with gzip.open(path, 'rt') as stream:
                for line in stream:
                    e = json.loads(line)
                    r = e.get('roundNumber')
                    if e.get('type') == 'TickEventForObserver':
                        seen.add(r)
                        ticks[r] = max(ticks.get(r, 0), e['turnNumber'])
                    elif e.get('type') == 'RoundEndedEventForObserver':
                        seen.add(r)
                        completed.add(r)
        except Exception as exc:
            error = type(exc).__name__ + ': ' + str(exc)
    return dict(observedPhysicalRoundIDs=sorted(seen), completedPhysicalRoundIDs=sorted(completed),
        observedPhysicalRounds=len(seen), completedPhysicalRounds=len(completed), lastTickByRound=ticks,
        partialRecordingError=error)


def recording_metrics(data, game, prepared):
    recordings = list(data.glob('*.battle.gz'))
    if len(recordings) != 1:
        raise ValueError('Expected exactly one raw recording')
    first, final, endpoints, turn_sequences = None, None, [], {}
    own_id = None
    pending = []
    counters = dict(shots=0, hits=0, shotEnergy=0.0, damageDealt=0.0, enemyShots=0, enemyHits=0,
        damageTaken=0.0, enemyShotEnergy=0.0, wallContacts=0, botContacts=0, bulletCollisions=0)
    per_round, power_mix, enemy_power_mix = {}, {}, {}
    # Official tick counters are reduced only after discovering immutable participant IDs.
    def tick_metrics(tick):
        r = tick['roundNumber']
        seq = turn_sequences.setdefault(r, [])
        seq.append(tick['turnNumber'])
        row = per_round.setdefault(r, {k: 0 for k in counters})
        for e in tick['events']:
            kind, bullet = e['type'], e.get('bullet', {})
            if kind in ('BulletFiredEvent', 'BulletHitBotEvent', 'BulletHitBulletEvent') and bullet.get('ownerId') not in {v['id'] for v in first['participants']}:
                raise ValueError('Unregistered official bullet owner')
            if kind == 'BulletFiredEvent':
                own = bullet['ownerId'] == own_id
                key = 'shots' if own else 'enemyShots'
                energy = 'shotEnergy' if own else 'enemyShotEnergy'
                counters[key] += 1; row[key] += 1
                counters[energy] += bullet['power']; row[energy] += bullet['power']
                mix = power_mix if own else enemy_power_mix
                value = str(bullet['power']); mix[value] = mix.get(value, 0) + 1
            elif kind == 'BulletHitBotEvent':
                if bullet['ownerId'] == own_id:
                    counters['hits'] += 1; row['hits'] += 1
                    counters['damageDealt'] += e['damage']; row['damageDealt'] += e['damage']
                elif e['victimId'] == own_id:
                    counters['enemyHits'] += 1; row['enemyHits'] += 1
                    counters['damageTaken'] += e['damage']; row['damageTaken'] += e['damage']
            elif kind == 'BulletHitBulletEvent':
                # Count an official collision involving own bullet once per event.
                bullets = [e.get(k, {}) for k in ('bullet', 'hitBullet')]
                if any(b.get('ownerId') == own_id for b in bullets):
                    counters['bulletCollisions'] += 1; row['bulletCollisions'] += 1
            elif kind == 'BotHitWallEvent' and e.get('victimId') == own_id:
                counters['wallContacts'] += 1; row['wallContacts'] += 1
            elif kind == 'BotHitBotEvent' and e.get('botId') == own_id:
                counters['botContacts'] += 1; row['botContacts'] += 1
    with gzip.open(recordings[0], 'rt') as stream:
        for line in stream:
            event = json.loads(line)
            if first is None:
                first = event
                setup = event['gameSetup']
                if setup['gameType'] != '1v1' or setup['numberOfRounds'] != game['rounds'] or setup['turnTimeout'] != 100000:
                    raise ValueError('Wrong registered game setup')
                if setup.get('arenaWidth') != 800 or setup.get('arenaHeight') != 600:
                    raise ValueError('Wrong arena')
            if event['type'] == 'GameStartedEventForObserver':
                # API recording participants may be nested botInfo or direct rows.
                bots = event.get('participants', event.get('botInfos', event.get('bots', [])))
                for bot in bots:
                    if bot.get('name') == prepared['name']:
                        own_id = bot['id']
            if event['type'] == 'TickEventForObserver':
                if own_id is None:
                    pending.append(event)
                else:
                    tick_metrics(event)
            elif event['type'] == 'RoundEndedEventForObserver':
                endpoints.append(event)
                if own_id is None:
                    for row in event['results']:
                        if row['name'] == prepared['name']:
                            own_id = row['id']
                    if own_id is not None:
                        for tick in pending:
                            tick_metrics(tick)
                        pending.clear()
            final = event
    if final is None or final['type'] != 'GameEndedEventForObserver' or final['numberOfRounds'] != game['rounds']:
        raise ValueError('Missing official complete game endpoint')
    results = final['results']
    expected_participants = sorted([(prepared['name'], prepared['version']),
        (prepared['opponentName'], prepared['opponentVersion'])])
    if sorted((row['name'], row['version']) for row in results) != expected_participants:
        raise ValueError('Official participant identity/version differs')
    own = next(row for row in results if row['name'] == prepared['name'])
    if (any(not isinstance(v.get('totalScore'), int) or v['totalScore'] < 0 for v in results)
            or sum(v['totalScore'] for v in results) <= 0):
        raise ValueError('Invalid official score denominator')
    if sorted((v['name'], v['version']) for v in first['participants']) != expected_participants:
        raise ValueError('Official starting participant identities differ')
    if own_id != own['id'] or pending:
        raise ValueError('Unresolved official participant ID')
    rounds = list(range(1, game['rounds'] + 1))
    if [e['roundNumber'] for e in endpoints] != rounds or sorted(turn_sequences) != rounds:
        raise ValueError('Missing/duplicate/out-of-population official rounds')
    previous_score, previous_first = {}, {}
    rows = []
    for endpoint in endpoints:
        r = endpoint['roundNumber'];seq = turn_sequences[r]
        if seq != list(range(1, endpoint['turnNumber'] + 1)):
            raise ValueError('Missing/duplicate official ticks')
        totals = {v['id']: v['totalScore'] for v in endpoint['results']}
        delta = {k: v - previous_score.get(k, 0) for k, v in totals.items()}
        firsts = {v['id']: v['firstPlaces'] for v in endpoint['results']}
        own_score = delta[own_id];all_score = sum(delta.values())
        if (all_score <= 0 or any(v < 0 for v in delta.values())
                or firsts[own_id] - previous_first.get(own_id, 0) not in (0, 1)):
            raise ValueError('Invalid cumulative official round scores/wins')
        rows.append(dict(round=r, turns=len(seq), ownScore=own_score, totalScore=all_score,
            share=own_score / all_score if all_score else None,
            win=firsts[own_id] - previous_first.get(own_id, 0), **per_round[r]))
        previous_score, previous_first = totals, firsts
    if sum(row['win'] for row in rows) != own['firstPlaces'] or sum(row['ownScore'] for row in rows) != own['totalScore']:
        raise ValueError('Round/game official summaries do not reconcile')
    if any(not math.isfinite(value) or value < 0 for value in counters.values()):
        raise ValueError('Nonfinite/negative official combat count')
    stats = parse_markers((data / 'markers.txt').read_text().splitlines(), game['rounds'], game['bot'] == 'candidate')
    if counters['enemyShots'] <= 0:
        raise ValueError('Undefined incoming shot/hit denominator')
    model_files = {}
    if game['opponent'] == 'ais':
        ais_path = data / 'ais.jsonl'
        ais = [json.loads(s) for s in ais_path.read_text().splitlines()]
        benchmark.validate_metrics(ais, game['rounds'], 'train', game['seed'], 0)
        if [v['round'] for v in ais] != rounds or any(v['mode'] != 'train' or v['seed'] != game['seed'] or v['checkpointLoaded'] is not False for v in ais):
            raise ValueError('Incomplete actual cold AIS TRAIN evidence')
        model = data / 'ais-model.dat'
        if not model.is_file():
            raise ValueError('Missing actual stored AIS model')
        raw = model.read_bytes()
        if len(raw) < 24 or int.from_bytes(raw[:4], 'big') != 0x41495356 or int.from_bytes(raw[4:8], 'big') != 1 or int.from_bytes(raw[20:24], 'big') != game['rounds']:
            raise ValueError('Incomplete actual AIS model file')
        model_files = {p.name: benchmark.digest(p) for p in (ais_path, model)}
    for row in rows:
        row.update(maxMs=stats[row['round']]['maxMs'], exposure=stats[row['round']]['exposure'])
    turns = sum(len(v) for v in turn_sequences.values())
    return dict(share=own['totalScore'] / sum(v['totalScore'] for v in results), ownScore=own['totalScore'],
        totalScore=sum(v['totalScore'] for v in results), wins=own['firstPlaces'],
        **counters, turns=turns, hitRate=counters['hits'] / counters['shots'] if counters['shots'] else None,
        enemyHitRate=counters['enemyHits'] / counters['enemyShots'], firingPer1000Turns=counters['shots'] * 1000 / turns,
        powerMix=power_mix, enemyPowerMix=enemy_power_mix, skips=0, maxMs=max(v['maxMs'] for v in stats.values()),
        exposure={k: sum(v['exposure'][k] for v in stats.values()) for k in EXPOSURE} if game['bot'] == 'candidate' else None,
        results=results, perRound=rows, recordingSHA256=benchmark.digest(recordings[0]),
        actualModelHashes=model_files or None, persistentKeelModelHashes=None,
        **partial_observation(data))


def decide(stage, rows, expected_ids):
    valid = [r for r in rows if r.get('status') == 'valid']
    checks = dict(allGamesValid=[r['id'] for r in valid] == expected_ids,
        zeroBothArmSkips=all(r.get('skips') == 0 for r in valid),
        bothArmTiming=all(type(r.get('maxMs')) is int and r['maxMs'] < 75 for r in valid))
    if not all(checks.values()):
        return dict(decision='invalid', checks=checks)
    by = lambda arm, opp: [r for r in valid if r['bot'] == arm and r['opponent'] == opp]
    value = lambda r: Fraction(r['ownScore'], r['totalScore']) if 'ownScore' in r else Fraction(str(r['share']))
    mean = lambda arm, opp: sum((value(r) for r in by(arm, opp)), Fraction()) / len(by(arm, opp))
    wins = lambda arm, opp: sum(r['wins'] for r in by(arm, opp))
    diamond = mean('candidate', 'diamond');delta_d = diamond - mean('baseline', 'diamond')
    retcon_share = mean('candidate', 'retcon');delta_r = retcon_share - mean('baseline', 'retcon')
    if stage == 'screen':
        checks.update(diamondAbove50=diamond > Fraction(1, 2), diamondDeltaAtLeast3pp=delta_d >= Fraction(3, 100),
            diamondWinsAtLeast16=wins('candidate', 'diamond') >= 16,
            diamondGamesAbove50=sum(r['share'] > .5 for r in by('candidate', 'diamond')) >= 2,
            retconAtLeast90=retcon_share >= Fraction(9, 10), retconDeltaAtLeastMinus3pp=delta_r >= -Fraction(3, 100),
            retconWinsAtLeast16=wins('candidate', 'retcon') >= 16,
            wallsWithin5pp=mean('candidate', 'walls') - mean('baseline', 'walls') >= -Fraction(5, 100),
            wallsWinsAtLeast4=wins('candidate', 'walls') >= 4,
            aisWithin5pp=mean('candidate', 'ais') - mean('baseline', 'ais') >= -Fraction(5, 100),
            aisWinsAtLeast4=wins('candidate', 'ais') >= 4)
        decision = 'progress; confirmation open' if all(checks.values()) else 'reject'
    else:
        checks.update(diamondAbove50=diamond > Fraction(1, 2), diamondDeltaAtLeast2pp=delta_d >= Fraction(2, 100),
            diamondWinsAbove60=wins('candidate', 'diamond') > 60,
            diamondGamesAbove50=sum(r['share'] > .5 for r in by('candidate', 'diamond')) >= 4,
            retconAtLeast90=retcon_share >= Fraction(9, 10), retconDeltaAtLeastMinus2pp=delta_r >= -Fraction(2, 100),
            retconWinsAtLeast24=wins('candidate', 'retcon') >= 24)
        decision = 'confirmed' if all(checks.values()) else 'not confirmed'
    return dict(decision=decision, checks=checks, diamondMean=float(diamond), diamondDelta=float(delta_d),
        retconMean=float(retcon_share), retconDelta=float(delta_r),
        betweenGameShareSD={arm + '/' + opp: statistics.stdev([r['share'] for r in by(arm, opp)])
            if len(by(arm, opp)) > 1 else None for arm in BOT for opp in ('diamond', 'retcon')})


def remaining_seconds():
    run = int(os.environ['GITHUB_RUN_ID'])
    jobs = api(f'actions/runs/{run}/jobs?filter=latest&per_page=100')['jobs']
    active = [j for j in jobs if j['name'] == 'registered-stage' and j['status'] == 'in_progress']
    if len(active) != 1:
        raise ValueError('Cannot establish exact active job start')
    return utc(active[0]['started_at']) + 340 * 60 - now(), jobs


def verify_worker(stage):
    out = BUILD / stage
    ident = load(out / 'profile.json')['identity']
    pr = api('pulls/' + str(ident['pullNumber']))
    if (pr['state'] != 'open' or pr['head']['sha'] != ident['headSHA']
            or ('KEEL-HELM-SOURCE-CLEARED ' + ident['headSHA']) not in pr['body']):
        raise ValueError('Current PR head/source clearance changed')
    verify_freeze()
    if runtime_metadata(out) != load(out / 'profile.json'):
        raise ValueError('Same-worker runtime/assets mutated; no repair allowed')
    for binding in load(out / 'commands.json'):
        game = approved_game(out, binding['id'])
        for name, value in game['inputFiles'].items():
            if dict(sha256=benchmark.digest(ROOT / name), mode=stat.S_IMODE((ROOT / name).stat().st_mode)) != value:
                raise ValueError('Prepared participant/opponent input mutated: ' + name)
    return out


def consume_receipt(stage, out):
    authority = revalidate_authority(stage, out, 'before-receipt')
    approval = load(out / 'approval.json')
    profile = load(out / 'profile.json')
    artifact = load(out / 'artifact.json')
    artifact_identity(artifact['id'], artifact['digest'].removeprefix('sha256:'), profile['identity'])
    expected = expected_approval(stage, out, artifact)
    if expected != approval or identity() != profile['identity']:
        raise ValueError('Approval identity differs from still-running worker')
    remain, jobs = remaining_seconds()
    save(out / 'admission-jobs.json', jobs)
    if remain < 2100:
        raise ValueError('Less than35minutes remain; no child started')
    tag_name = 'experiment-receipts/keel-helm-v1-' + stage
    tag_ref = 'refs/tags/' + tag_name
    try:
        api('git/ref/tags/' + tag_name)
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise
    else:
        raise FileExistsError('Permanent stage receipt already exists; never reset/resume')
    payload = dict(schemaVersion=1, stage=stage, approval=approval,
        acceptedAuthority=authority, acceptedAuthoritySHA256=sha(canonical(authority)),
        acceptedComment=load(out / 'approval-comment.json'), reviewReport=load(out / 'review-report-comment.json'),
        dispatchAssignment=load(out / 'dispatch-assignment-comment.json'),
        jobRemainingSecondsAtAdmission=remain, receiptRef=tag_ref)
    tag = api('git/tags', 'POST', dict(tag=tag_name, message=json.dumps(payload, sort_keys=True),
        object=approval['headSHA'], type='commit', tagger=dict(name='Stéphane Lee',
            email='30698997+stephanelee@users.noreply.github.com', date=dt.datetime.now(dt.timezone.utc).isoformat().replace('+00:00', 'Z'))))
    save(out / 'receipt-tag.json', dict(payload=payload, annotatedTag=tag))
    created = api('git/refs', 'POST', dict(ref=tag_ref, sha=tag['sha']))
    save(out / 'receipt.json', dict(payload=payload, annotatedTag=tag, createdRef=created))


def stop_child(process):
    import signal
    # start_new_session=True made this exact owned PID the process-group ID.
    # The Java root may have exited while its server/bot descendants remain.
    pgid = process.pid
    evidence = dict(ownedProcessGroup=pgid, termSent=False, killSent=False)
    def signal_group(sig):
        try:
            os.killpg(pgid, sig)
            return True
        except ProcessLookupError:
            return False
    if signal_group(signal.SIGTERM):
        evidence['termSent'] = True
        deadline = time.monotonic() + 10
        while signal_group(0) and time.monotonic() < deadline:
            time.sleep(.05)
        if signal_group(0):
            evidence['killSent'] = signal_group(signal.SIGKILL)
    # Waiting/reaping the parent is separate from terminating the owned group.
    process.wait(timeout=5)
    return evidence


def execute_game(game, out):
    data = out / 'games' / game['id']
    prepared = approved_game(out, game['id'])
    home = Path(prepared['ownWritableHome'])
    if list(home.iterdir()) or (data / 'ais-model.dat').exists() or (data / 'markers.txt').exists():
        raise ValueError('Cold writable/model state is not pristine')
    save(data / 'attempt.json', dict(game=game, headSHA=head(), command=prepared['command'], cwd=prepared['cwd'],
        environment=prepared['environment'], startUTC=dt.datetime.now(dt.timezone.utc).isoformat(),
        freshProcess=True, freshWritableDirectory=True, importedModels=[], persistentKeelModelHashes=None))
    deadline = time.monotonic() + 1800
    with (data / 'runner.log').open('w') as log:
        process = subprocess.Popen(prepared['command'], cwd=prepared['cwd'], env=prepared['environment'],
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while process.poll() is None:
                if time.monotonic() >= deadline:
                    raise TimeoutError('Registered1800second child cap')
                marker = data / 'markers.txt'
                if marker.exists():
                    text = marker.read_text()
                    # Ignore a final in-flight write fragment; complete newline records are mandatory at finish.
                    parse_markers(text[:text.rfind('\n') + 1].splitlines(), game['rounds'], game['bot'] == 'candidate', complete=False)
                time.sleep(.25)
            if process.returncode:
                raise RuntimeError('Owned child exited ' + str(process.returncode))
        finally:
            save(data / 'teardown.json', stop_child(process))
    return recording_metrics(data, game, prepared)


def verify_confirmation_gate(out):
    gate = load(REG / 'screen-evidence.json')
    required = {'schemaVersion', 'sourceFreezeSHA256', 'screenRunId', 'receiptTagSHA',
        'decisionFile', 'decisionSHA256', 'retainedEvidenceFiles', 'independentOutcomeReviewCommentId'}
    if set(gate) != required or gate['schemaVersion'] != 1:
        raise ValueError('Missing exact reviewed screen evidence gate')
    if gate['sourceFreezeSHA256'] != benchmark.digest(REG / 'freeze.json'):
        raise ValueError('Candidate/source freeze differs from passing screen')
    for name, expected in gate['retainedEvidenceFiles'].items():
        path = (ROOT / name).resolve(strict=True)
        if not path.is_relative_to(ROOT) or benchmark.digest(path) != expected:
            raise ValueError('Retained passing-screen evidence differs')
    if gate['retainedEvidenceFiles'].get(gate['decisionFile']) != gate['decisionSHA256']:
        raise ValueError('Screen decision is not in retained raw evidence manifest')
    decision = load(ROOT / gate['decisionFile'])
    expected = decide('screen', decision['games'], [g['id'] for g in stage_plan('screen')])
    if decision.get('stage') != 'screen' or decision.get('decision') != expected['decision'] or expected['decision'] != 'progress; confirmation open':
        raise ValueError('Registered screen did not pass every fixed gate')
    if any(row.get('recordingSHA256') not in gate['retainedEvidenceFiles'].values() for row in decision['games']):
        raise ValueError('Complete passing-screen recordings are not retained')
    ref = api('git/ref/tags/experiment-receipts/keel-helm-v1-screen')
    if ref['object']['sha'] != gate['receiptTagSHA'] or ref['object']['type'] != 'tag':
        raise ValueError('Screen permanent receipt differs')
    tag = api('git/tags/' + gate['receiptTagSHA'])
    receipt = json.loads(tag['message'])
    approval = receipt['approval']
    if (receipt['stage'] != 'screen' or approval['freezeSHA256'] != gate['sourceFreezeSHA256']
            or approval['runId'] != gate['screenRunId'] or approval['runAttempt'] != 1
            or approval['headSHA'] != decision['headSHA']):
        raise ValueError('Screen receipt/run/head/population binding differs')
    run = api('actions/runs/' + str(gate['screenRunId']))
    if run['status'] != 'completed' or run['conclusion'] != 'success' or run['head_sha'] != decision['headSHA'] or run['run_attempt'] != 1:
        raise ValueError('Screen run is not verified complete attempt1')
    review = api('issues/comments/' + str(gate['independentOutcomeReviewCommentId']))
    report = prefixed(review['body'], 'KEEL_HELM_OUTCOME_REVIEW_V1')
    implementation = load(out / 'dispatch-assignment.json')['implementationTasks']
    if (report is None or review['user']['login'] != 'stephanelee'
            or report.get('verdict') != 'SCREEN-PASS' or report.get('stage') != 'screen'
            or report.get('sourceFreezeSHA256') != gate['sourceFreezeSHA256']
            or report.get('runId') != gate['screenRunId'] or report.get('headSHA') != decision['headSHA']
            or report.get('decisionSHA256') != gate['decisionSHA256']
            or not report.get('reviewerTask') or report['reviewerTask'] in implementation
            or not isinstance(report.get('agentToolProvenance'), dict) or not report['agentToolProvenance']):
        raise ValueError('Missing separate independent screen outcome review')
    save(out / 'screen-gate-verified.json', dict(gate=gate, receipt=tag, run=run, review=review))


def run(stage):
    out = BUILD / stage
    if not (out / 'approval.json').is_file():
        raise ValueError('No same-worker independent approval')
    if (out / 'decision.json').exists() or (out / 'receipt.json').exists():
        raise FileExistsError('Stage cannot resume or repeat')
    if stage == 'confirmation':
        verify_confirmation_gate(out)
    plan = stage_plan(stage)
    report = dict(stage=stage, headSHA=head(), games=[], unplayedIDs=[g['id'] for g in plan],
        decision='pending', observedPhysicalRounds=0, completedPhysicalRounds=0)
    save(out / 'decision.json', report)
    verify_worker(stage)
    consume_receipt(stage, out)
    for game in plan:
        data = out / 'games' / game['id']
        try:
            verify_worker(stage)
            revalidate_authority(stage, out, 'before-' + game['id'])
            remain, jobs = remaining_seconds()
            save(data / 'admission-jobs.json', jobs)
            if remain < 2100:
                raise ValueError('Less than35minutes remain; game not started')
            row = dict(**game, status='valid', **execute_game(game, out))
        except Exception as exc:
            row = dict(**game, status='invalid', error=type(exc).__name__ + ': ' + str(exc),
                traceback=traceback.format_exc(), **partial_observation(data))
            save(data / 'failure.json', row)
        report['games'].append(row)
        report['unplayedIDs'] = [g['id'] for g in plan[len(report['games']):]]
        report['observedPhysicalRounds'] = sum(v.get('observedPhysicalRounds', 0) for v in report['games'])
        report['completedPhysicalRounds'] = sum(v.get('completedPhysicalRounds', 0) for v in report['games'])
        save(out / 'decision.json', report)
        if row['status'] != 'valid':
            report.update(decision='invalid', integrityFailure=row['error'])
            break
    if report['decision'] == 'pending':
        report.update(decide(stage, report['games'], [g['id'] for g in plan]))
    save(out / 'decision.json', report)
    print(json.dumps(dict(decision=report['decision'], physicalRounds=report['observedPhysicalRounds'],
        unplayedIDs=report['unplayedIDs']), sort_keys=True), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    sub.add_parser('freeze');sub.add_parser('capture-build')
    for cmd in ('prepare', 'run'):
        s = sub.add_parser(cmd);s.add_argument('stage', choices=('screen', 'confirmation'))
    s = sub.add_parser('approve');s.add_argument('stage', choices=('screen', 'confirmation'))
    s.add_argument('--artifact-id', required=True, type=int);s.add_argument('--artifact-digest', required=True)
    a = p.parse_args()
    try:
        if a.command == 'freeze':freeze()
        elif a.command == 'capture-build':capture_build()
        elif a.command == 'prepare':prepare(a.stage)
        elif a.command == 'approve':wait_approval(a.stage, a.artifact_id, a.artifact_digest)
        elif a.command == 'run':run(a.stage)
    except Exception:
        if hasattr(a, 'stage'):
            out = BUILD / a.stage
            out.mkdir(parents=True, exist_ok=True)
            if (out / 'decision.json').exists():
                report = load(out / 'decision.json')
                if report.get('decision') == 'pending':
                    report.update(decision='invalid', infrastructureError=traceback.format_exc())
                    save(out / 'decision.json', report)
            save(out / ('failure-' + a.command + '.json'), dict(phase=a.command,
                stage=a.stage, registeredIDs=[g['id'] for g in stage_plan(a.stage)],
                unplayedIDs=load(out / 'decision.json')['unplayedIDs'] if (out / 'decision.json').exists() else [g['id'] for g in stage_plan(a.stage)],
                error=traceback.format_exc(), timeUTC=dt.datetime.now(dt.timezone.utc).isoformat(),
                noAutomaticRetry=True))
        raise


if __name__ == '__main__':
    main()
