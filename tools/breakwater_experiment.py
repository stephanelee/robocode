"""Run the frozen Breakwater pilot once, conditionally followed by three fresh games."""
import argparse
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import statistics
import subprocess
import sys
import sysconfig

import benchmark
import strong_melee_opponents as native

ROOT = benchmark.ROOT
BOT = ROOT / 'experiments/Breakwater'
BUNDLE = BOT / 'build/bot/Breakwater'
PLAN = ROOT / 'docs/improvements/breakwater-v1.md'
FREEZE = ROOT / 'tools/breakwater-freeze.json'
FILES = ('Breakwater.jar', 'Breakwater.json', 'Breakwater.sh', 'Breakwater.cmd')
BATTLE = 'robocode-tankroyale-runner-1.4.0.jar'
BASE = 'dbdeb5649fcb3e85c0c84748afdbe110bb016bbb'
VERSION = '0.1.0'
INTEGERS = ('turns', 'shotsFired', 'bulletHits', 'wallCollisions', 'botCollisions',
            'skippedTurns', 'rangeObservedTurns', 'inferredWaves', 'shieldSolutions',
            'shieldShots', 'attackShots', 'unclassifiedShots', 'shieldIntercepts',
            'attackHits', 'shieldHits', 'matchedRays')
REALS = ('damageDealt', 'damageTaken', 'energySpent', 'rangeSum', 'attackDamage', 'shieldDamage')


def layout(stage):
    if stage == 'pilot':
        return [('breakwater-v1-pilot', 531801)]
    if stage == 'confirmation':
        return [(f'breakwater-v1-confirm-g{i:02d}', 531810 + i) for i in range(1, 4)]
    raise ValueError('Unknown stage')


def decision(games, stage):
    if [(g['gameId'], g['seed']) for g in games] != layout(stage):
        raise ValueError('Incomplete registered cohort')
    shares = [g['scoreShare'] for g in games]
    wins = sum(g['wins'] for g in games)
    mean = statistics.mean(shares)
    checks = {'zeroSkips': all(g['metrics']['skippedTurns'] == 0 for g in games)}
    if stage == 'pilot':
        checks.update(progressShare=mean > .20, oneWin=wins >= 1)
    else:
        checks.update(winningMean=mean > .50, majorityWins=wins > 30,
                      majorityGames=sum(s > .50 for s in shares) >= 2)
    status = 'reject'
    if all(checks.values()):
        status = 'advance' if stage == 'pilot' else 'accept'
    return dict(status=status, checks=checks, wins=wins, equalGameMeanShare=mean, shares=shares)


def check_identity(record, seed):
    if (record.get('name'), record.get('version')) != ('Breakwater', VERSION):
        raise ValueError('Unexpected challenger')
    if type(record.get('seed')) is not int or record['seed'] != seed:
        raise ValueError('Wrong challenger seed')
    if (type(record.get('schemaVersion')) is not int or record['schemaVersion'] != 1
            or record.get('checkpointLoaded') is not False):
        raise ValueError('Unexpected schema or loaded checkpoint')


def check_number(value, key):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError('Invalid number: ' + key)
    if key in INTEGERS and type(value) is not int:
        raise ValueError('Invalid integer: ' + key)


def validate(directory, game_id, seed):
    data = json.loads((directory / 'battle.json').read_bytes())
    results = data['results']
    if data['numberOfRounds'] != 20 or sorted((r['name'], r['version']) for r in results) != [
            ('Breakwater', VERSION), ('Diamond Port', '1.8.28')]:
        raise ValueError('Official round count/participants differ')
    if len({r['id'] for r in results}) != 2:
        raise ValueError('Duplicate official participant')
    for row in results:
        for key in ('totalScore', 'survival', 'bulletDamage', 'firstPlaces'):
            if type(row[key]) is not int or row[key] < 0:
                raise ValueError('Invalid official counter ' + key)
        if row['firstPlaces'] > 20:
            raise ValueError('Invalid official win count')
    own = next(r for r in results if r['name'] == 'Breakwater')
    total_score = sum(row['totalScore'] for row in results)
    if total_score <= 0:
        raise ValueError('Zero official score denominator')
    rows = [json.loads(line) for line in (directory / 'metrics.jsonl').read_bytes().splitlines()]
    if len(rows) != 20:
        raise ValueError('Incomplete telemetry')
    for i, row in enumerate(rows, 1):
        check_identity(row, seed)
        if type(row['round']) is not int or row['round'] != i or not row['turns']:
            raise ValueError('Incomplete/out-of-order round')
        if type(row['won']) is not bool or type(row['died']) is not bool or (row['won'] and row['died']):
            raise ValueError('Invalid terminal flags')
        for key in INTEGERS + REALS:
            check_number(row.get(key), key)
        if row['unclassifiedShots'] or row['shotsFired'] != row['shieldShots'] + row['attackShots']:
            raise ValueError('Unclassified shot')
        if (row['bulletHits'] != row['attackHits'] + row['shieldHits']
                or row['shieldHits'] > row['shieldShots']
                or row['shieldIntercepts'] > row['shieldShots']
                or row['attackHits'] > row['attackShots']):
            raise ValueError('Impossible shot outcomes')
        if not math.isclose(row['damageDealt'], row['shieldDamage'] + row['attackDamage'], abs_tol=1e-6):
            raise ValueError('Unreconciled shot damage')
        if not (.1 * row['shotsFired'] - 1e-6 <= row['energySpent'] <= row['shotsFired'] + 1e-6):
            raise ValueError('Confirmed shot energy outside registered range')
    if sum(r['won'] for r in rows) != own['firstPlaces']:
        raise ValueError('Official first places disagree with round wins')
    model = json.loads((directory / 'model.json').read_bytes())
    check_identity(model, seed)
    if type(model['roundsCompleted']) is not int or model['roundsCompleted'] != 20:
        raise ValueError('Incomplete endpoint')
    totals = {k: sum(row[k] for row in rows) for k in INTEGERS + REALS}
    for k, v in totals.items():
        actual = model['totals'][k]
        check_number(actual, k)
        if (actual != v if k in INTEGERS else not math.isclose(v, actual, abs_tol=2e-6)):
            raise ValueError('Endpoint sum mismatch: ' + k)
    for key in ('corrections', 'attackScores'):
        if not isinstance(model[key], list) or any(type(v) not in (int, float) or not math.isfinite(v) for v in model[key]):
            raise ValueError('Invalid model ' + key)
    corrections = model['corrections']
    if len(corrections) != min(8, totals['matchedRays']) or any(abs(x) > 12 for x in corrections):
        raise ValueError('Invalid bounded ray correction memory')
    history = model['attackHistory']
    if not isinstance(history, list) or len(history) > min(32, totals['attackShots']):
        raise ValueError('Invalid attack history length')
    if any(type(mask) is not int or mask not in (0, 1, 2, 3) for mask in history):
        raise ValueError('Invalid attack history mask')
    counts = [sum(mask & 1 for mask in history), sum(mask >> 1 for mask in history), len(history)]
    if model['attackScores'] != counts:
        raise ValueError('Attack score totals differ from ordered history')
    return dict(gameId=game_id, seed=seed, rounds=20, wins=own['firstPlaces'],
                scoreShare=own['totalScore'] / total_score, official=data,
                metrics=totals, endpointSHA256=benchmark.digest(directory / 'model.json'))


def runtime():
    java_home = Path(os.environ['JAVA_HOME'])
    spec = subprocess.check_output([str(java_home / 'bin/java'), '-version'], stderr=subprocess.STDOUT, text=True)
    osinfo = platform.freedesktop_os_release()
    if (osinfo.get('ID'), osinfo.get('VERSION_ID'), platform.machine()) != ('ubuntu', '26.04', 'x86_64'):
        raise ValueError('Registered Ubuntu 26.04 x86_64 runner required')
    if not re.search(r'version "25[."]', spec) or sys.version_info[:2] != (3, 13):
        raise ValueError('Registered Java 25/Python 3.13 required')
    paths = {p: java_home / p for p in ('bin/java', 'lib/modules', 'lib/server/libjvm.so', 'release')}
    paths.update(python=Path(sys.executable),
                 libpython=Path(sysconfig.get_config_var('LIBDIR')) / sysconfig.get_config_var('LDLIBRARY'))
    return dict(javaVersion=spec, pythonVersion=sys.version, hashes={k: benchmark.digest(p) for k, p in paths.items()})


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def preflight(freeze):
    if not re.fullmatch('[0-9a-f]{40}', freeze['source']):
        raise ValueError('Invalid source SHA')
    protected = ('experiments/Breakwater', 'tools', '.github/workflows', 'src', 'AISurvivor',
                 'experiments/DiamondChallenger', 'gradle', 'gradlew', 'build.gradle', 'settings.gradle')
    changed = git('diff', '--name-only', freeze['source'], '--', *protected).splitlines()
    if any(p != 'tools/breakwater-freeze.json' for p in changed):
        raise ValueError(f'Frozen runtime changed: {changed}')
    if git('diff', '--name-only', BASE, '--', 'AISurvivor', 'src', 'build.gradle', 'settings.gradle',
           'experiments/DiamondChallenger', 'tools/BenchmarkBattle.java', 'tools/strong_melee_opponents.py',
           'tools/opponents/strong-melee.lock.json', 'tools/benchmark.py', 'tools/retcon.py'):
        raise ValueError('Production, shared runner or opponent loader changed')
    expected = {PLAN: 'planSha256', BOT / FILES[0]: 'jarSha256', native.LOCK: 'lockSha256'}
    for path, key in expected.items():
        if benchmark.digest(path) != freeze[key]:
            raise ValueError('Frozen hash mismatch: ' + path.name)
    if freeze['runnerSHA256'] != benchmark.ASSETS[BATTLE] or freeze['runtime'] != runtime():
        raise ValueError('Frozen runtime mismatch')
    if set(p.name for p in BUNDLE.iterdir()) != set(FILES):
        raise ValueError('Unexpected bot bundle assets')
    for name in FILES:
        source = BOT / name if name.endswith('.jar') else BOT / 'bot' / name
        if benchmark.digest(source) != benchmark.digest(BUNDLE / name):
            raise ValueError('Bundle differs from registered source: ' + name)


def save(path, data):
    temp = path.with_suffix('.part')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def play(directory, game_id, seed, runner, source, python):
    directory.mkdir()
    bot = directory / 'Breakwater'
    shutil.copytree(BUNDLE, bot)
    (bot / 'Breakwater.sh').chmod(0o755)
    env = {k: v for k, v in native.clean_environment(os.environ).items()
           if not k.startswith(('BREAKWATER_', 'CHALLENGER_'))}
    env.update(native.stage(source, directory, python))
    opponent = directory / 'opponents/diamond-port'
    benchmark.isolate_opponent(source / 'bots/ports/diamond-port', opponent)
    staged = native.verify_staged(directory, ['diamond-port'])
    save(directory / 'staged-hashes.json', staged)
    env.update(BREAKWATER_SEED=str(seed), BREAKWATER_MODEL_PATH=str(directory / 'model.json'),
               BREAKWATER_METRICS_PATH=str(directory / 'metrics.jsonl'), PYTHONDONTWRITEBYTECODE='1')
    java = Path(os.environ['JAVA_HOME']) / 'bin/java'
    env['PATH'] = str(java.parent) + os.pathsep + env.get('PATH', '')
    command = [str(java), '-cp', str(runner), str(ROOT / 'tools/BenchmarkBattle.java'),
               '20', '900', str(directory / 'battle.json'), str(bot), str(opponent)]
    save(directory / 'command.json', command)
    benchmark.run_process(command, ROOT, env, directory / 'runner.log', 990)
    if native.verify_staged(directory, ['diamond-port']) != staged:
        raise ValueError('Opponent drifted during play')
    return validate(directory, game_id, seed)


def run(output):
    output.mkdir(exist_ok=False, parents=True)
    report = dict(status='preflight', executionHead=git('rev-parse', 'HEAD'), games=[])
    target = output / 'experiment.json'
    save(target, report)
    try:
        freeze = json.loads(FREEZE.read_bytes())
        shutil.copyfile(FREEZE, output / 'freeze.json')
        shutil.copyfile(PLAN, output / 'registered-plan.md')
        preflight(freeze)
        cache = output.parent / '.breakwater-cache'
        source, python, provenance = native.prepare(cache, sys.executable)
        runner = benchmark.download(cache, BATTLE, freeze['runnerSHA256'])
        report.update(status='running', pythonOpponents=provenance, runtime=runtime(),
                      runnerSHA256=benchmark.digest(runner), productionJarSHA256=benchmark.digest(ROOT / 'AISurvivor/AISurvivor.jar'))
        save(target, report)
        for stage in ('pilot', 'confirmation'):
            stage_games = []
            for game_id, seed in layout(stage):
                report['attempting'] = game_id
                save(target, report)
                game = play(output / game_id, game_id, seed, runner, source, python)
                report['games'].append(game)
                stage_games.append(game)
                del report['attempting']
                save(target, report)
            value = report[stage + 'Decision'] = decision(stage_games, stage)
            if value['status'] != 'advance':
                report.update(status='complete', decision=value)
                print(json.dumps(value), flush=True)
                break
    except BaseException as exc:
        report.update(status='invalid', failure=repr(exc))
        raise
    finally:
        save(target, report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--output', type=Path)
    mode.add_argument('--show-runtime', action='store_true')
    args = parser.parse_args()
    if args.show_runtime:
        print(json.dumps(runtime(), indent=2))
    else:
        run(args.output.resolve())
