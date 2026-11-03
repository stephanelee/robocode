"""Registered Aegis v1 run: one fresh 20-round Diamond Port pilot, then fresh 3x20 confirmation only if it passes."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import benchmark
import compare
import retcon_stats
import strong_melee_opponents as native

REGISTRATION = '46b36f642c3e6e0e42c74106e7d101723c47608b'
PLAN = 'docs/improvements/aegis-shield-v1.md'
SOURCE = '7fd5b8b2064dd66de0aa9f9076064e5cc70cd00e'
JAR = '50a6b48fc1c20c1153c02702de3c69d345f12bf02125e3eb67585e7bf0505e24'
AIS_JAR = 'b41b5de479b1288f087e0cbc9efa64abb68f12381b41d33ad65ff48be5914800'
PRODUCTION_SOURCE = '9b1624da2d9b09ec9cc4f580b51bc82edd2c1485'
NAME, VERSION = 'Aegis', '0.1.0'
FILES = ('Aegis.jar', 'Aegis.json', 'Aegis.sh', 'Aegis.cmd')
GAME_TIMEOUT = 7200
COUNTERS = ('turns', 'shotsFired', 'bulletHits', 'wallCollisions', 'botCollisions', 'skippedTurns', 'enemyShotsDetected',
            'shieldsFired', 'bulletsBlocked', 'hitsTaken', 'unmatchedRemovals', 'inactivityTurns', 'maxDecisionMicros', 'slowTurns')


def layout(stage):
    if stage == 'pilot':
        return [dict(gameId='aegis-shield-v1-pilot', stage=stage, seed=20263801, rounds=20)]
    if stage == 'confirmation':
        return [dict(gameId=f'aegis-shield-v1-confirm-g{i:02d}', stage=stage, seed=20263810 + i, rounds=20) for i in range(1, 4)]
    raise ValueError('Unregistered stage')


def check_rows(rows, rounds, seed):
    if len(rows) != rounds:
        raise ValueError('Incomplete Aegis telemetry')
    for i, row in enumerate(rows, 1):
        expected = dict(schemaVersion=1, name=NAME, version=VERSION, round=i, roundsCompleted=i, seed=seed, mode='train', checkpointLoaded=False)
        if any(row.get(k) != v for k, v in expected.items()) or any(type(row[k]) is not int for k in ('schemaVersion', 'round', 'roundsCompleted', 'seed')):
            raise ValueError('Wrong Aegis identity/round/cold policy')
        for key in COUNTERS:
            if type(row.get(key)) is not int or row[key] < 0:
                raise ValueError('Invalid Aegis counter ' + key)
        for key in ('damageDealt', 'damageTaken', 'energySpent', 'meanLabelError'):
            if type(row.get(key)) not in (float, int) or not math.isfinite(row[key]) or row[key] < 0:
                raise ValueError('Nonfinite Aegis measurement ' + key)
        if any(type(row.get(k)) is not bool for k in ('won', 'died')) or (row['won'] and row['died']):
            raise ValueError('Invalid terminal flags')
        outcomes = row.get('shotOutcomes')
        if not isinstance(outcomes, dict) or any(type(v) is not int or v < 0 for v in outcomes.values()):
            raise ValueError('Invalid shot outcomes')
        if sum(outcomes.values()) != row['enemyShotsDetected'] or outcomes.get('blocked', 0) != row['bulletsBlocked']:
            raise ValueError('Shot outcomes do not reconcile')
        # shieldsFired counts fire commands, which the engine refuses at or below 0.1 energy or on the final turn,
        # so it has no upper bound against confirmed bullets; only blocks need a command.
        if row['bulletsBlocked'] > row['shieldsFired'] or row['bulletHits'] > row['shotsFired']:
            raise ValueError('Inconsistent shield counters')
        if not isinstance(row.get('skippedTurnNumbers'), list) or len(row['skippedTurnNumbers']) != min(row['skippedTurns'], 200):
            raise ValueError('Skipped turn list differs from count')
        if not 0.1 * row['shotsFired'] - 1e-6 <= row['energySpent'] <= 3 * row['shotsFired'] + 1e-6:
            raise ValueError('Confirmed shot energy outside legal power range')


def summarize(rows):
    keys = ('turns', 'shotsFired', 'bulletHits', 'damageDealt', 'damageTaken', 'energySpent', 'wallCollisions', 'botCollisions', 'skippedTurns',
            'enemyShotsDetected', 'shieldsFired', 'bulletsBlocked', 'hitsTaken', 'unmatchedRemovals', 'inactivityTurns', 'slowTurns')
    total = {k: sum(r[k] for r in rows) for k in keys}
    total['maxDecisionMicros'] = max(r['maxDecisionMicros'] for r in rows)
    total['roundWins'] = sum(r['won'] for r in rows)
    total['blockRate'] = total['bulletsBlocked'] / total['enemyShotsDetected'] if total['enemyShotsDetected'] else None
    total['meanLabelErrorByRound'] = [r['meanLabelError'] for r in rows]
    outcomes = {}
    for r in rows:
        for k, v in r['shotOutcomes'].items():
            outcomes[k] = outcomes.get(k, 0) + v
    total['shotOutcomes'] = outcomes
    return total


def decision(games, stage):
    expected = layout(stage)
    if len(games) != len(expected) or [{k: g[k] for k in e} for g, e in zip(games, expected)] != expected:
        raise ValueError('Incomplete registered game cohort')
    shares = [g['scoreShare'] for g in games]
    wins = sum(g['battle']['firstPlaces'] for g in games)
    checks = dict(complete=True, noOwnSkips=all(g['metrics']['skippedTurns'] == 0 for g in games))
    if stage == 'pilot':
        checks.update(progressShare=shares[0] > .20, oneWin=wins >= 1)
    else:
        checks.update(winningMean=statistics.mean(shares) > .50, majorityWins=wins > 30, majorityGames=sum(v > .50 for v in shares) >= 2)
    result = dict(status=('advance' if stage == 'pilot' else 'confirmed') if all(checks.values()) else 'reject', checks=checks,
                  meanShare=statistics.mean(shares), firstPlaces=wins, shares=shares)
    if stage == 'confirmation':
        result['descriptiveTLower95'] = statistics.mean(shares) - 2.919985580353724 * statistics.stdev(shares) / math.sqrt(3)
        result['intervalLimitation'] = 'Descriptive game-level Student-t; not a gate or independent-round inference.'
    return result


def play(args, game, runner, source, python):
    root = args.output / game['gameId']; root.mkdir(); run = root / 'run'; run.mkdir()
    bot = run / NAME; shutil.copytree(args.bot_dir, bot)
    if {f.name for f in bot.iterdir() if f.is_file()} != set(FILES):
        raise ValueError('Unexpected Aegis bundle assets')
    if benchmark.digest(bot / 'Aegis.jar') != JAR:
        raise ValueError('Aegis JAR changed')
    (bot / 'Aegis.sh').chmod(0o755)
    descriptor = json.loads((bot / 'Aegis.json').read_bytes())
    if (descriptor['name'], descriptor['version']) != (NAME, VERSION):
        raise ValueError('Wrong Aegis descriptor')
    env = native.clean_environment(os.environ)
    env = {k: v for k, v in env.items() if not k.startswith(('AEGIS_', 'CHALLENGER_', 'DIAMOND_'))}
    env.update(native.stage(source, run, python))
    opponent = run / 'opponents/diamond-port'; benchmark.isolate_opponent(source / 'bots/ports/diamond-port', opponent)
    opponent_info = json.loads((opponent / 'diamond-port.json').read_bytes())
    metrics = root / 'metrics.jsonl'
    env.update(AEGIS_SEED=str(game['seed']), AEGIS_METRICS_PATH=str(metrics))
    env['PATH'] = str(Path(args.java).parent) + os.pathsep + env.get('PATH', '')
    official = run / 'battle.json'
    command = [args.java, '-cp', str(runner), str(benchmark.ROOT / 'tools/BenchmarkBattle.java'), str(game['rounds']), str(GAME_TIMEOUT), str(official), str(bot), str(opponent)]
    benchmark.run_process(command, benchmark.ROOT, env, run / 'runner.log', GAME_TIMEOUT + 120)
    data = json.loads(official.read_bytes())
    expected = sorted([(NAME, VERSION), (opponent_info['name'], opponent_info['version'])])
    if data['numberOfRounds'] != game['rounds'] or sorted((r['name'], r['version']) for r in data['results']) != expected or len({r['id'] for r in data['results']}) != 2:
        raise ValueError('Official round count/identities differ')
    if (opponent_info['name'], opponent_info['version']) != ('Diamond Port', '1.8.28'):
        raise ValueError('Wrong opponent')
    own = next(r for r in data['results'] if r['name'] == NAME)
    rows = [json.loads(line) for line in metrics.read_bytes().splitlines()]
    check_rows(rows, game['rounds'], game['seed'])
    if sum(r['won'] for r in rows) != own['firstPlaces']:
        raise ValueError('Official wins differ from terminal events')
    battle = dict(**own, results=data['results'])
    return dict(**game, battle=battle, scoreShare=compare.score_share(battle), metrics=summarize(rows),
                botAssetHashes={name: benchmark.digest(bot / name) for name in FILES}, pythonSourceHashes=native.verify_staged(run, ['diamond-port']))


def main(args):
    args.output, args.cache, args.bot_dir = args.output.resolve(), args.cache.resolve(), args.bot_dir.resolve()
    if args.output.exists():
        raise ValueError('Never replace an attempt')
    args.output.mkdir(parents=True)
    plan = subprocess.check_output(['git', 'show', REGISTRATION + ':' + PLAN], cwd=benchmark.ROOT)
    (args.output / 'registered-plan.md').write_bytes(plan)
    if subprocess.check_output(['git', 'diff', '--name-only', PRODUCTION_SOURCE, 'HEAD', '--', 'src/main/java', 'src/test/java', 'AISurvivor', 'build.gradle', 'settings.gradle'], cwd=benchmark.ROOT).strip():
        raise ValueError('Experiment changed production AIS')
    if subprocess.check_output(['git', 'diff', '--name-only', SOURCE, 'HEAD', '--', 'experiments/Aegis'], cwd=benchmark.ROOT).strip():
        raise ValueError('Frozen Aegis source changed')
    if subprocess.check_output(['git', 'diff', '--name-only', REGISTRATION, 'HEAD', '--', PLAN], cwd=benchmark.ROOT).strip():
        raise ValueError('Registered plan changed before execution')
    if benchmark.digest(benchmark.ROOT / 'AISurvivor/AISurvivor.jar') != AIS_JAR or benchmark.digest(benchmark.ROOT / 'experiments/Aegis/Aegis.jar') != JAR:
        raise ValueError('Tracked JAR changed')
    report = dict(status='running', registration=REGISTRATION, planSHA256=hashlib.sha256(plan).hexdigest(), source=SOURCE, jarSHA256=JAR,
                  driverSHA256=benchmark.digest(Path(__file__).resolve()),
                  productionAISJarSHA256=AIS_JAR, executionSource=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=benchmark.ROOT, text=True).strip(),
                  runtimeOS=retcon_stats.os_provenance(), games=[])
    target = args.output / 'experiment.json'

    def save():
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    save()
    try:
        report['runtimeJava'] = subprocess.check_output([args.java, '-XshowSettings:properties', '-version'], stderr=subprocess.STDOUT, text=True, timeout=15)
        source, python, provenance = native.prepare(args.cache, args.python); report['pythonOpponents'] = provenance; save()
        runner = benchmark.download(args.cache, 'robocode-tankroyale-runner-1.4.0.jar', benchmark.ASSETS['robocode-tankroyale-runner-1.4.0.jar'])
        for stage in ('pilot', 'confirmation'):
            if stage == 'confirmation' and report['pilotDecision']['status'] != 'advance':
                break
            for game in layout(stage):
                report['inProgress'] = game['gameId']; save()
                report['games'].append(play(args, game, runner, source, python)); del report['inProgress']; save()
                print(json.dumps({k: report['games'][-1][k] for k in ('gameId', 'scoreShare', 'metrics')}), flush=True)
            report[stage + 'Decision'] = decision([g for g in report['games'] if g['stage'] == stage], stage); save()
        report.update(status='complete', decision=report.get('confirmationDecision', report['pilotDecision']))
        print(json.dumps(report['decision']), flush=True)
    except BaseException as error:
        report.update(status='invalid', failure=str(error)); raise
    finally:
        save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('output', 'bot-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--cache', type=Path, default=benchmark.ROOT / '.benchmark-cache')
    parser.add_argument('--java', required=True)
    parser.add_argument('--python', default=sys.executable)
    main(parser.parse_args())
