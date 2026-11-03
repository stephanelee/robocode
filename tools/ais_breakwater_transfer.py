"""Frozen whole-product AIS movement transfer: one screen, then fresh confirmation only on pass."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import statistics
import subprocess
import sys
from unittest.mock import patch

import benchmark
import coherent_duel_v2_experiment as coherent
import compare
import default_duel_policy_screen as policy
import retcon
import retcon_stats
import strong_melee_opponents as native

REGISTRATION = 'PENDING'
SOURCE = '1cd8db2ea6ccfb56914472bcc51757c0e91d2cff'
PLAN = 'docs/improvements/ais-breakwater-transfer-v1.md'
PROTOCOL = 'tools/fixtures/ais-breakwater-transfer/protocol.json'
BASE = '9b1624da2d9b09ec9cc4f580b51bc82edd2c1485'
JARS = dict(previous='b41b5de479b1288f087e0cbc9efa64abb68f12381b41d33ad65ff48be5914800', candidate='4aceb212ab23e994bcbd95ab1a71920c4422708d386a0b4b032b8eac1b0b79a3')
BREAKWATER = 'b9f61e536c0ff5860f6aabc0f6d3f4a11685a3d9'
BREAKWATER_JAR = 'e6c98feef83e083149df782d643496fb6b6fe3fab33c95356ea2634981c6baf0'
OPS = ('Breakwater', 'Retcon', 'diamond-port', 'adaptive-prime')
T95_DF2 = 2.919985580355516


def layout(stage):
    games = []
    if stage in ('screen', 'confirmation'):
        blocks, rounds, offset = (1, 10, 0) if stage == 'screen' else (3, 20, 1000)
        for i, kind in enumerate(OPS):
            for block in range(1, blocks + 1):
                for arm in (('previous', 'candidate') if (i + block) % 2 else ('candidate', 'previous')):
                    index = len(games)
                    games.append(dict(gameId=f'ais-bw-v1-{stage}-{kind}-b{block}-{arm}', stage=stage,
                        kind=kind, block=block, rounds=rounds, arms=[arm],
                        seeds={arm: 202638000 + offset + 2 * index}, opponentSeed=202638001 + offset + 2 * index))
        for block in range(1, blocks + 1):
            index = len(games)
            arms = ['candidate', 'previous'] if block % 2 else ['previous', 'candidate']
            games.append(dict(gameId=f'ais-bw-v1-{stage}-self-b{block}', stage=stage, kind='self',
                block=block, rounds=10, arms=arms,
                seeds={arm: 202638000 + offset + 2 * index + n for n, arm in enumerate(arms)}))
    elif stage == 'samples':
        for block, kind in enumerate(('Target', 'SpinBot', 'Walls'), 1):
            for arm in (('previous', 'candidate') if block % 2 else ('candidate', 'previous')):
                index = len(games)
                games.append(dict(gameId=f'ais-bw-v1-sample-{kind}-{arm}', stage=stage, kind=kind,
                    block=block, rounds=5, arms=[arm], seeds={arm: 202640000 + index}))
    else:
        raise ValueError('Unregistered stage')
    return games


def validate_layout(games, stages):
    expected = [g for stage in stages for g in layout(stage)]
    if len(games) != len(expected) or [{k: g[k] for k in e} for g, e in zip(games, expected)] != expected:
        raise ValueError('Incomplete or changed fixed game population/order')


def opponent_result(games, kind):
    selected = [g for g in games if g['kind'] == kind]
    blocks = [{arm: next(g['battles'][arm] for g in selected if g['block'] == block and arm in g['arms'])
               for arm in ('previous', 'candidate')} for block in sorted({g['block'] for g in selected})]
    deltas = [compare.score_share(b['candidate']) - compare.score_share(b['previous']) for b in blocks]
    return dict(deltas=deltas, meanGain=statistics.mean(deltas),
        totals={arm: compare.aggregate(blocks, arm) for arm in ('previous', 'candidate')},
        candidateShares=[compare.score_share(b['candidate']) for b in blocks])


def wall_safe(result):
    return result['totals']['candidate']['wallsPer1000Turns'] <= result['totals']['previous']['wallsPer1000Turns'] + 10


def movement_activation(rows, arm):
    total = 0
    for row in rows:
        count = row.get('breakwaterMovementTurns', 0)
        expected = row['forcedDuelTurns'] if arm == 'candidate' else 0
        if type(count) is not int or count < 0 or count != expected:
            raise ValueError('Breakwater movement activation differs from executed forced-duel decisions')
        total += count
    if arm == 'candidate' and total == 0:
        raise ValueError('No candidate Breakwater movement was executed')
    return total


def decision(games, confirmation=False):
    validate_layout(games, ['confirmation', 'samples'] if confirmation else ['screen'])
    results = {kind: opponent_result(games, kind) for kind in OPS + ('self',)}
    bw, ret, selves = (results[k] for k in ('Breakwater', 'Retcon', 'self'))
    advanced = [results[k] for k in OPS[2:]]
    shares = selves['candidateShares']
    checks = dict(
        breakwaterGain=bw['meanGain'] >= .05,
        breakwaterWinningShare=statistics.mean(bw['candidateShares']) > .5,
        breakwaterWins=bw['totals']['candidate']['wins'] > (30 if confirmation else 5),
        retconRelative=ret['meanGain'] >= -.05,
        retconAbsolute=all(s > .5 for s in ret['candidateShares']) and ret['totals']['candidate']['wins'] > (30 if confirmation else 5),
        advancedEachNI=all(r['meanGain'] >= -.05 for r in advanced),
        advancedMeanNI=statistics.mean(r['meanGain'] for r in advanced) >= 0,
        selfShare=statistics.mean(shares) > .5,
        selfWins=selves['totals']['candidate']['wins'] > (15 if confirmation else 5),
        wallRate=all(wall_safe(r) for r in results.values()),
        zeroAISSkips=all(b['metrics']['skippedTurns'] == 0 for g in games for b in g['battles'].values()))
    extra = {}
    if confirmation:
        lower = statistics.mean(ret['deltas']) - T95_DF2 * statistics.stdev(ret['deltas']) / math.sqrt(3)
        checks.update(retconTLowerNI=lower >= -.05, breakwaterEveryGain=all(d > 0 for d in bw['deltas']),
                      selfGames=sum(s > .5 for s in shares) >= 2)
        samples = {kind: opponent_result(games, kind) for kind in ('Target', 'SpinBot', 'Walls')}
        checks['samples'] = all(r['meanGain'] >= -.05 and wall_safe(r) for r in samples.values())
        extra.update(retconTLower95=lower, samples=samples)
    return dict(status=('retain' if confirmation else 'advance') if all(checks.values()) else 'reject',
                checks=checks, opponents=results, **extra)


def runtime_fingerprint(java, python):
    home = Path(java).resolve().parent.parent
    query = 'import sys,sysconfig,json;print(json.dumps(dict(version=sys.version,executable=sys.executable,libdir=sysconfig.get_config_var("LIBDIR"),library=sysconfig.get_config_var("LDLIBRARY"))))'
    py = json.loads(subprocess.check_output([python, '-c', query], text=True, timeout=15))
    paths = {name: home / name for name in ('bin/java', 'lib/modules', 'lib/server/libjvm.so', 'release')}
    paths.update(python=Path(py['executable']).resolve(), libpython=Path(py['libdir']) / py['library'])
    return dict(javaVersion=subprocess.check_output([java, '-version'], stderr=subprocess.STDOUT, text=True, timeout=15),
                pythonVersion=py['version'], hashes={key: benchmark.digest(path) for key, path in paths.items()})


def stage_breakwater(directory, java, seed):
    bot = directory / 'opponents/Breakwater'
    bot.mkdir(parents=True)
    for name in ('Breakwater.jar', 'Breakwater.json'):
        source = 'experiments/Breakwater/' + ('bot/' if name.endswith('.json') else '') + name
        data = subprocess.check_output(['git', 'show', BREAKWATER + ':' + source], cwd=benchmark.ROOT)
        (bot / name).write_bytes(data)
    if benchmark.digest(bot / 'Breakwater.jar') != BREAKWATER_JAR:
        raise ValueError('Frozen Breakwater JAR differs')
    variables = dict(BREAKWATER_SEED=str(seed), BREAKWATER_GUN_MODE='envelope',
        BREAKWATER_MODEL_PATH=str(directory / 'breakwater-model.json'), BREAKWATER_METRICS_PATH=str(directory / 'breakwater.jsonl'))
    script = '#!/bin/sh\nset -eu\ncd "$(dirname "$0")"\n'
    script += ''.join('export ' + k + '=' + shlex.quote(v) + '\n' for k, v in variables.items())
    script += 'exec ' + shlex.quote(java) + ' -jar Breakwater.jar\n'
    (bot / 'Breakwater.sh').write_text(script, encoding='utf-8', newline='\n')
    (bot / 'Breakwater.sh').chmod(0o755)
    (bot / 'launch-provenance.json').write_text(json.dumps(dict(source=BREAKWATER, jarSHA256=BREAKWATER_JAR, variables=variables,
        hashes={p.name: benchmark.digest(p) for p in bot.iterdir() if p.is_file()}), indent=2) + '\n')
    return bot


def check_breakwater(directory, rounds, seed, official):
    rows = [json.loads(line) for line in (directory / 'breakwater.jsonl').read_bytes().splitlines()]
    endpoint = json.loads((directory / 'breakwater-model.json').read_bytes())
    expected = dict(schemaVersion=3, name='Breakwater', version='0.3.0', gunMode='envelope', seed=seed, checkpointLoaded=False)
    if len(rows) != rounds or any(any(row.get(k) != v for k, v in expected.items()) or row.get('round') != i for i, row in enumerate(rows, 1)):
        raise ValueError('Breakwater cold identity/round coverage differs')
    first_places = 0
    for row in rows:
        if type(row['officialRoundWins']) is not int or row['officialRoundWins'] not in (0, 1):
            raise ValueError('Invalid Breakwater official win increment')
        first_places += row['officialRoundWins']
        if row['officialFirstPlaces'] != first_places or any(type(row.get(k)) is not bool for k in ('won', 'died')):
            raise ValueError('Invalid Breakwater round outcome accounting')
        if row['bulletHits'] > row['shotsFired'] or not .1 * row['shotsFired'] - 1e-6 <= row['energySpent'] <= 3 * row['shotsFired'] + 1e-6:
            raise ValueError('Invalid Breakwater confirmed shot measurements')
    if any(endpoint.get(k) != v for k, v in expected.items()) or endpoint.get('roundsCompleted') != rounds:
        raise ValueError('Breakwater endpoint differs')
    if endpoint['officialFirstPlaces'] != official['firstPlaces'] or sum(r['officialRoundWins'] for r in rows) != official['firstPlaces']:
        raise ValueError('Breakwater official score reconciliation differs')
    for key, total in endpoint['totals'].items():
        if type(total) not in (float, int) or not math.isfinite(total) or total < 0 or any(type(r.get(key)) not in (float, int) or not math.isfinite(r[key]) or r[key] < 0 for r in rows) or not math.isclose(sum(r[key] for r in rows), total, rel_tol=1e-9, abs_tol=1e-6):
            raise ValueError('Breakwater cumulative endpoint differs: ' + key)
    if not isinstance(endpoint.get('corrections'), list) or len(endpoint['corrections']) > 32 or any(type(v) not in (float, int) or not math.isfinite(v) or abs(v) > 1 for v in endpoint['corrections']):
        raise ValueError('Invalid Breakwater learned correction memory')
    scores, history = endpoint.get('attackScores'), endpoint.get('attackHistory')
    if not isinstance(scores, list) or len(scores) != 4 or any(type(v) is not int or v < 0 for v in scores) or not isinstance(history, list) or len(history) > 32 or scores[3] != len(history) or any(type(v) is not int or not 0 <= v < 8 for v in history) or scores[:3] != [sum((v >> i) & 1 for v in history) for i in range(3)]:
        raise ValueError('Invalid Breakwater virtual gun history')
    if any(r['skippedTurns'] != 0 for r in rows):
        raise ValueError('Breakwater skipped turn invalidates transfer comparison')
    return dict(modelSHA256=benchmark.digest(directory / 'breakwater-model.json'), metrics=benchmark.summarize_metrics(rows),
                rawWon=sum(r['won'] for r in rows), officialFirstPlaces=official['firstPlaces'])


def play(args, game, sources, runner, samples, python_source, python, retcon_source, api):
    directory = args.output / game['gameId']
    directory.mkdir()
    entries, bots = [], {}
    for arm in game['arms']:
        destination = directory / arm / 'AISurvivor'
        with patch.dict(coherent.JARS, JARS, clear=True):
            bots[arm] = coherent.prepare_bot(sources[arm], destination, arm, game['seeds'][arm], args.java)
        entries.append(str(destination))
    env = native.clean_environment(os.environ)
    env = {k: v for k, v in env.items() if not k.startswith(('CHALLENGER_', 'BREAKWATER_'))}
    env['PATH'] = str(Path(args.java).parent) + os.pathsep + env.get('PATH', '')
    kind, capture, native_stats, opponent_info = game['kind'], None, None, None
    if kind != 'self':
        if kind == 'Breakwater':
            opponent = stage_breakwater(directory, args.java, game['opponentSeed'])
            descriptor = opponent / 'Breakwater.json'
        elif kind in OPS[2:]:
            env.update(native.stage(python_source, directory, python))
            opponent = directory / 'opponents' / kind
            spec = next(b for b in native.manifest()['opponents'] if Path(b['descriptor']).stem == kind)
            benchmark.isolate_opponent(python_source / spec['sourcePath'], opponent)
            descriptor = opponent / (kind + '.json')
        else:
            if kind == 'Retcon':
                source = retcon_source
                if retcon.source_hash(source) != retcon_stats.SOURCE_SHA256:
                    raise ValueError('Retcon source differs')
                capture = retcon_stats.Capture(retcon_stats.java_temp_directory(args.java))
                env.update(RUMBLE_JAVA_CACHE=str(api), RETCON_STATS_DIR=str(capture.path))
                config = source / 'Retcon.json'
            else:
                configs = list(samples.rglob(kind + '.json'))
                if len(configs) != 1:
                    raise ValueError('Ambiguous sample descriptor')
                config = configs[0]
            opponent = directory / 'opponents' / kind
            benchmark.isolate_opponent(config.parent, opponent)
            descriptor = opponent / config.name
        opponent_info = json.loads(descriptor.read_bytes())
        entries.append(str(opponent))
    official_path = directory / 'battle.json'
    command = [args.java, '-cp', str(runner), str(benchmark.ROOT / 'tools/BenchmarkBattle.java'),
               str(game['rounds']), '900', str(official_path), *entries]
    controls = {k: env[k] for k in ('ROBOCODE_PYTHON_BIN', 'ROBOCODE_ENV_FILE', 'ROBOCODE_GUN_ENV_FILE',
                'PYTHONNOUSERSITE', 'RUMBLE_JAVA_CACHE', 'RETCON_STATS_DIR') if k in env}
    (directory / 'command.json').write_text(json.dumps(dict(argv=command, environmentOverrides=controls), indent=2) + '\n')
    try:
        benchmark.run_process(command, benchmark.ROOT, env, directory / 'runner.log', 990)
        official = json.loads(official_path.read_bytes())
        expected = [(b['name'], b['version']) for b in bots.values()]
        if opponent_info:
            expected.append((opponent_info['name'], opponent_info['version']))
        if official['numberOfRounds'] != game['rounds'] or sorted((r['name'], r['version']) for r in official['results']) != sorted(expected) or len({r['id'] for r in official['results']}) != 2:
            raise ValueError('Official roster/round coverage differs')
        opponent_row = next((r for r in official['results'] if opponent_info and r['name'] == opponent_info['name']), None)
        if capture:
            native_stats = capture.collect(directory / 'retcon-stats', game['rounds'], opponent_row['id'], require=True)
    finally:
        if capture:
            try:
                if native_stats is None:
                    capture.collect(directory / 'retcon-stats', game['rounds'], None, require=False)
            finally:
                capture.close()
    battles = {}
    for arm, bot in bots.items():
        identity = next(r for r in official['results'] if (r['name'], r['version']) == (bot['name'], bot['version']))
        rows = [json.loads(line) for line in Path(bot['metrics']).read_bytes().splitlines() if line.strip()]
        benchmark.validate_metrics(rows, game['rounds'], 'train', bot['seed'], game['rounds'])
        metrics = benchmark.summarize_metrics(rows)
        if metrics['skippedTurns'] or metrics['checkpointLoaded']:
            raise ValueError('Skipped turn/loaded model invalidates comparison')
        battles[arm] = dict(**identity, rounds=game['rounds'], results=official['results'], metrics=metrics,
            endpoint=coherent.validate_endpoint(Path(bot['model']), game['rounds']), mechanism=policy.mechanism(rows), bot=bot,
            breakwaterMovementTurns=movement_activation(rows, arm))
    return dict(**game, battles=battles, retconStats=native_stats,
        pythonSourceHashes=native.verify_staged(directory, [kind]) if kind in OPS[2:] else {},
        breakwater=check_breakwater(directory, game['rounds'], game['opponentSeed'], opponent_row) if kind == 'Breakwater' else None)


def main(args):
    if len(REGISTRATION) != 40 or len(SOURCE) != 40 or any(len(v) != 64 for v in JARS.values()):
        raise ValueError('Freeze source/JAR/registration before execution')
    args.output, args.cache = args.output.resolve(), args.cache.resolve()
    if args.output.exists():
        raise ValueError('Never replace an attempt')
    args.output.mkdir(parents=True)
    sources = dict(previous=args.previous.resolve(), candidate=args.candidate.resolve())
    plan = subprocess.check_output(['git', 'show', REGISTRATION + ':' + PLAN], cwd=benchmark.ROOT)
    encoded = subprocess.check_output(['git', 'show', REGISTRATION + ':' + PROTOCOL], cwd=benchmark.ROOT)
    registered = json.loads(encoded)
    if registered['games'] != {s: layout(s) for s in ('screen', 'confirmation', 'samples')} or registered['candidate'] != dict(source=SOURCE, jarSHA256=JARS['candidate']) or registered['baseline'] != dict(source=BASE, jarSHA256=JARS['previous']):
        raise ValueError('Frozen protocol/source differs')
    (args.output / 'registered-plan.md').write_bytes(plan)
    (args.output / 'registered-protocol.json').write_bytes(encoded)
    report = dict(status='running', registration=REGISTRATION, sources=dict(previous=BASE, candidate=SOURCE), jarSHA256=JARS,
        planSHA256=hashlib.sha256(plan).hexdigest(), protocolSHA256=hashlib.sha256(encoded).hexdigest(),
        executionSource=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=benchmark.ROOT, text=True).strip(),
        runtimeOS=retcon_stats.os_provenance(), games=[])
    output = args.output / 'experiment.json'
    def save(): output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    save()
    try:
        for arm, path in sources.items():
            if benchmark.digest(path / 'AISurvivor.jar') != JARS[arm]:
                raise ValueError('AIS frozen JAR differs')
        if subprocess.check_output(['git', 'diff', '--name-only', SOURCE, 'HEAD', '--', 'src/main/java', 'src/test/java', 'AISurvivor'], cwd=benchmark.ROOT).strip():
            raise ValueError('Candidate changed after freeze')
        actual_runtime = runtime_fingerprint(args.java, args.python)
        report['runtime'] = actual_runtime
        save()
        if actual_runtime['hashes'] != registered['runtime']['hashes']:
            raise ValueError('Runtime file hashes differ from registration')
        python_source, python, provenance = native.prepare(args.cache, args.python)
        retcon_source, api, retcon_provenance = retcon.prepare(args.cache)
        runner = benchmark.download(args.cache, 'robocode-tankroyale-runner-1.4.0.jar', benchmark.ASSETS['robocode-tankroyale-runner-1.4.0.jar'])
        archive = benchmark.download(args.cache, 'sample-bots-Java-1.4.0.zip', benchmark.ASSETS['sample-bots-Java-1.4.0.zip'])
        samples = args.output / 'samples'
        benchmark.extract(archive, samples)
        report.update(pythonOpponents=provenance, retconProvenance=retcon_provenance,
                      runnerSHA256=benchmark.digest(runner), sampleAssetSHA256=benchmark.digest(archive))
        save()
        for stage in ('screen', 'confirmation', 'samples'):
            if stage != 'screen' and report['screenDecision']['status'] != 'advance':
                break
            for game in layout(stage):
                report['inProgressGame'] = game['gameId']
                save()
                report['games'].append(play(args, game, sources, runner, samples, python_source, python, retcon_source, api))
                del report['inProgressGame']
                save()
            if stage == 'screen':
                report['screenDecision'] = decision(report['games'])
                save()
                print(json.dumps(report['screenDecision'], indent=2), flush=True)
        report['decision'] = decision([g for g in report['games'] if g['stage'] != 'screen'], True) if report['screenDecision']['status'] == 'advance' else dict(status='reject-at-screen', confirmationRounds=0)
        report['status'] = 'complete'
        print(json.dumps(report['decision'], indent=2), flush=True)
    except BaseException as error:
        report.update(status='invalid', error=str(error))
        raise
    finally:
        save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('previous', 'candidate', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--java', required=True)
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--cache', type=Path, default=benchmark.ROOT / '.benchmark-cache')
    main(parser.parse_args())
