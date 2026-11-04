"""Frozen Breakwater 0.3.0 Diamond pilot, conditional confirmation and AIS transfer."""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
from statistics import mean

import benchmark
import breakwater_experiment as v1
import breakwater_mobile_experiment as v2
import compare
import strong_melee_opponents as native

ROOT = benchmark.ROOT
BASE = 'e1f58155b31ef7d1a62ccc65a21b94b536c1325b'
FREEZE = ROOT / 'tools/breakwater-v3-freeze.json'
PLAN = ROOT / 'docs/improvements/breakwater-envelope-v3.md'
BOT = ROOT / 'experiments/Breakwater'
BUNDLE = BOT / 'build/bot/Breakwater'
AIS = ROOT / 'AISurvivor'
AIS_SEED = 533022
INTS = v2.INTS + ('envelopeDecisions', 'envelopeOpportunities', 'envelopeHolds', 'envelopeFireRequests', 'envelopeShots', 'envelopeHits')
REALS = v2.REALS


def layout(stage):
    if stage == 'shadow':
        return [('breakwater-envelope-v3-shadow', 533000)]
    if stage == 'pilot':
        return [('breakwater-envelope-v3-pilot', 533001)]
    if stage == 'confirmation':
        return [(f'breakwater-envelope-v3-confirm-g{i:02d}', 533010 + i) for i in range(1, 4)]
    if stage == 'transfer':
        return [('breakwater-envelope-v3-ais-g01', 533021)]
    raise ValueError('Unknown stage')


def decision(games, stage):
    if stage not in ('pilot', 'confirmation') or [(g['gameId'], g['seed']) for g in games] != layout(stage):
        raise ValueError('Invalid Diamond decision cohort')
    shares = [g['scoreShare'] for g in games]
    wins = sum(g['wins'] for g in games)
    checks = dict(zeroSkips=all(g['metrics']['skippedTurns'] == 0 for g in games))
    if stage == 'pilot':
        checks.update(progressShare=shares[0] > .20, oneWin=wins >= 1)
    else:
        checks.update(winningMean=mean(shares) > .50, majorityWins=wins > 30,
                      majorityGames=sum(s > .50 for s in shares) >= 2)
    status = 'advance' if stage == 'pilot' else 'accept'
    return dict(status=status if all(checks.values()) else 'reject', checks=checks,
                wins=wins, equalGameMeanShare=mean(shares), shares=shares)


def check_identity(record, seed, mode):
    if ((record.get('name'), record.get('version')) != ('Breakwater', '0.3.0')
            or type(record.get('seed')) is not int or record['seed'] != seed
            or type(record.get('schemaVersion')) is not int or record['schemaVersion'] != 3 or record.get('gunMode') != mode
            or record.get('checkpointLoaded') is not False):
        raise ValueError('Wrong bot identity/cold policy')


def check_counters(record):
    for key in INTS:
        if type(record.get(key)) is not int or record[key] < 0:
            raise ValueError('Invalid counter: ' + key)
    for key in REALS:
        if type(record.get(key)) not in (float, int) or not math.isfinite(record[key]) or record[key] < 0:
            raise ValueError('Invalid measurement: ' + key)


def official(data, transfer, rounds):
    results = data['results']
    opponent = ('AISurvivor', '1.2.1') if transfer else ('Diamond Port', '1.8.28')
    if (data['numberOfRounds'] != rounds or len(results) != 2 or len({r['id'] for r in results}) != 2
            or {(r['name'], r['version']) for r in results} != {opponent, ('Breakwater', '0.3.0')}):
        raise ValueError('Incomplete or unexpected official participants')
    for r in results:
        for key in ('firstPlaces', 'totalScore', 'survival', 'bulletDamage'):
            if type(r[key]) is not int or r[key] < 0:
                raise ValueError('Invalid official score')
        if r['firstPlaces'] > rounds:
            raise ValueError('Impossible wins')
    own = next(r for r in results if r['name'] == 'Breakwater')
    return own, compare.score_share(dict(**own, results=results))


def replay_opportunity(op, future, stop):
    """Reconstruct both proposed rays and the actual-heading proxy on the same scans."""
    from bisect import bisect_left
    c = op['creationTurn']
    speed = 20 - 3 * op['power']
    dxdy = [(math.cos(math.radians(op[key])), math.sin(math.radians(op[key])))
            for key in ('baselineAim', 'candidateAim', 'currentGun')]
    times, scans = future
    hits, expected = 0, c
    for idx in range(bisect_left(times, c), len(times)):
        t = times[idx]
        if t != expected:
            return 'gap', t, -1
        if t - c >= 90:
            return 'horizon', t, -1
        x = scans[t]['x'] - op['x']
        y = scans[t]['y'] - op['y']
        lo = speed * (t - c + 1)
        for bit, (dx, dy) in enumerate(dxdy):
            along = min(lo + speed, max(lo, x * dx + y * dy))
            if math.hypot(x - along * dx, y - along * dy) <= 18:
                hits |= 1 << bit
        if lo > math.hypot(x, y) + 18:
            return 'resolved', t, hits
        expected += 1
    return 'terminal', stop, -1


def paired_shadow(directory, metrics):
    streams = [([], {}) for _ in metrics]
    for line in (directory / 'shadow-scans.jsonl').read_bytes().splitlines():
        scan = json.loads(line)
        rnd, turn = scan['round'], scan['turn']
        if type(rnd) is not int or not 1 <= rnd <= len(metrics) or type(turn) is not int or scan['kind'] != 'scan':
            raise ValueError('Invalid legal scan identity')
        times, indexed = streams[rnd - 1]
        if turn < 0 or (times and turn <= times[-1]):
            raise ValueError('Duplicate/out-of-order legal scan')
        if any(type(scan.get(k)) not in (int, float) or not math.isfinite(scan[k])
               for k in ('x', 'y', 'heading', 'speed', 'energy')):
            raise ValueError('Invalid legal scan')
        times.append(turn)
        indexed[turn] = scan
    cohorts = {name: dict(opportunities=0, resolved=0, baselineRayHits=0, candidateRayHits=0,
                          currentHeadingProxyHits=0, masks=Counter())
               for name in ('all', 'supported', 'unsupported', 'candidateEligible', 'candidateHold',
                            'baselineEligible', 'bothEligible')}
    seen = set()
    per_round, held = Counter(), Counter()
    for line in (directory / 'shadow.jsonl').read_bytes().splitlines():
        op = json.loads(line)
        if any(type(op.get(k)) is not int for k in ('round', 'commandTurn', 'creationTurn', 'lastScanTurn',
                                                    'baselineGun', 'support', 'actualSupport', 'endTurn', 'hits')):
            raise ValueError('Invalid opportunity metadata')
        rnd, q = op['round'], op['commandTurn']
        if (not 1 <= rnd <= len(metrics) or q < 0 or q > metrics[rnd - 1]['turns'] or (rnd, q) in seen
                or op['creationTurn'] != q + 1 or op['lastScanTurn'] not in (q, q - 1)
                or op['support'] not in range(4) or op['actualSupport'] not in range(4) or op['baselineGun'] not in range(3)):
            raise ValueError('Duplicate or mistimed opportunity')
        seen.add((rnd, q))
        per_round[rnd] += 1
        if any(type(op.get(k)) not in (int, float) or not math.isfinite(op[k])
               for k in ('x', 'y', 'targetX', 'targetY', 'targetSpeed', 'targetHeading', 'width', 'height',
                         'power', 'baselineAim', 'candidateAim', 'currentGun')):
            raise ValueError('Nonfinite opportunity geometry')
        if (op['width'], op['height']) != (800, 600) or not .1 <= op['power'] <= 2:
            raise ValueError('Wrong arena or power')
        _, scans = streams[rnd - 1]
        scan = scans.get(op['lastScanTurn'])
        if scan is None or any(op['target' + k.capitalize()] != scan[k] for k in ('x', 'y', 'heading', 'speed')):
            raise ValueError('Opportunity differs from source scan')
        half = math.degrees(math.asin(18 / max(18, math.hypot(op['targetX'] - op['x'], op['targetY'] - op['y']))))
        def aligned(key):
            return abs((op[key] - op['currentGun'] + 180) % 360 - 180) <= .75 * half
        supported = op['support'] >= 2
        baseline = aligned('baselineAim')
        candidate = supported and op['actualSupport'] >= 2 and aligned('candidateAim')
        if op['baselineEligible'] is not baseline or op['candidateEligible'] is not candidate:
            raise ValueError('Opportunity alignment mismatch')
        held[rnd] += not candidate
        outcome = replay_opportunity(op, streams[rnd - 1], metrics[rnd - 1]['turns'])
        if (op['status'], op['endTurn'], op['hits']) != outcome:
            raise ValueError('Joint ray replay mismatch')
        groups = ['all', 'supported' if supported else 'unsupported',
                  'candidateEligible' if candidate else 'candidateHold']
        if baseline:
            groups.append('baselineEligible')
            if candidate:
                groups.append('bothEligible')
        for name in groups:
            group = cohorts[name]
            group['opportunities'] += 1
            group['masks'][op['status']] += 1
            if op['status'] == 'resolved':
                group['resolved'] += 1
                group['baselineRayHits'] += op['hits'] & 1
                group['candidateRayHits'] += (op['hits'] >> 1) & 1
                group['currentHeadingProxyHits'] += (op['hits'] >> 2) & 1
    for row in metrics:
        if per_round[row['round']] != row['envelopeOpportunities'] or held[row['round']] != row['envelopeHolds']:
            raise ValueError('Shadow opportunity count differs from sealed metrics')
    return cohorts


def validate(directory, game_id, seed, stage):
    transfer = stage == 'transfer'
    rounds = 5 if stage == 'shadow' else 20
    mode = 'shadow' if stage == 'shadow' else 'envelope'
    data = json.loads((directory / 'battle.json').read_bytes())
    own, share = official(data, transfer, rounds)
    rows = [json.loads(line) for line in (directory / 'metrics.jsonl').read_bytes().splitlines()]
    if len(rows) != rounds:
        raise ValueError('Incomplete telemetry')
    previous_wins = 0
    for i, r in enumerate(rows, 1):
        check_identity(r, seed, mode)
        check_counters(r)
        if (type(r['round']) is not int or r['round'] != i or not r['turns'] or type(r['won']) is not bool
                or type(r['died']) is not bool or (r['won'] and r['died'])):
            raise ValueError('Invalid round/terminal state')
        tally = r.get('officialFirstPlaces')
        delta = r.get('officialRoundWins')
        if (type(tally) is not int or type(delta) is not int or delta not in (0, 1)
                or tally != previous_wins + delta):
            raise ValueError('Missing or inconsistent official round wins')
        previous_wins = tally
        guns = ('linear', 'stopped', 'circular', 'envelope')
        if (r['unclassifiedShots'] or r['shotsFired'] != r['attackShots'] or r['bulletHits'] != r['attackHits']
                or r['attackShots'] != sum(r[g + 'Shots'] for g in guns)
                or r['attackHits'] != sum(r[g + 'Hits'] for g in guns)
                or not math.isclose(r['damageDealt'], r['attackDamage'], abs_tol=1e-6)
                or any(r[g + 'Hits'] > r[g + 'Shots'] for g in guns)):
            raise ValueError('Unreconciled shot outcomes')
        if (r['cwDecisions'] + r['ccwDecisions'] != r['decisions']
                or max(r[k] for k in ('waveDecisions', 'movingTargetTurns', 'observedMovingTurns', 'rangeObservedTurns')) > r['decisions']
                or sum(r[g + 'Decisions'] for g in guns[:3]) > r['decisions']):
            raise ValueError('Invalid movement/selection coverage')
        if (r['envelopeDecisions'] > r['decisions'] or r['envelopeOpportunities'] > r['envelopeDecisions']
                or r['envelopeHolds'] > r['envelopeOpportunities']
                or r['envelopeFireRequests'] > r['envelopeOpportunities'] - r['envelopeHolds']
                or r['envelopeShots'] > r['envelopeFireRequests']):
            raise ValueError('Invalid envelope coverage')
        if stage == 'shadow' and (r['envelopeFireRequests'] or r['envelopeShots']):
            raise ValueError('Challenger controlled a shadow shot')
        if stage != 'shadow' and r['envelopeShots'] != r['shotsFired']:
            raise ValueError('Competitive shot used a legacy ray')
        if not .1 * r['shotsFired'] - 1e-6 <= r['energySpent'] <= 2 * r['shotsFired'] + 1e-6:
            raise ValueError('Invalid shot energy')
    if previous_wins != own['firstPlaces']:
        raise ValueError('Official per-round wins disagree with battle')
    totals = {k: sum(r[k] for r in rows) for k in INTS + REALS}
    model = json.loads((directory / 'model.json').read_bytes())
    check_identity(model, seed, mode)
    check_counters(model['totals'])
    if type(model.get('officialFirstPlaces')) is not int or model['officialFirstPlaces'] != previous_wins:
        raise ValueError('Endpoint official wins disagree with round totals')
    if type(model['roundsCompleted']) is not int or model['roundsCompleted'] != rounds:
        raise ValueError('Incomplete model')
    for k, v in totals.items():
        actual = model['totals'][k]
        if actual != v and (k in INTS or not math.isclose(actual, v, abs_tol=1e-6)):
            raise ValueError('Unreconciled endpoint: ' + k)
    corrections = model['corrections']
    if (not isinstance(corrections, list) or len(corrections) > min(totals['matchedRays'], 32)
            or any(type(x) not in (int, float) or not -1 <= x <= 1 for x in corrections)):
        raise ValueError('Invalid danger memory')
    history = model['attackHistory']
    if (not isinstance(history, list) or len(history) > min(32, totals['attackShots'])
            or any(type(mask) is not int or mask not in range(8) for mask in history)):
        raise ValueError('Invalid gun history')
    counts = [sum(bool(mask & bit) for mask in history) for bit in (1, 2, 4)] + [len(history)]
    if model['attackScores'] != counts:
        raise ValueError('Gun scores differ from history')
    game = dict(gameId=game_id, seed=seed, rounds=rounds, wins=own['firstPlaces'], scoreShare=share,
                official=data, metrics=totals, endpointSHA256=benchmark.digest(directory / 'model.json'))
    game['wonFlags'] = sum(r['won'] for r in rows)
    game['winFlagDisagreements'] = [r['round'] for r in rows if int(r['won']) != r['officialRoundWins']]
    if stage != 'shadow' and any((directory / p).exists() for p in ('shadow.jsonl', 'shadow-scans.jsonl')):
        raise ValueError('Competitive game emitted shadow evidence')
    if stage == 'shadow':
        game['pairedShadow'] = paired_shadow(directory, rows)
        if totals['skippedTurns'] or game['pairedShadow']['all']['resolved'] == 0:
            raise ValueError('Invalid shadow diagnostic: own skip or no paired resolutions')
    if transfer:
        ais_rows = [json.loads(line) for line in (directory / 'ais.jsonl').read_bytes().splitlines()]
        benchmark.validate_metrics(ais_rows, 20, 'train', AIS_SEED, 0)
        path = directory / 'ais-model.dat'
        diagnostic = compare.checkpoint_diagnostics(path)
        if diagnostic['trainingRounds'] != 20:
            raise ValueError('Incomplete AIS endpoint')
        game.update(aisSeed=AIS_SEED, aisMetrics=benchmark.summarize_metrics(ais_rows),
                    aisWonFlags=sum(row['won'] for row in ais_rows),
                    aisEndpointSHA256=benchmark.digest(path), aisEndpoint=diagnostic)
    return game


def preflight(freeze, runtime):
    if not re.fullmatch('[0-9a-f]{40}', freeze['source']):
        raise ValueError('Invalid source SHA')
    changed = v1.git('diff', '--name-only', freeze['source'], '--', 'experiments', 'tools', '.github/workflows',
                     'src', 'AISurvivor', 'gradle', 'gradlew', 'build.gradle', 'settings.gradle').splitlines()
    if any(p != 'tools/breakwater-v3-freeze.json' for p in changed):
        raise ValueError('Executable drift since freeze')
    if v1.git('diff', '--name-only', BASE, '--', 'AISurvivor', 'src', 'gradle', 'build.gradle', 'settings.gradle',
              'tools/BenchmarkBattle.java', 'tools/strong_melee_opponents.py', 'tools/breakwater_experiment.py',
              'tools/breakwater_mobile_experiment.py', 'tools/benchmark.py', 'tools/compare.py', 'tools/retcon.py', 'tools/opponents/strong-melee.lock.json'):
        raise ValueError('Frozen dependencies changed')
    for path, key in ((PLAN, 'planSha256'), (AIS / 'AISurvivor.jar', 'productionJarSHA256'),
                      (BOT / 'Breakwater.jar', 'jarSha256'), (native.LOCK, 'lockSha256')):
        if benchmark.digest(path) != freeze[key]:
            raise ValueError('Frozen hash mismatch: ' + key)
    previous = json.loads(v2.FREEZE.read_bytes())
    if freeze['runtime'] != previous['runtime'] or freeze['runtime'] != runtime or freeze['runnerSHA256'] != benchmark.ASSETS[v1.BATTLE]:
        raise ValueError('Unexpected runtime')
    if set(p.name for p in BUNDLE.iterdir()) != set(v1.FILES):
        raise ValueError('Unexpected Breakwater assets')
    for name in v1.FILES:
        source = BOT / name if name.endswith('.jar') else BOT / 'bot' / name
        if benchmark.digest(source) != benchmark.digest(BUNDLE / name):
            raise ValueError('Unfrozen Breakwater bundle')


def play(directory, stage, game_id, seed, runner, source, python):
    directory.mkdir()
    bot = directory / 'Breakwater'
    shutil.copytree(BUNDLE, bot)
    (bot / 'Breakwater.sh').chmod(0o755)
    env = {k: v for k, v in native.clean_environment(os.environ).items()
           if not k.startswith(('BREAKWATER_', 'CHALLENGER_'))}
    transfer = stage == 'transfer'
    if transfer:
        opponent = directory / 'AISurvivor'
        opponent.mkdir()
        for ext in ('jar', 'sh', 'cmd', 'json'):
            shutil.copy2(AIS / f'AISurvivor.{ext}', opponent)
        (opponent / 'AISurvivor.sh').chmod(0o755)
        staged = {p.name: benchmark.digest(p) for p in opponent.iterdir()}
        env.update(AISURVIVOR_MODE='train', AISURVIVOR_SEED=str(AIS_SEED),
                   AISURVIVOR_METRICS_PATH=str(directory / 'ais.jsonl'),
                   AISURVIVOR_MODEL_PATH=str(directory / 'ais-model.dat'))
    else:
        env.update(native.stage(source, directory, python))
        opponent = directory / 'opponents/diamond-port'
        benchmark.isolate_opponent(source / 'bots/ports/diamond-port', opponent)
        staged = native.verify_staged(directory, ['diamond-port'])
    v1.save(directory / 'staged-hashes.json', staged)
    env['BREAKWATER_GUN_MODE'] = 'shadow' if stage == 'shadow' else 'envelope'
    if stage == 'shadow':
        env['BREAKWATER_SHADOW_PATH'] = str(directory / 'shadow.jsonl')
        env['BREAKWATER_SHADOW_SCANS_PATH'] = str(directory / 'shadow-scans.jsonl')
    env.update(BREAKWATER_SEED=str(seed), BREAKWATER_MODEL_PATH=str(directory / 'model.json'),
               BREAKWATER_METRICS_PATH=str(directory / 'metrics.jsonl'), PYTHONDONTWRITEBYTECODE='1')
    java = Path(os.environ['JAVA_HOME']) / 'bin/java'
    env['PATH'] = str(java.parent) + os.pathsep + env.get('PATH', '')
    rounds = 5 if stage == 'shadow' else 20
    command = [str(java), '-cp', str(runner), str(ROOT / 'tools/BenchmarkBattle.java'), str(rounds), '900',
               str(directory / 'battle.json'), str(bot), str(opponent)]
    v1.save(directory / 'command.json', command)
    benchmark.run_process(command, ROOT, env, directory / 'runner.log', 990)
    if not transfer and native.verify_staged(directory, ['diamond-port']) != staged:
        raise ValueError('Opponent drifted during play')
    return validate(directory, game_id, seed, stage)


def run(output):
    output.mkdir(exist_ok=False, parents=True)
    report = dict(status='preflight', executionHead=v1.git('rev-parse', 'HEAD'), games=[])
    target = output / 'experiment.json'
    v1.save(target, report)
    try:
        freeze = json.loads(FREEZE.read_bytes())
        shutil.copyfile(FREEZE, output / 'freeze.json')
        shutil.copyfile(PLAN, output / 'registered-plan.md')
        report['runtime'] = v1.runtime()
        v1.save(target, report)
        preflight(freeze, report['runtime'])
        cache = output.parent / '.breakwater-v3-cache'
        source, python, provenance = native.prepare(cache, sys.executable)
        runner = benchmark.download(cache, v1.BATTLE, freeze['runnerSHA256'])
        report.update(status='running', pythonOpponents=provenance,
                      runnerSHA256=benchmark.digest(runner), productionJarSHA256=freeze['productionJarSHA256'])
        for stage in ('shadow', 'pilot', 'confirmation', 'transfer'):
            if stage == 'confirmation' and report['pilotDecision']['status'] != 'advance':
                continue
            cohort = []
            for gid, seed in layout(stage):
                report['attempting'] = gid
                v1.save(target, report)
                game = play(output / gid, stage, gid, seed, runner, source, python)
                cohort.append(game)
                report['games'].append(game)
                del report['attempting']
                v1.save(target, report)
            if stage in ('pilot', 'confirmation'):
                report[stage + 'Decision'] = decision(cohort, stage)
                report['diamondDecision'] = report[stage + 'Decision']
                v1.save(target, report)
        report['status'] = 'complete'
    except BaseException as exc:
        report.update(status='invalid', failure=repr(exc))
        raise
    finally:
        v1.save(target, report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output.resolve())
