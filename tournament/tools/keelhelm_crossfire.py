"""Frozen acquisition and exact decision for KeelHelm Crossfire, issue #138."""
import argparse
from collections import Counter
from fractions import Fraction
import gzip
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import zipfile

import keelhelm_risk as risk
from keelhelm_randomization import exact_tail, unrank

core = risk.core
resources = risk.resources
ROOT = risk.ROOT
CONTROL = 'KeelHelm'
CANDIDATE = 'KeelHelm Crossfire'
FOLDER = 'KeelHelmCrossfire'
SHA = 'fe6498d461ebc6c92e5d2d03e047c91771166c3d339e49ed647cf47966164d96'
SERVER_SHA = '16c277795bd823c9eda0b650f1d09a418ff35cfe43d59b820c52d5e6816f905c'
SCENARIOS = [('melee', 32, 5), ('Diamond Port', 4, 10), ('Retcon', 4, 10),
             ('Walls', 2, 5), ('SpinBot', 2, 5)]
COUNTS = re.compile(r'(\d+) crossfire requests=(\d+) accepted=(\d+) confirmed=(\d+) unknown=(\d+)')
BULLET = re.compile(r'(\d+) crossfireBullet turn=(\d+) bullet=(\d+) intendedTarget=(\d+) power=(\S+) direction=(\S+)')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def make_schedule(assignment):
    require(len(assignment) == len(set(assignment)) == 16 and set(assignment) <= set(range(32)),
            'Invalid balanced assignment')
    rng = random.Random(2026120101)
    rows = []
    pair = 0
    for scenario, count, rounds in SCENARIOS:
        melee = scenario == 'melee'
        names = risk.NAMES if melee else [CONTROL, scenario]
        x_range, y_range = ((100, 900), (100, 900)) if melee else ((80, 720), (80, 520))
        for layout in range(count):
            while True:
                points = [[rng.randint(*x_range), rng.randint(*y_range), rng.randrange(360)] for _ in names]
                if all(math.dist(a[:2], b[:2]) >= 250 for i, a in enumerate(points) for b in points[:i]):
                    break
            seat = layout % len(names)
            candidate_first = pair in assignment if melee else layout % 2 == 0
            arms = ('candidate', 'control') if candidate_first else ('control', 'candidate')
            for slot, arm in enumerate(arms):
                rows.append(dict(id=f'crossfire-{pair + 1:02d}-{slot + 1}-{arm}', pair=pair, slot=slot,
                                 arm=arm, mode='fixed', scenario=scenario, rounds=rounds,
                                 seed=2026120200 + pair, order=list(names[seat:]+names[:seat]),
                                 positions=dict(zip(names, points))))
            pair += 1
    return rows


def entrant_name(game):
    return CANDIDATE if game['arm'] == 'candidate' else CONTROL


def transport_game(game):
    name = entrant_name(game)
    order = [name if bot == CONTROL else bot for bot in game['order']]
    positions = {name if bot == CONTROL else bot: pos for bot, pos in game['positions'].items()}
    return dict(game, order=order, positions=positions)


def audit_alternates(recording, raw_rows, game):
    """Score only private confirmations that match the independent physical stream."""
    counts, traces, per_round = {}, {}, Counter()
    for row in raw_rows:
        if not (' crossfire ' in row or ' crossfireBullet ' in row):
            continue
        require(game['arm'] == 'candidate', 'Crossfire telemetry from control')
        if m := COUNTS.fullmatch(row):
            r, requested, accepted, confirmed, unknown = map(int, m.groups())
            require(r not in counts and requested >= accepted == confirmed + unknown,
                    'Invalid Crossfire count conservation')
            counts[r] = dict(requests=requested, accepted=accepted, confirmed=confirmed, unknown=unknown)
        else:
            m = BULLET.fullmatch(row)
            require(m is not None, 'Malformed Crossfire event')
            r, turn, bullet, target = map(int, m.groups()[:4])
            power, direction = map(float, m.groups()[4:])
            require((r, bullet) not in traces and turn > 0 and 0.1 <= power <= 3 and math.isfinite(direction),
                    'Invalid Crossfire event')
            traces[r, bullet] = (turn, target, power, direction)
            per_round[r] += 1
    if game['arm'] == 'candidate':
        require(sorted(counts) == list(range(1, game['rounds'] + 1)) and set(per_round) <= set(counts)
                and all(c['confirmed'] == per_round[r] for r, c in counts.items()),
                'Incomplete Crossfire telemetry')
        require(game['scenario'] == 'melee' or not any(c['requests'] for c in counts.values()),
                'Crossfire intervention in a start-to-end duel')

    names, matched, hits, energy, wins = {}, {}, {}, {}, {}
    previous_own = None
    unmatched_reasons = {}
    own = None
    spent, shots, hit_count, ties = Fraction(), 0, 0, 0
    with gzip.open(recording, 'rt', encoding='utf-8') as stream:
        for line in stream:
            event = json.loads(line)
            if event['type'] == 'GameStartedEventForObserver':
                names = {b['id']: b['name'] for b in event['participants']}
                own = next(key for key, value in names.items() if value == entrant_name(game))
            if event['type'] == 'RoundEndedEventForObserver':
                totals = {bot['id']: bot['firstPlaces'] for bot in event['results']}
                require(set(totals) <= set(names) and set(wins) <= set(totals), 'Round score roster changed')
                deltas = [total - wins.get(bot_id, 0) for bot_id, total in totals.items()]
                require(all(delta in (0, 1) for delta in deltas) and sum(deltas) >= 1, 'Invalid round first places')
                ties += sum(deltas) > 1
                wins = totals
            if event['type'] != 'TickEventForObserver':
                continue
            r, turn = event['roundNumber'], event['turnNumber']
            request_is_melee = (previous_own is not None and previous_own[:2] == (r, turn - 1)
                                and previous_own[2] is not None and previous_own[2] > 1)
            current_own = next((bot for bot in event['botStates'] if bot['id'] == own), None)
            for bot in event['botStates']:
                if bot['id'] == own:
                    energy.setdefault(r, [bot['energy'], bot['energy']])[1] = bot['energy']
            for entry in event['events']:
                bullet = entry.get('bullet')
                if not bullet or bullet.get('ownerId') != own:
                    continue
                key = r, bullet['bulletId']
                if entry['type'] == 'BulletFiredEvent':
                    shots += 1
                    spent += Fraction(str(bullet['power']))
                    if key not in traces:
                        continue
                    expected_turn, target, power, direction = traces[key]
                    require(key not in matched and target in names and target != own, 'Invalid alternate identity')
                    if (request_is_melee and expected_turn == entry['turnNumber'] == turn and power == bullet['power']
                            and abs(math.remainder(direction - bullet['direction'], 360)) <= 1e-6):
                        matched[key] = dict(round=r, turn=turn, bullet=key[1], intendedTarget=names[target],
                                            power=power, requestEnemyCount=previous_own[2])
                    else:
                        unmatched_reasons[key] = ('missing-or-nonmelee-request-state' if not request_is_melee
                                                  else 'physical-turn-power-ray-mismatch')
                elif entry['type'] == 'BulletHitBotEvent':
                    hit_count += 1
                    if key in traces:
                        hits.setdefault(key, []).append(dict(actualTarget=names[entry['victimId']],
                                                             damage=entry['damage'], turn=entry.get('turnNumber', turn)))
            previous_own = ((r, turn, current_own.get('enemyCount')) if current_own is not None else None)
    return dict(byRound=counts, matched=[dict(row, hitEvents=hits.get(key, []),
                              actualTargetAmbiguous=len(hits.get(key, [])) > 1,
                              **({k: hits[key][0][k] for k in ('actualTarget', 'damage')}
                                 if len(hits.get(key, [])) == 1 else {}))
                         for key, row in sorted(matched.items())],
                unmatchedCallback=[list(key) for key in sorted(traces.keys() - matched.keys())],
                unmatchedCallbackDetails=[dict(round=key[0], bullet=key[1],
                                               reason=unmatched_reasons.get(key, 'no-physical-fire'))
                                          for key in sorted(traces.keys() - matched.keys())],
                ownShots=shots, ownBulletHits=hit_count, fireEnergy=float(spent), visibleRoundEnergy=energy,
                firstPlaceTieRounds=ties, officialSnapshotFirstPlaces=wins)


def share(game, row, component):
    denominator = sum(bot[component] for bot in game['official']['results'])
    return Fraction(row[component], denominator) if denominator else Fraction()


def summarize(games, assignment):
    expected = make_schedule(assignment)
    require(len(games) == len(expected), 'Incomplete comparison')
    for game, planned in zip(games, expected):
        require(game['status'] == 'complete' and all(game.get(k) == v for k, v in planned.items()),
                'Comparison rows do not match frozen schedule')

    groups, means, result = {}, {}, {}
    for scenario, count, rounds in SCENARIOS:
        result[scenario] = {}
        for arm in ('control', 'candidate'):
            group = [g for g in games if g['scenario'] == scenario and g['arm'] == arm]
            groups[scenario, arm] = group
            mean = sum(share(g, g['helm'], 'totalScore') for g in group) / count
            means[scenario, arm] = mean
            result[scenario][arm] = dict(meanShare=float(mean), meanShareExact=str(mean),
                                         officialFirstPlaces=sum(g['helm']['firstPlaces'] for g in group),
                                         firstPlaceTieRounds=sum(g['firstPlaceTieRounds'] for g in group))
        result[scenario]['deltaPP'] = float(100 * (means[scenario, 'candidate'] - means[scenario, 'control']))

    controls, candidates = groups['melee', 'control'], groups['melee', 'candidate']
    diamond = [next(row for row in g['official']['results'] if row['name'] == 'Diamond Port') for g in candidates]
    diamond_share = sum(share(g, row, 'totalScore') for g, row in zip(candidates, diamond)) / len(candidates)
    diamond_beaten = sum(g['helm']['totalScore'] > row['totalScore'] for g, row in zip(candidates, diamond))
    unique_wins = sum(all(g['helm']['totalScore'] > row['totalScore'] for row in g['official']['results']
                          if row['name'] != CANDIDATE) for g in candidates)
    scores = Counter()
    for game in candidates:
        for row in game['official']['results']:
            scores[row['name']] += row['totalScore']

    survival = [sum(share(g, g['helm'], 'survival') for g in arm) / len(arm) for arm in (controls, candidates)]
    damage = [Fraction(sum(g['helm']['bulletDamage'] for g in arm), sum(g['rounds'] for g in arm))
              for arm in (controls, candidates)]
    exposure = [len(g['crossfire']['matched']) for g in candidates]
    slot_contrasts = [share(a, a['helm'], 'totalScore') - share(b, b['helm'], 'totalScore')
                      for a, b in zip(games[:64:2], games[1:64:2])]
    randomization = exact_tail(slot_contrasts, assignment)
    duel = groups['Diamond Port', 'candidate']
    gates = dict(
        diamondDuelShare=means['Diamond Port', 'candidate'] > Fraction(1, 2),
        diamondDuelFirstPlaces=sum(g['helm']['firstPlaces'] for g in duel) > 20,
        diamondDuelMajorityGames=sum(share(g, g['helm'], 'totalScore') > Fraction(1, 2) for g in duel) >= 3,
        retconShare=means['Retcon', 'candidate'] > Fraction(1, 2),
        meleeDiamondMean=means['melee', 'candidate'] > diamond_share,
        meleeDiamondGames=diamond_beaten >= 17,
        meleeAggregateFirst=all(scores[CANDIDATE] > score for name, score in scores.items() if name != CANDIDATE),
        meleeUniqueFirstGames=unique_wins >= 17,
        meleeGain=means['melee', 'candidate'] - means['melee', 'control'] >= Fraction(1, 50),
        randomization=Fraction(randomization['pExact']) <= Fraction(1, 20),
        meleeDamage=damage[1] > damage[0],
        meleeSurvival=survival[1] - survival[0] >= -Fraction(3, 100),
        alternateShots=sum(exposure) >= 64, alternateGames=sum(n > 0 for n in exposure) >= 8,
    )
    for scenario, _, _ in SCENARIOS[1:]:
        gates[f'{scenario} noninferiority'] = means[scenario, 'candidate'] - means[scenario, 'control'] >= -Fraction(1, 20)
    return dict(scenarios=result, gates=gates, randomization=randomization,
                meleeSlotContrasts=[str(v) for v in slot_contrasts], meleeDiamondShare=float(diamond_share),
                meleeDiamondGames=diamond_beaten, meleeUniqueFirstGames=unique_wins, meleeAggregateScores=dict(scores),
                meleeSurvivalShare=list(map(float, survival)), meleeDamagePerRound=list(map(float, damage)),
                alternateConfirmed=sum(exposure), alternateGames=sum(n > 0 for n in exposure),
                decision='pass-registered-comparison' if all(gates.values()) else 'reject-candidate')


def read_assignment(path, origin):
    draw = risk.read_assignment(path, origin)
    require(draw['study'] == 'keelhelm-crossfire-138', 'Assignment is not registered for Crossfire')
    return draw


def prepare(args):
    """Zero-game staging and hashing. The operator supplies the single persisted draw."""
    source, out, archive = args.source.resolve(), args.evidence.resolve(), args.archive.resolve()
    require(archive.is_dir() and shutil.disk_usage(archive).free >= 2 * 1024 ** 3,
            'Durable archive missing or below admission floor')
    java17 = subprocess.check_output([str(args.java17.absolute()), '-version'], stderr=subprocess.STDOUT, text=True)
    build_log = args.build_log.read_text(encoding='utf-8')
    require('version "17.' in java17 and core.sha(args.rebuilt_jar) == SHA
            and re.search(r'^> Task :test\s*$', build_log, re.MULTILINE)
            and 'BUILD SUCCESSFUL' in build_log and 'BUILD FAILED' not in build_log,
            'Clean Java 17 build, tests and rebuilt/committed JAR agreement required')
    out.mkdir(parents=True, exist_ok=False)
    project = ROOT / 'experiments' / FOLDER
    require(core.sha(project / (FOLDER + '.jar')) == SHA, 'Candidate JAR does not match')
    require(core.sha(ROOT / 'experiments/KeelHelm/KeelHelm.jar') == risk.BASE, 'Control JAR does not match')
    shutil.copy2(args.build_log, out / 'java17-build.log')
    core.save(out / 'build-parity.json', dict(runtime=java17, rebuiltSHA256=SHA, committedSHA256=SHA,
                                             buildLogSHA256=core.sha(out / 'java17-build.log')))
    shutil.copytree(ROOT / 'experiments/KeelHelm', out / 'control-source',
                    ignore=shutil.ignore_patterns('build', '.gradle', '__pycache__'))
    draw = read_assignment(args.assignment, args.assignment_source)
    shutil.copy2(args.assignment, out / 'assignment.json')
    shutil.copy2(args.assignment_source, out / 'assignment-origin-source.py')
    rows = make_schedule(draw['candidateFirst'])
    core.save(out / 'schedule.json', rows)
    server = out / 'server.jar'
    with zipfile.ZipFile(source / '.benchmark-cache/robocode-tankroyale-runner-1.4.0.jar') as runner:
        server.write_bytes(runner.read('robocode-tankroyale-server.jar'))
    require(core.sha(server) == SERVER_SHA, 'Unexpected embedded server hash')
    extras = [ROOT / path for path in [
        'tournament/tools/keelhelm_crossfire.py', 'tournament/tools/test_keelhelm_crossfire.py',
        'tournament/tools/keelhelm_risk.py', 'tournament/tools/keelhelm_calibration.py',
        'tournament/tools/keelhelm_randomization.py', 'tournament/tools/paired_transport.py',
        'tournament/scripts/PairedBattle.java', 'tools/keelhelm_resources.py',
        'docs/improvements/keelhelm-crossfire-protocol.md']]
    extras += [server, out / 'assignment.json', out / 'assignment-origin-source.py', out / 'schedule.json',
               out / 'java17-build.log', out / 'build-parity.json']
    freeze = core.collect(source, out, args.pins.resolve(), args.java.absolute(), args.python.absolute(),
                          control_jar_sha=risk.BASE, schedule_rows=rows, extra_paths=extras, candidate_project=project)
    freeze.update(candidateFolder=FOLDER, serverJar=str(server), assignment=draw, archive=str(archive),
                  benchmark=dict(name='keelhelm-crossfire-138', games=88, rounds=520, concurrency=1))
    # Stage one of every scenario/arm pair first without starting the booter.
    for game in (next(g for g in rows if g['scenario'] == name and g['arm'] == arm)
                 for name, _, _ in SCENARIOS for arm in ('control', 'candidate')):
        folders, identities = risk.stage(game, freeze, out, out / 'preflight' / game['id'])
        verify_staging(game, folders, identities)
    core.save(out / 'freeze.json', freeze)
    print('PREPARED', core.sha(out / 'freeze.json'), flush=True)


def verify_staging(game, folders, identities):
    own, name = entrant_name(game), FOLDER if game['arm'] == 'candidate' else CONTROL
    jar_sha, version = (SHA, '0.1.0') if game['arm'] == 'candidate' else (risk.BASE, '0.2.1')
    require(len(set(folders)) == len(identities) == len(game['order'])
            and (own, version) in identities, 'Staged roster mismatch')
    folder = next(Path(folder) for folder in folders if Path(folder).name == name)
    require(core.sha(folder / f'{name}.jar') == jar_sha, 'Staged entrant JAR mismatch')


def observed_rounds(run):
    rounds = set()
    try:
        for recording in (run / 'recordings').glob('*.gz'):
            with gzip.open(recording, 'rt', encoding='utf-8') as stream:
                for line in stream:
                    item = json.loads(line)
                    if item['type'] == 'RoundEndedEventForObserver':
                        rounds.add(item['roundNumber'])
    except (OSError, EOFError, ValueError):
        pass  # Raw prefix retained; this is only the count of visible complete endpoints.
    return len(rounds)


def play(game, freeze, out, monitor):
    row = dict(game, status='setup-failed')
    run = out / 'games' / game['id']
    try:
        run.mkdir(parents=True)
        folders, identities = risk.stage(game, freeze, out, run)
        verify_staging(game, folders, identities)
        (run / 'home').mkdir()
        (run / 'recordings').mkdir()
        java = Path(freeze['java'])
        env = {k: os.environ[k] for k in ('TMPDIR', 'LANG', 'LC_ALL', 'TZ') if k in os.environ}
        env.update(HOME=str(run / 'home'), JAVA_HOME=str(java.parent.parent),
                   PATH=str(java.parent) + ':/usr/bin:/bin:/opt/homebrew/bin',
                   ROBOCODE_PYTHON_BIN=freeze['python'], PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
                   ROBOCODE_ENV_FILE=str(run / 'absent.env'), ROBOCODE_GUN_ENV_FILE=str(run / 'absent-guns.env'),
                   RUMBLE_JAVA_CACHE=str(Path(freeze['source']) / '.benchmark-cache/retcon-api'),
                   KEEL_HELM_METRICS=str(run / 'helm-metrics.txt'), KEEL_HELM_SEED=str(game['seed']),
                   KEEL_SEED=str(game['seed']), PRISM_SEED=str(game['seed']))
        native = transport_game(game)
        config = dict(run=str(run), java=freeze['java'], serverJar=freeze['serverJar'], rounds=game['rounds'],
                      runner=str(Path(freeze['source']) / '.benchmark-cache/robocode-tankroyale-runner-1.4.0.jar'),
                      adapter=str(ROOT / 'tournament/scripts/PairedBattle.java'), bots=folders,
                      identities=dict(identities), positions=risk.native_game(native)['positions'])
        core.save(run / 'config.json', config)
        cmd = [freeze['python'], str(ROOT / 'tournament/tools/paired_transport.py'), str(run / 'config.json')]
        core.save(run / 'invocation.json', dict(command=cmd, environment=env))
        row['status'] = 'attempted'

        def check():
            if monitor.error:
                raise ValueError(monitor.error)

        core.execute(cmd, env, run, monitor.register, failure_check=check)
        official = json.loads((run / 'battle.json').read_text())
        require(official['numberOfRounds'] == game['rounds']
                and sorted((b['name'], b['version']) for b in official['results']) == sorted(identities),
                'Official roster or rounds differ')
        recordings = list((run / 'recordings').glob('*.gz'))
        require(len(recordings) == 1, 'Expected one complete recording')
        recording = recordings[0]
        audit = core.audit(recording, official, game, identities, own_name=entrant_name(game))
        telemetry = core.telemetry(run / 'helm-metrics.txt', game['rounds'])
        transport = risk.inspect_transport(run / 'transport.json', native, identities)
        start = risk.starts(recording, native)
        crossfire = audit_alternates(recording, telemetry, game)
        require({b['id']: b['firstPlaces'] for b in official['results']} == crossfire['officialSnapshotFirstPlaces'],
                'Final first places disagree with round snapshots')
        own = next(b for b in official['results'] if b['name'] == entrant_name(game))
        core.save(run / 'starts.json', start)
        row.update(status='complete', official=official, helm=own, audit=audit, crossfire=crossfire,
                   firstPlaceTieRounds=crossfire['firstPlaceTieRounds'], transportAudit=transport,
                   recording=str(recording.relative_to(out)))
    except BaseException as error:
        row['error'] = repr(error)
    if run.exists():
        row['observedCompletedRounds'] = game['rounds'] if row['status'] == 'complete' else observed_rounds(run)
        core.save(run / 'result.json', row)
    return row


def validate_approval(approval, out, freeze_sha, freeze):
    approved = json.loads(approval.read_text())
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    require(approved['head'] == head == freeze['parent'] and approved['freezeSHA256'] == freeze_sha
            and approved['study'] == freeze['benchmark']['name'] and freeze['evidence'] == str(out)
            and re.fullmatch(r'https://github\.com/stephanelee/robocode/pull/139#(?:issuecomment-|pullrequestreview-)\d+',
                             approved['reviewURL']),
            'Separate exact-head premeasurement approval required')
    return approved


def run(args):
    out = args.evidence.resolve()
    manifest = out / 'freeze.json'
    freeze_sha = core.sha(manifest)
    freeze = json.loads(manifest.read_text())
    approved = validate_approval(args.approval, out, freeze_sha, freeze)
    assignment = freeze['assignment']['candidateFirst']
    require(freeze['schedule'] == make_schedule(assignment), 'Schedule differs from the frozen assignment')
    require(not (out / 'results.json').exists(), 'Evidence directory already attempted')
    core.verify_inputs(freeze)
    durable = Path(freeze['archive'])
    require(durable.is_dir() and shutil.disk_usage(durable).free >= 2 * 1024 ** 3,
            'Durable archive below admission floor')
    receipt = durable / 'keelhelm-crossfire-138-acquisition.json'
    with receipt.open('x', encoding='utf-8') as stream:
        json.dump(approved, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    shutil.copy2(receipt, out / 'acquisition-receipt.json')
    report = dict(status='running', freezeSHA256=freeze_sha, plannedGames=88, plannedRounds=520, games=[],
                  unplayed=[g['id'] for g in freeze['schedule']])
    try:
        core.save(out / 'results.json', report)
        for game in freeze['schedule']:
            core.verify_inputs(freeze)
            admission = resources.read_resources()
            admission['diskFreeBytes'] = shutil.disk_usage(out).free
            core.save(out / f"{game['id']}-admission.json", admission)
            require(admission['availablePercent'] >= 35 and admission['diskFreeBytes'] >= 2 * 1024 ** 3,
                    'Per-game admission failed')
            monitor = resources.Monitor(out / f"{game['id']}-resources.jsonl")
            monitor.start()
            try:
                result = play(game, freeze, out, monitor)
            finally:
                monitor.close()
            report['unplayed'].pop(0)
            result.update(resourceSamples=monitor.samples, resourceError=monitor.error)
            report['games'].append(result)
            core.save(out / 'results.json', report)
            require(result['status'] == 'complete' and not monitor.error, str(result.get('error') or monitor.error))
        report.update(status='complete', summary=summarize(report['games'], assignment))
    except BaseException as error:
        report.update(status='invalid', error=repr(error))
        raise
    finally:
        report['observedCompletedRounds'] = sum(g.get('observedCompletedRounds', 0) for g in report['games'])
        core.save(out / 'results.json', report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='action', required=True)
    staging = subparsers.add_parser('prepare')
    for option in ('source', 'evidence', 'archive', 'pins', 'java', 'python', 'assignment', 'assignment-source',
                   'java17', 'rebuilt-jar', 'build-log'):
        staging.add_argument('--' + option, type=Path, required=True)
    acquisition = subparsers.add_parser('run')
    for option in ('evidence', 'approval'):
        acquisition.add_argument('--' + option, type=Path, required=True)
    args = parser.parse_args()
    (prepare if args.action == 'prepare' else run)(args)


if __name__ == '__main__':
    main()
