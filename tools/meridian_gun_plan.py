"""Issue #90: frozen descriptive 12-game comparison. No resume or replacement games."""
import argparse
import copy
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import socket
import statistics
import struct
import subprocess

import benchmark
import compare
import meridian_duels as old
import retcon
import strong_melee_opponents as native
from meridian_profile import ROOT, OUT, CACHE

REG = ROOT / 'benchmarks/meridian-gun-plan'
EXEC = ROOT / 'local-comparison/execution'
BASE = '809b880cdcee98d7946d05e6bcd30f623f604348'
CANDIDATE_SOURCE = '5268264c49dd1150eb862b4cd59814a9e589abb3e5f284c56bf07dd07989d0d1'
CANDIDATE_JAR = '7667ef48974e67fe1e42ecee15005905bfe915c4c4afe7cba8e52c146ec2cafe'
RECEIPT = 'refs/tags/experiment-receipts/meridian-gp60-issue90'
SETUP = dict(gameType='1v1', arenaWidth=800, arenaHeight=600,
             minNumberOfParticipants=2, maxNumberOfParticipants=2, numberOfRounds=5,
             gunCoolingRate=.1, maxInactivityTurns=450, turnTimeout=100000,
             readyTimeout=10000000, defaultTurnsPerSecond=-1)


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.part')
    temporary.write_text(json.dumps(obj, indent=2, allow_nan=False, sort_keys=True) + '\n')
    temporary.replace(path)


def profile_expected(profile):
    result = copy.deepcopy(profile)
    result['assets']['experiments/Meridian/src/Meridian.java'] = CANDIDATE_SOURCE
    result['assets']['experiments/Meridian/Meridian.jar'] = CANDIDATE_JAR
    return result


def validate_profile(profile):
    profiles = [json.loads((ROOT / 'benchmarks/meridian-aim' / name).read_text())
                for name in ('profile.json', 'profile-20261026.json')]
    if profile not in [profile_expected(p) for p in profiles]:
        raise ValueError('Actual runtime/assets differ from both registered exact profiles')
    for name, expected in profile['assets'].items():
        if benchmark.digest(ROOT / name) != expected:
            raise ValueError('Artifact differs from profile: ' + name)


def package_contents(profile):
    """Hash all installed distribution files, independent of mutable venv roots."""
    python = CACHE / 'strong-python' / profile['python']['commit'] / 'venv/bin/python'
    code = '''import hashlib,importlib.metadata as m,json
result={}
for d in sorted(m.distributions(),key=lambda d:d.metadata['Name'].lower()):
 rows={}
 for f in sorted(d.files or [],key=str):
  p=d.locate_file(f)
  if p.is_file() and not str(f).endswith('.pyc') and '__pycache__' not in str(f):
   rows[str(f)]=hashlib.sha256(p.read_bytes()).hexdigest()
 result[d.metadata['Name']]={'version':d.version,'files':rows}
print(json.dumps(result,sort_keys=True))'''
    return json.loads(subprocess.check_output([str(python), '-I', '-c', code], text=True))


def preflight():
    validate_frozen_tools()
    profile = json.loads((OUT / 'profile.json').read_text())
    validate_profile(profile)
    save(OUT / 'gun-plan-package-contents.json', package_contents(profile))
    return profile


def validate_frozen_tools():
    manifest = json.loads((REG / 'tooling-sha256.json').read_text())
    required = {str(path.relative_to(ROOT)) for path in (
        Path(__file__), REG / 'games.json', REG / 'MeridianBattle.java',
        ROOT / 'tools/meridian_duels.py', ROOT / 'tools/meridian_profile.py',
        ROOT / 'tools/benchmark.py', ROOT / 'tools/compare.py',
        ROOT / 'tools/retcon_stats.py', ROOT / 'tools/verify_jar.py',
        ROOT / 'tools/retcon.py', ROOT / 'tools/strong_melee_opponents.py',
        ROOT / 'tools/meridian_gun_plan_activate.py',
        ROOT / 'tools/meridian_gun_plan_fixture.py',
        ROOT / '.github/workflows/meridian-gun-plan.yml',
        ROOT / 'benchmarks/meridian-aim/profile.json',
        ROOT / 'benchmarks/meridian-aim/profile-20261026.json',
        ROOT / 'tools/opponents/retcon.lock.json', ROOT / 'tools/opponents/strong-melee.lock.json')}
    if not required <= manifest.keys():
        raise ValueError('Incomplete frozen tooling manifest')
    for name, expected in manifest.items():
        path = ROOT / name
        if path.resolve().is_relative_to(ROOT.resolve()) is False or benchmark.digest(path) != expected:
            raise ValueError('Frozen tooling changed: ' + name)


def materialize(arm, profile):
    folder = EXEC / 'bundles' / arm
    (folder / 'bot').mkdir(parents=True)
    for name in ('Meridian.jar', 'bot/Meridian.json', 'bot/Meridian.sh', 'bot/Meridian.cmd'):
        target = folder / name
        key = 'experiments/Meridian/' + name
        raw = (subprocess.check_output(['git', 'show', BASE + ':' + key], cwd=ROOT)
               if arm == 'baseline' else (ROOT / key).read_bytes())
        expected = (json.loads((ROOT / 'benchmarks/meridian-aim/profile.json').read_text())['assets'][key]
                    if arm == 'baseline' else profile['assets'][key])
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError('Unregistered bundle: ' + arm + '/' + name)
        target.write_bytes(raw)
    (folder / 'bot/Meridian.sh').chmod(0o755)
    return folder


def validate_receipt(receipt, env, head):
    if (receipt.get('experiment') != 'meridian-gp60-issue90'
            or receipt.get('payloadSHA') != head or env.get('PAYLOAD_SHA') != head
            or receipt.get('runID') != env.get('GITHUB_RUN_ID')
            or str(receipt.get('runAttempt')) != '1' or env.get('GITHUB_RUN_ATTEMPT') != '1'
            or receipt.get('receiptRef') != RECEIPT
            or len(receipt.get('receiptObject', '')) != 40):
        raise ValueError('Missing or wrong durable activation receipt')


def round_metrics(ticks, uid):
    row = dict(turns=len(ticks), shots=0, hits=0, enemyShots=0, hitsTaken=0,
               damageDealt=0., damageTaken=0., shotEnergy=0., wallContacts=0,
               botContacts=0, commonOpportunities=0, noFireOpportunities=0,
               longestCommonOpportunityNoFireRun=0)
    streak = 0
    for index, tick in enumerate(ticks):
        for e in tick['events']:
            kind, b = e['type'], e.get('bullet', {})
            if 'Skipped' in kind:
                raise ValueError('Observer-reported skipped turn')
            if kind == 'BulletFiredEvent':
                if b['ownerId'] == uid:
                    row['shots'] += 1
                    row['shotEnergy'] += b['power']
                else:
                    row['enemyShots'] += 1
            elif kind == 'BulletHitBotEvent':
                if b['ownerId'] == uid:
                    row['hits'] += 1
                    row['damageDealt'] += e['damage']
                elif e['victimId'] == uid:
                    row['hitsTaken'] += 1
                    row['damageTaken'] += e['damage']
            elif kind == 'BotHitWallEvent' and e['victimId'] == uid:
                row['wallContacts'] += 1
            elif kind == 'BotHitBotEvent' and e['botId'] == uid:
                row['botContacts'] += 1
        own = next((b for b in tick['botStates'] if b['id'] == uid), None)
        scans = [e for e in tick['events'] if e['type'] == 'ScannedBotEvent' and e['scannedByBotId'] == uid]
        # Terminal decisions have no next-tick fire confirmation and are excluded.
        common = False
        if own is not None and scans and index + 1 < len(ticks):
            target = scans[-1]
            distance = math.hypot(target['x'] - own['x'], target['y'] - own['y'])
            energy = own['energy']
            power = max(.1, min(3, min(2.8 if distance < 300 else 1.65,
                                      energy / 12, max(.1, (target['energy'] + 2) / 6))))
            common = (own['gunHeat'] == 0 and energy > power
                      and (energy > 1.2 or target['energy'] <= .1))
        fired_next = index + 1 < len(ticks) and any(
            e['type'] == 'BulletFiredEvent' and e['bullet']['ownerId'] == uid
            for e in ticks[index + 1]['events'])
        row['commonOpportunities'] += int(common)
        held = common and not fired_next
        row['noFireOpportunities'] += int(held)
        streak = streak + 1 if held else 0
        row['longestCommonOpportunityNoFireRun'] = max(row['longestCommonOpportunityNoFireRun'], streak)
    row['hitRate'] = row['hits'] / row['shots'] if row['shots'] else None
    row['enemyHitRate'] = row['hitsTaken'] / row['enemyShots'] if row['enemyShots'] else None
    row['shotsPer1000ObserverTurns'] = 1000 * row['shots'] / row['turns'] if row['turns'] else None
    row['enemyShotsPer1000ObserverTurns'] = 1000 * row['enemyShots'] / row['turns'] if row['turns'] else None
    row['damagePer100Shots'] = 100 * row['damageDealt'] / row['shots'] if row['shots'] else None
    row['damagePer100EnemyShots'] = 100 * row['damageTaken'] / row['enemyShots'] if row['enemyShots'] else None
    return row


def metrics(path, game):
    row = old.metrics(path, game)
    recording, = path.glob('*.battle.gz')
    with gzip.open(recording, 'rt') as fd:
        events = [json.loads(line) for line in fd]
    json.dumps(events, allow_nan=False)
    if any('Skipped' in e['type'] for e in events):
        raise ValueError('Observer-reported skipped turn')
    setup = events[0]['gameSetup']
    if any(setup.get(key) != value for key, value in SETUP.items()):
        raise ValueError('Wrong exact setup')
    if row['skips'] or not row['shots'] or not row['enemyShots']:
        raise ValueError('Private skip or undefined shot ratio')
    uid = next(r['id'] for r in row['results'] if r['name'] == 'Meridian')
    rounds = []
    previous = {r['id']: dict(totalScore=0, firstPlaces=0) for r in row['results']}
    for number in range(1, 6):
        ticks = [e for e in events if e['type'] == 'TickEventForObserver' and e['roundNumber'] == number]
        result = round_metrics(ticks, uid)
        endpoint, = [e for e in events if e['type'] == 'RoundEndedEventForObserver' and e['roundNumber'] == number]
        scores = {r['id']: r for r in endpoint['results']}
        delta = {i: scores[i]['totalScore'] - previous[i]['totalScore'] for i in scores}
        if any(v < 0 for v in delta.values()):
            raise ValueError('Nonmonotonic official round scores')
        total = sum(delta.values())
        result.update(round=number, score=delta[uid], totalScore=total,
                      share=delta[uid] / total if total else None,
                      wins=scores[uid]['firstPlaces'] - previous[uid]['firstPlaces'])
        previous = scores
        rounds.append(result)
    if any(previous[r['id']]['totalScore'] != r['totalScore']
           or previous[r['id']]['firstPlaces'] != r['firstPlaces'] for r in row['results']):
        raise ValueError('Final official endpoint does not reconcile with round endpoints')
    row.update(perRound=rounds, hitRate=row['efficiency'], damagePer100Shots=100*row['damageDealt']/row['shots'],
               winRate=row['wins']/game['rounds'],
               enemyHitRate=row['hitsTaken']/row['enemyShots'],
               shotsPer1000ObserverTurns=1000*row['shots']/sum(row['turnsByRound']),
               enemyShotsPer1000ObserverTurns=1000*row['enemyShots']/sum(row['turnsByRound']),
               commonOpportunities=sum(r['commonOpportunities'] for r in rounds),
               noFireOpportunities=sum(r['noFireOpportunities'] for r in rounds),
               longestCommonOpportunityNoFireRun=max(r['longestCommonOpportunityNoFireRun'] for r in rounds))
    if game['enemy'] == 'AISurvivor':
        model = path / 'ais-model.dat'
        if compare.checkpoint_rounds(model) != 5:
            raise ValueError('AIS endpoint round count mismatch')
        compare.checkpoint_diagnostics(model)
        raw = model.read_bytes()
        if any(not math.isfinite(v) for v in struct.unpack('>' + str((len(raw)-28)//8) + 'd', raw[28:])):
            raise ValueError('Nonfinite AIS endpoint')
        ais = [json.loads(x) for x in (path / 'ais.jsonl').read_text().splitlines()]
        benchmark.validate_metrics(ais, 5, 'train', game['seed'], 0)
        row['aisEndpointSHA256'] = benchmark.digest(model)
    return row


def port_free(port):
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', port))


def play(game, profile, bundle):
    data, stage = EXEC / 'raw' / game['id'], EXEC / 'stages' / game['id']
    data.mkdir(parents=True)
    stage.mkdir(parents=True)
    home = stage / 'home'
    home.mkdir()
    own = stage / 'Meridian'
    shutil.copytree(bundle / 'bot', own)
    shutil.copy2(bundle / 'Meridian.jar', own / 'Meridian.jar')
    env = {k: os.environ[k] for k in ('PATH', 'JAVA_HOME', 'LANG') if k in os.environ}
    env.update(HOME=str(home), MERIDIAN_SEED=str(game['seed']),
               MERIDIAN_METRICS=str(data / 'markers.txt'), PYTHONDONTWRITEBYTECODE='1')
    if game['enemy'] == 'Retcon':
        src, api, rc = retcon.prepare(CACHE)
        other = stage / 'Retcon'
        benchmark.isolate_opponent(src, other)
        if retcon.source_hash(other) != rc['sourceSHA256']:
            raise ValueError('Retcon tree mismatch')
        env['RUMBLE_JAVA_CACHE'] = str(api)
    elif game['enemy'] == 'Diamond Port':
        base = CACHE / 'strong-python' / profile['python']['commit']
        src = base / 'source'
        env.update(native.stage(src, stage, base / 'venv/bin/python'))
        other = stage / 'opponents/diamond-port'
        benchmark.isolate_opponent(src / 'bots/ports/diamond-port', other)
        native.verify_staged(stage, ['diamond-port'])
    else:
        other = stage / 'AISurvivor'
        shutil.copytree(ROOT / 'AISurvivor', other)
        env.update(AISURVIVOR_MODEL_PATH=str(data / 'ais-model.dat'), AISURVIVOR_METRICS_PATH=str(data / 'ais.jsonl'),
                   AISURVIVOR_SEED=str(game['seed']), AISURVIVOR_MODE='train')
    command = [str(Path(env['JAVA_HOME']) / 'bin/java'), '-cp', str(CACHE / 'robocode-tankroyale-runner-1.4.0.jar'),
               str(REG / 'MeridianBattle.java'), str(game['port']), '5', str(data), str(own), str(other)]
    hashes = {str(f.relative_to(stage)): benchmark.digest(f) for f in stage.rglob('*') if f.is_file()}
    save(data / 'attempt.json', dict(game=game, command=command, environment=env, stagedSHA256=hashes,
         runtimeProfileSHA256=benchmark.digest(OUT / 'profile.json'), execution=subprocess.check_output(
             ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()))
    port_free(game['port'])
    benchmark.run_process(command, stage, env, data / 'runner.log', 1680)
    return metrics(data, game)


def summarize(rows):
    report = {}
    keys = ('share', 'wins', 'winRate', 'shots', 'hits', 'hitRate', 'damageDealt', 'damageTaken', 'shotEnergy',
            'damagePer100Shots', 'damagePer100EnemyShots', 'longestCommonOpportunityNoFireRun',
            'enemyHitRate', 'shotsPer1000ObserverTurns', 'enemyShotsPer1000ObserverTurns')
    for enemy in ('Diamond Port', 'Retcon', 'AISurvivor'):
        arms = {}
        for arm in ('baseline', 'candidate'):
            group = [r for r in rows if r['enemy'] == enemy and r['arm'] == arm]
            arms[arm] = {key: dict(mean=statistics.mean(r[key] for r in group),
                                  values=[r[key] for r in group],
                                  sampleSD=statistics.stdev(r[key] for r in group)) for key in keys}
        arms['candidateMinusBaseline'] = {key: arms['candidate'][key]['mean'] - arms['baseline'][key]['mean'] for key in keys}
        report[enemy] = arms
    return report


def check_frozen_packages(profile):
    if package_contents(profile) != json.loads((REG / 'package-contents.json').read_text()):
        raise ValueError('Installed package contents differ from reviewed preflight freeze')


def main(run=False, check_frozen=False):
    profile = preflight()
    if check_frozen or run:
        check_frozen_packages(profile)
    if not run:
        return
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    receipt = json.loads((OUT / 'activation.json').read_text())
    validate_receipt(receipt, os.environ, head)
    EXEC.mkdir(parents=True)  # fail closed on any prior root; no resume
    save(EXEC / 'activation.json', receipt)
    save(EXEC / 'profile.json', profile)
    save(EXEC / 'package-contents.json', package_contents(profile))
    bundles = {arm: materialize(arm, profile) for arm in ('baseline', 'candidate')}
    games = json.loads((REG / 'games.json').read_text())
    report = dict(status='pending', games=[], decision='no comparison decision')
    save(EXEC / 'decision.json', report)
    for game in games:
        try:
            row = dict(**game, status='valid', **play(game, profile, bundles[game['arm']]))
        except Exception as error:
            report['games'].append(dict(**game, status='invalid', error=str(error)))
            report.update(status='invalid', decision='incomplete; no comparison decision')
            save(EXEC / 'decision.json', report)
            raise
        report['games'].append(row)
        save(EXEC / 'decision.json', report)
    report.update(status='complete', decision='descriptive screen complete; no promotion',
                  comparison=summarize(report['games']))
    save(EXEC / 'decision.json', report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--run', action='store_true', help='Requires reviewed frozen package contents and durable activation receipt')
    mode.add_argument('--preflight', action='store_true', help='No games (default)')
    mode.add_argument('--check-frozen', action='store_true', help='Verify frozen package contents before activation; no games')
    args = parser.parse_args()
    main(args.run, args.check_frozen)
