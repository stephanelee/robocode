"""Strict official-observer and explicit-own-DTO reduction for registered Peak v1."""
import gzip
import json
import math
from pathlib import Path
import statistics
import struct

import benchmark
import compare

ROOT = benchmark.ROOT
REG = ROOT / 'benchmarks/meridian-peak-v1'
OUT = ROOT / 'build/meridian-peak'
CACHE = ROOT / '.benchmark-cache'
SETUP = dict(gameType='1v1', arenaWidth=800, arenaHeight=600, minNumberOfParticipants=2,
             maxNumberOfParticipants=2, gunCoolingRate=.1, maxInactivityTurns=450,
             turnTimeout=100000, readyTimeout=10000000, defaultTurnsPerSecond=-1)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.part')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def finite(value):
    if isinstance(value, dict):
        for item in value.values():
            finite(item)
    elif isinstance(value, list):
        for item in value:
            finite(item)
    elif isinstance(value, (int, float)) and not isinstance(value, bool) and not math.isfinite(value):
        raise ValueError('Nonfinite private or official evidence')


def integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError('Invalid integer counter: ' + name)
    return value


def counters(ticks, uid):
    row = dict(turns=len(ticks), shots=0, hits=0, enemyShots=0, hitsTaken=0,
               damageDealt=0., damageTaken=0., shotEnergy=0., wallContacts=0, botContacts=0, skips=0,
               callbackDeliverableOutgoingHits=0, callbackDeliverableOutgoingDamage=0.,
               observerOnlyOutgoingHits=0, observerOnlyOutgoingDamage=0.)
    for tick in ticks:
        own_state_present = any(bot['id'] == uid for bot in tick['botStates'])
        for event in tick['events']:
            kind, bullet = event['type'], event.get('bullet', {})
            if 'Skipped' in kind:
                raise ValueError('Observer-reported skip')
            if kind == 'BulletFiredEvent':
                if bullet['ownerId'] == uid:
                    row['shots'] += 1
                    row['shotEnergy'] += bullet['power']
                else:
                    row['enemyShots'] += 1
            elif kind == 'BulletHitBotEvent':
                if bullet['ownerId'] == uid:
                    row['hits'] += 1
                    row['damageDealt'] += event['damage']
                    if own_state_present:
                        row['callbackDeliverableOutgoingHits'] += 1
                        row['callbackDeliverableOutgoingDamage'] += event['damage']
                    else:
                        row['observerOnlyOutgoingHits'] += 1
                        row['observerOnlyOutgoingDamage'] += event['damage']
                elif event['victimId'] == uid:
                    row['hitsTaken'] += 1
                    row['damageTaken'] += event['damage']
            elif kind == 'BotHitWallEvent' and event['victimId'] == uid:
                row['wallContacts'] += 1
            elif kind == 'BotHitBotEvent' and event['botId'] == uid:
                row['botContacts'] += 1
    return rates(row)


def rates(row):
    row['hitRate'] = row['hits'] / row['shots'] if row['shots'] else None
    row['efficiency'] = row['hitRate']
    row['enemyHitRate'] = row['hitsTaken'] / row['enemyShots'] if row['enemyShots'] else None
    row['damagePer100Shots'] = 100 * row['damageDealt'] / row['shots'] if row['shots'] else None
    row['damagePer100EnemyShots'] = 100 * row['damageTaken'] / row['enemyShots'] if row['enemyShots'] else None
    row['shotsPer1000ObserverTurns'] = 1000 * row['shots'] / row['turns'] if row['turns'] else None
    row['enemyShotsPer1000ObserverTurns'] = 1000 * row['enemyShots'] / row['turns'] if row['turns'] else None
    return row


def validate_ais(path, game):
    model = path / 'ais-model.dat'
    if compare.checkpoint_rounds(model) != game['rounds']:
        raise ValueError('AIS endpoint round count mismatch')
    compare.checkpoint_diagnostics(model)
    raw = model.read_bytes()
    if any(not math.isfinite(v) for v in struct.unpack('>' + str((len(raw)-28)//8) + 'd', raw[28:])):
        raise ValueError('AIS nonfinite endpoint')
    records = [json.loads(line) for line in (path / 'ais.jsonl').read_text().splitlines()]
    benchmark.validate_metrics(records, game['rounds'], 'train', game['seed'], 0)
    return dict(sha256=benchmark.digest(model), rounds=game['rounds'])


def metrics(path, game, *, validate_private=True):
    recordings = list(path.glob('*.battle.gz'))
    if len(recordings) != 1:
        raise ValueError('Expected one complete official recording')
    with gzip.open(recordings[0], 'rt') as fd:
        events = [json.loads(line) for line in fd]
    finite(events)
    if not events or events[0]['type'] != 'GameStartedEventForObserver' or events[-1]['type'] != 'GameEndedEventForObserver':
        raise ValueError('Missing game boundaries')
    setup = events[0]['gameSetup']
    if any(setup.get(key) != value for key, value in dict(SETUP, numberOfRounds=game['rounds']).items()):
        raise ValueError('Wrong exact setup')
    if events[-1]['numberOfRounds'] != game['rounds']:
        raise ValueError('Incomplete official game')
    results = events[-1]['results']
    identities = sorted((r['name'], r['version']) for r in results)
    expected = sorted([(game['ownName'], game['ownVersion']), (game['enemy'], game['version'])])
    if identities != expected or sorted((p['name'], p['version']) for p in events[0]['participants']) != expected:
        raise ValueError('Wrong participant identity/version')
    me = next(r for r in results if r['name'] == game['ownName'])
    uid = me['id']
    markers = (path / 'markers.txt').read_text().splitlines()
    numbers = list(range(1, game['rounds'] + 1))
    for tag in ('start', 'end'):
        if [int(m.split()[0]) for m in markers if m.endswith(' ' + tag)] != numbers:
            raise ValueError('Incomplete/duplicate private round markers')
    if any(' skip ' in line for line in markers) or any('Skipped' in e['type'] for e in events):
        raise ValueError('Skipped turn')
    endpoints = [e for e in events if e['type'] == 'RoundEndedEventForObserver']
    if [e['roundNumber'] for e in endpoints] != numbers:
        raise ValueError('Incomplete official round endpoints')
    per_round, all_ticks = [], []
    previous = {r['id']: dict(totalScore=0, firstPlaces=0) for r in results}
    for number, endpoint in zip(numbers, endpoints):
        ticks = [e for e in events if e['type'] == 'TickEventForObserver' and e['roundNumber'] == number]
        if [e['turnNumber'] for e in ticks] != list(range(1, endpoint['turnNumber'] + 1)):
            raise ValueError('Missing or duplicate official ticks')
        scores = {r['id']: r for r in endpoint['results']}
        if set(scores) != set(previous):
            raise ValueError('Round identities changed')
        delta = {key: scores[key]['totalScore'] - previous[key]['totalScore'] for key in scores}
        if min(delta.values()) < 0:
            raise ValueError('Nonmonotonic round score')
        total = sum(delta.values())
        row = counters(ticks, uid)
        row.update(round=number, score=delta[uid], totalScore=total, share=delta[uid]/total if total else None,
                   wins=scores[uid]['firstPlaces'] - previous[uid]['firstPlaces'])
        per_round.append(row)
        all_ticks.extend(ticks)
        previous = scores
    if any(previous[r['id']]['totalScore'] != r['totalScore'] or previous[r['id']]['firstPlaces'] != r['firstPlaces'] for r in results):
        raise ValueError('Round/final official endpoint mismatch')
    row = counters(all_ticks, uid)
    if not row['shots'] or not row['enemyShots']:
        raise ValueError('Undefined required own/enemy shot denominator')
    total = sum(r['totalScore'] for r in results)
    if total <= 0:
        raise ValueError('Undefined official score share')
    row.update(share=me['totalScore']/total, wins=me['firstPlaces'], winRate=me['firstPlaces']/game['rounds'],
               perRound=per_round, results=results, turnsByRound=[r['turns'] for r in per_round],
               recordingSHA256=benchmark.digest(recordings[0]))
    if game['enemy'] == 'AISurvivor':
        row['aisEndpoint'] = validate_ais(path, game)
    if game['ownName'] == 'MeridianPeak' and validate_private:
        row['privateState'] = validate_peak_state(path, game, markers, row)
    else:
        row['privateState'] = dict(available=False, limitation='Original v3 has markers only; no model/CPU snapshot instrumentation')
    return row


def validate_peak_state(path, game, markers, official):
    # Exact DTO validator is completed against the bot agent's frozen explicit schema.
    from meridian_peak_state import validate
    return validate(path / 'state', game, markers, official)


def passes_screen(rows):
    if len(rows) != 5 or any(r.get('status') != 'valid' or r['skips'] != 0 or r['shots'] <= 0 or r['enemyShots'] <= 0 for r in rows):
        return False
    control, diamond, *transfer = rows
    return (diamond['share'] > control['share'] and diamond['damagePer100EnemyShots'] < control['damagePer100EnemyShots']
            and all(r['share'] > .35 and r['wins'] >= 1 for r in [diamond, *transfer]))


def confirmation_decision(rows):
    guards, variation = {}, {}
    for enemy in ('Diamond Port', 'Retcon', 'AISurvivor'):
        group = [r for r in rows if r['enemy'] == enemy]
        complete = len(group) == 3 and all(r.get('status') == 'valid' and r['skips'] == 0 and r['shots'] > 0 and r['enemyShots'] > 0 for r in group)
        guards[enemy] = complete and statistics.mean(r['share'] for r in group) > .5 and sum(r['wins'] for r in group) > 30 and sum(r['share'] > .5 for r in group) >= 2
        if complete:
            shares = [r['share'] for r in group]
            variation[enemy] = dict(scoreShares=shares, equalGameMeanShare=statistics.mean(shares), sampleSD=statistics.stdev(shares),
                                    wins=sum(r['wins'] for r in group), rounds=60, wilson95Descriptive=compare.wilson(sum(r['wins'] for r in group), 60))
    return dict(passAll=len(rows) == 9 and all(guards.values()), opponentGuards=guards, variation=variation,
                limitations='Three cold in-roster games per opponent; adaptive rounds are not IID; no screen pooling')
