"""Validate only MeridianPeak's explicit, cold-start, sealed own-state DTOs."""
import json
import math

import benchmark
from meridian_peak_common import finite, integer

STATS_INTS = ('confirmedShots', 'hits', 'hitsTaken', 'skippedTurns', 'crossingLabels', 'acceptedHitLabels',
              'plannerDecisions', 'kernelEvaluations', 'decisionTotalNanos', 'maximumDecisionNanos',
              'maximumActiveWaves', 'maximumModeledRays', 'maximumRaysPerWave',
              'crossingMaskedMissingBracket', 'crossingMaskedInsideFront', 'crossingMaskedOutsideOffset',
              'trialScored', 'trialMissingScan')
STATS_FLOATS = ('shotEnergy', 'damageDealt', 'damageTaken', 'maximumWaveRisk')
ROOT_KEYS = {'schemaVersion', 'name', 'version', 'seed', 'checkpointLoaded', 'round', 'completedRounds', 'revision',
             'sealed', 'coldStateSHA256', 'cumulativeCrossings', 'cumulativeAcceptedHits', 'stats', 'models'}


def numeric(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError('Expected a finite numeric DTO field')


def pose(value):
    if set(value) != {'x', 'y', 'h', 'v'}:
        raise ValueError('Pose DTO schema differs')
    for item in value.values():
        numeric(item)


def close(a, b, label, tolerance=1e-6):
    if not math.isclose(a, b, rel_tol=1e-10, abs_tol=tolerance):
        raise ValueError('Private/official or model counter mismatch: ' + label)


def context(value):
    if set(value) != {'lateral', 'radial', 'wall', 'flight'}:
        raise ValueError('Crossing context schema differs')
    for item in value.values():
        numeric(item)
    if not (0 <= value['lateral'] <= 8 + 1e-9 and -8 - 1e-9 <= value['radial'] <= 8 + 1e-9
            and 0 <= value['wall'] <= 1 and value['flight'] >= 0):
        raise ValueError('Crossing context outside declared legal feature bounds')


def launch(value):
    if set(value) != {'turn', 'x', 'y', 'power', 'bearing', 'sign', 'range', 'context'}:
        raise ValueError('Launch DTO schema differs')
    integer(value['turn'], 'launch turn', 1)
    for key in ('x', 'y', 'power', 'bearing', 'sign', 'range'):
        numeric(value[key])
    if not (.1 <= value['power'] <= 3 and value['sign'] in (-1, 1) and value['range'] >= 0):
        raise ValueError('Illegal own launch DTO')
    context(value['context'])


def trial(value):
    if set(value) != {'due', 'angles', 'x', 'y'} or len(value['angles']) != 3:
        raise ValueError('Trial DTO differs')
    integer(value['due'], 'trial due')
    for item in [value['x'], value['y'], *value['angles']]:
        numeric(item)


def journal(path, rounds):
    entries = [json.loads(line) for line in path.read_text().splitlines()]
    finite(entries)
    previous_round, launches = 0, set()
    for row in entries:
        if set(row) != {'round', 'type', 'label'}:
            raise ValueError('Own label journal schema differs')
        number = integer(row['round'], 'journal round', 1)
        if number < previous_round or number > rounds:
            raise ValueError('Out-of-order/out-of-range journal round')
        previous_round = number
        value = row['label']
        if row['type'] == 'crossing':
            if set(value) != {'turn', 'launch', 'enemy', 'previousTurn', 'previous', 'reason', 'offset'}:
                raise ValueError('Crossing journal label differs')
            launch(value['launch'])
            pose(value['enemy'])
            if value['previous'] is not None:
                pose(value['previous'])
            if value['offset'] is not None:
                numeric(value['offset'])
            integer(value['turn'], 'crossing label turn', 1)
            key = (number, value['launch']['turn'])
            if key in launches:
                raise ValueError('Crossing launch labelled twice')
            launches.add(key)
            if value['reason'] not in ('RESOLVED', 'MISSING_BRACKET', 'INSIDE_PREVIOUS_FRONT', 'OUTSIDE_OFFSET'):
                raise ValueError('Unknown crossing resolution/mask reason')
            if value['reason'] == 'RESOLVED' and (value['offset'] is None or abs(value['offset']) > 1.25):
                raise ValueError('Resolved crossing offset invalid')
            if value['reason'] == 'OUTSIDE_OFFSET' and (value['offset'] is None or abs(value['offset']) <= 1.25):
                raise ValueError('Outside crossing offset invalid')
        elif row['type'] == 'trial':
            if set(value) != {'observedTurn', 'reason', 'trial', 'enemy'}:
                raise ValueError('Trial journal label differs')
            trial(value['trial'])
            pose(value['enemy'])
            integer(value['observedTurn'], 'observed trial turn', 1)
            due = value['trial']['due']
            if not ((value['reason'] == 'SCORED' and due == value['observedTurn'])
                    or (value['reason'] == 'MISSING_SCAN' and due < value['observedTurn'])):
                raise ValueError('Trial resolved/masked phase mismatch')
        else:
            raise ValueError('Unknown own label journal type')
    return entries


def snapshot(value, game, number, cold_hash):
    finite(value)
    if set(value) != ROOT_KEYS or type(value['schemaVersion']) is not int or value['schemaVersion'] != 1 or value['name'] != 'MeridianPeak' or value['version'] != '0.1.0':
        raise ValueError('Peak snapshot schema/identity differs')
    if type(value['seed']) is not int or value['seed'] != game['seed'] or value['checkpointLoaded'] is not False:
        raise ValueError('Peak seed or cold-start evidence differs')
    for key in ('round', 'completedRounds', 'revision', 'cumulativeCrossings', 'cumulativeAcceptedHits'):
        integer(value[key], key)
    if value['round'] != number or value['completedRounds'] != number:
        raise ValueError('Peak completed-round counter advanced twice or is incomplete')
    if number == 0:
        if value['sealed'] is not False or value['revision'] != 0 or value['coldStateSHA256'] is not None:
            raise ValueError('Invalid cold-state phase')
    elif value['sealed'] is not True or value['revision'] < 1 or value['coldStateSHA256'] != cold_hash:
        raise ValueError('Missing latest seal or wrong cold-state linkage')
    stats = value['stats']
    if set(stats) != set(STATS_INTS + STATS_FLOATS):
        raise ValueError('Missing/unknown required Peak counters')
    for key in STATS_INTS:
        integer(stats[key], key)
    if any(type(stats[key]) not in (int, float) or stats[key] < 0 for key in STATS_FLOATS):
        raise ValueError('Invalid Peak damage/energy/risk counter')
    if stats['skippedTurns'] or stats['maximumWaveRisk'] > 80 + 1e-9:
        raise ValueError('Peak skip or per-wave risk bound violation')
    if stats['maximumRaysPerWave'] > 23 or stats['maximumModeledRays'] > 23 * stats['maximumActiveWaves']:
        raise ValueError('Peak ray counter exceeds explicit learner support')
    if stats['maximumDecisionNanos'] > stats['decisionTotalNanos']:
        raise ValueError('Impossible decision duration counters')
    models = value['models']
    if set(models) != {'hitAngles', 'crossingGun', 'gunSkill', 'pendingTrials'}:
        raise ValueError('Unknown/missing model DTO')
    angles = models['hitAngles']
    if set(angles) != {'weights', 'effectiveTotal'} or len(angles['weights']) != 21 or min(angles['weights']) < 0 or angles['effectiveTotal'] < 0:
        raise ValueError('Malformed 21-bin incoming model')
    for item in [angles['effectiveTotal'], *angles['weights']]:
        numeric(item)
    close(sum(angles['weights']), angles['effectiveTotal'], 'decayed incoming mass', 1e-9)
    expected_total = (1 - .95 ** value['cumulativeAcceptedHits']) / .05
    close(angles['effectiveTotal'], expected_total, 'decayed effective total vs cumulative accepted labels', 1e-9)
    gun = models['crossingGun']
    if set(gun) != {'capacity', 'count', 'next', 'examples', 'inFlight', 'requested'} or gun['capacity'] != 192:
        raise ValueError('Malformed crossing-bank DTO')
    integer(gun['count'], 'crossing count')
    integer(gun['next'], 'crossing next')
    if gun['count'] != min(192, value['cumulativeCrossings']) or gun['next'] != value['cumulativeCrossings'] % 192:
        raise ValueError('Capped bank state disagrees with cumulative legal labels')
    if len(gun['examples']) != gun['count'] or [e['index'] for e in gun['examples']] != list(range(gun['count'])):
        raise ValueError('Missing/duplicate crossing examples')
    for example in gun['examples']:
        if set(example) != {'index', 'context', 'offset'} or abs(example['offset']) > 1.25:
            raise ValueError('Illegal crossing example')
        numeric(example['offset'])
        context(example['context'])
    for pending in gun['inFlight']:
        launch(pending)
    if gun['requested'] is not None:
        launch(gun['requested'])
    if len(models['gunSkill']) != 3 or any(not 0 <= v <= 1 for v in models['gunSkill']):
        raise ValueError('Invalid gun skill DTO')
    for item in models['gunSkill']:
        numeric(item)
    for pending in models['pendingTrials']:
        trial(pending)
    if number == 0:
        if any(stats.values()) or value['cumulativeCrossings'] or value['cumulativeAcceptedHits'] or any(angles['weights']) or gun['inFlight'] or gun['requested'] is not None or models['gunSkill'] != [.2, .3, .25] or models['pendingTrials']:
            raise ValueError('Cold state is not empty original-v3 learner state')


def validate(directory, game, markers, official):
    expected = {'cold.json', 'endpoint.json', 'labels.jsonl', *(f'round-{n:03d}.json' for n in range(1, game['rounds'] + 1))}
    if {p.name for p in directory.iterdir()} != expected:
        raise ValueError('Missing/extra/temporary private model files')
    cold = directory / 'cold.json'
    cold_hash = benchmark.digest(cold)
    snapshot(json.loads(cold.read_text()), game, 0, None)
    labels_journal = journal(directory / 'labels.jsonl', game['rounds'])
    sealed, cumulative_crossings, cumulative_hits, total_nanos, max_nanos = [], 0, 0, 0, 0
    for number, official_round in enumerate(official['perRound'], 1):
        path = directory / f'round-{number:03d}.json'
        value = json.loads(path.read_text())
        snapshot(value, game, number, cold_hash)
        stats = value['stats']
        entries = [entry for entry in labels_journal if entry['round'] == number]
        for label_type, reason, counter in (('crossing','RESOLVED','crossingLabels'),
                ('crossing','MISSING_BRACKET','crossingMaskedMissingBracket'),
                ('crossing','INSIDE_PREVIOUS_FRONT','crossingMaskedInsideFront'),
                ('crossing','OUTSIDE_OFFSET','crossingMaskedOutsideOffset'),
                ('trial','SCORED','trialScored'),('trial','MISSING_SCAN','trialMissingScan')):
            if sum(e['type']==label_type and e['label']['reason']==reason for e in entries) != stats[counter]:
                raise ValueError('Resolved/masked journal counts differ: ' + counter)
        round_markers = [line.split() for line in markers if int(line.split()[0]) == number]
        fired = [line for line in round_markers if len(line) == 4 and line[1] == 'fire']
        turns = [int(line[2]) for line in fired]
        if len(turns) != len(set(turns)) or any(t < 1 or t > official_round['turns'] for t in turns):
            raise ValueError('Duplicate/out-of-range private fire confirmation')
        labels = [line[2] for line in round_markers if len(line) == 3 and line[1] == 'hit']
        if any(label not in ('ACCEPTED', 'UNMATCHED', 'AMBIGUOUS', 'UNORIENTED', 'OUTSIDE') for label in labels):
            raise ValueError('Unknown hit label status')
        crossings = sum(integer(int(line[2]), 'marker crossing count') for line in round_markers if len(line) == 3 and line[1] == 'crossings')
        if stats['crossingLabels'] != crossings or stats['acceptedHitLabels'] != labels.count('ACCEPTED') or stats['hitsTaken'] != len(labels):
            raise ValueError('Model label counters do not reconcile with complete own markers')
        if len(fired) != stats['confirmedShots']:
            raise ValueError('Confirmed shot markers incomplete')
        close(sum(float(line[3]) for line in fired), stats['shotEnergy'], 'private confirmed fire energy')
        for private, observer in (('confirmedShots', 'shots'), ('hits', 'callbackDeliverableOutgoingHits'), ('hitsTaken', 'hitsTaken'), ('skippedTurns', 'skips')):
            if stats[private] != official_round[observer]:
                raise ValueError('Private callback/official event counter mismatch: ' + private)
        close(stats['damageDealt'], official_round['callbackDeliverableOutgoingDamage'], 'deliverable outgoing damage')
        for key in ('shotEnergy', 'damageTaken'):
            close(stats[key], official_round[key], key)
        if stats['plannerDecisions'] > official_round['turns']:
            raise ValueError('More planner decisions than observed turns')
        if stats['crossingLabels'] > stats['confirmedShots']:
            raise ValueError('More resolved crossing labels than confirmed own shots in the round')
        if len(value['models']['crossingGun']['inFlight']) > stats['confirmedShots']:
            raise ValueError('More unresolved launches than confirmed shots')
        cumulative_crossings += stats['crossingLabels']
        cumulative_hits += stats['acceptedHitLabels']
        if value['cumulativeCrossings'] != cumulative_crossings or value['cumulativeAcceptedHits'] != cumulative_hits:
            raise ValueError('Cumulative model labels double counted/missing across seals')
        total_nanos += stats['decisionTotalNanos']
        max_nanos = max(max_nanos, stats['maximumDecisionNanos'])
        sealed.append(dict(round=number, revision=value['revision'], sha256=benchmark.digest(path), stats=stats))
    final = directory / f"round-{game['rounds']:03d}.json"
    if (directory / 'endpoint.json').read_bytes() != final.read_bytes():
        raise ValueError('Endpoint differs from latest final-round seal')
    return dict(available=True, coldSHA256=cold_hash, endpointSHA256=benchmark.digest(directory / 'endpoint.json'),
                labelsSHA256=benchmark.digest(directory / 'labels.jsonl'), labelRows=len(labels_journal),
                sealedRounds=sealed, cumulativeCrossings=cumulative_crossings, cumulativeAcceptedHits=cumulative_hits,
                decisionWallTotalNanos=total_nanos, maximumDecisionWallNanos=max_nanos,
                timingDefinition='System.nanoTime durations around own decision path; not process CPU time')
