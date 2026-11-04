#!/usr/bin/env python3
"""Independent official-recording audit; stdlib only, no bot/harness imports.

Reads existing evidence only. Never launches or imports gameplay or benchmark code.
Missing or partial evidence is retained as an invalid/incomplete stage, never filled.
"""
import argparse
import collections
from fractions import Fraction
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys
import zipfile
import zlib

EXPOSURE = ('stoppedRolloutSteps', 'headingChangedSteps', 'secondWaveEvaluations', 'executedStopNow')
COUNTS = ('shots', 'hits', 'shotEnergy', 'damageDealt', 'enemyShots', 'enemyHits',
          'enemyShotEnergy', 'damageTaken', 'wallContacts', 'botContacts', 'bulletCollisions')
OPPONENTS = {'diamond': ('Diamond Port', '1.8.28'), 'retcon': ('Retcon', '0.2.0'),
             'ais': ('AISurvivor', '1.2.1')}
APPROVAL_KEYS = {'schemaVersion', 'verdict', 'repository', 'issue', 'stage', 'headSHA',
    'runId', 'runAttempt', 'bootId', 'workspace', 'artifactId', 'artifactDigestSHA256',
    'profileSHA256', 'freezeSHA256', 'gamesSHA256', 'participantManifestSHA256',
    'opponentManifestSHA256', 'commandsSHA256', 'reviewerTask', 'dispatchAssignmentSHA256'}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def check(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    def unique(pairs):
        out = {}
        for key, value in pairs:
            check(key not in out, 'Duplicate JSON key: ' + key)
            out[key] = value
        return out
    return json.loads(Path(path).read_bytes(), object_pairs_hook=unique,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Nonfinite JSON: ' + x)))


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def ratio(value):
    return {'numerator': value.numerator, 'denominator': value.denominator, 'decimal': float(value)}


def finite(value, label):
    check(type(value) in (float, int) and math.isfinite(value) and value >= 0,
          'Invalid nonnegative finite ' + label)
    return value


def markers(path, rounds, candidate):
    records = {r: {'start': [], 'end': [], 'stats': [], 'skip': []} for r in range(1, rounds + 1)}
    for line_number, line in enumerate(Path(path).read_text().splitlines(), 1):
        match = re.fullmatch(r'([1-9][0-9]*) (start|end|stats(?: .*)?|skip(?: .*)?)', line)
        check(match is not None, f'Unparseable private marker at line {line_number}')
        r, rest = int(match[1]), match[2]
        check(r in records, f'Out-of-population marker round {r}')
        kind = rest.split(' ', 1)[0]
        records[r][kind].append((line_number, rest))
    out = {}
    for r, row in records.items():
        check(len(row['start']) == len(row['stats']) == len(row['end']) == 1,
              f'Round {r} needs exactly one start/stats/end')
        check(row['start'][0][0] < row['stats'][0][0] < row['end'][0][0],
              f'Round {r} marker order differs')
        raw = row['stats'][0][1]
        timing = re.findall(r'(?:^| )maxMs=([^ ]+)', raw)
        check(len(timing) == 1 and re.fullmatch(r'0|[1-9][0-9]*', timing[0]) is not None,
              f'Round {r} invalid integer maxMs')
        exposures = None
        if candidate:
            exposures = {}
            for key in EXPOSURE:
                values = re.findall(r'(?:^| )' + key + r'=([^ ]+)', raw)
                check(len(values) == 1 and re.fullmatch(r'0|[1-9][0-9]*', values[0]) is not None,
                      f'Round {r} invalid/missing candidate exposure {key}')
                exposures[key] = int(values[0])
        out[r] = {'maxMs': int(timing[0]), 'skipCount': len(row['skip']), 'exposure': exposures}
    return out


def initial_identity(event, own_name, own_version, enemy_name, enemy_version, rounds):
    check(event.get('type') == 'GameStartedEventForObserver', 'Recording does not start with official game start')
    s = event['gameSetup']
    expected = {'gameType': '1v1', 'arenaWidth': 800, 'arenaHeight': 600,
                'numberOfRounds': rounds, 'turnTimeout': 100000}
    check(all(s.get(k) == v for k, v in expected.items()), 'Wrong official setup')
    people = event['participants']
    check(len(people) == 2 and len({p['id'] for p in people}) == 2, 'Expected exactly two distinct participants')
    check(sorted((p['name'], p['version']) for p in people) ==
          sorted([(own_name, own_version), (enemy_name, enemy_version)]), 'Wrong official participants/versions')
    own = next(p['id'] for p in people if p['name'] == own_name)
    return own, next(p['id'] for p in people if p['id'] != own), people


def reduce_game(recording, marker_path, own_name, enemy_name, enemy_version, rounds,
                candidate=False, own_version='0.1.0'):
    """Return raw reduction including partial counts on failure; never repairs input."""
    result = {'recording': str(recording), 'markers': str(marker_path), 'valid': False,
              'recordingSHA256': digest(recording), 'markerSHA256': digest(marker_path) if Path(marker_path).exists() else None}
    last_turn, ended, seen_rounds = {}, {}, set()
    totals = {k: 0 for k in COUNTS}
    per_round = {}
    powers, enemy_powers = collections.Counter(), collections.Counter()
    final = None
    try:
        with gzip.open(recording, 'rt', encoding='utf-8') as stream:
            first_line = next(stream)
            first = json.loads(first_line)
            own, enemy, people = initial_identity(first, own_name, own_version, enemy_name, enemy_version, rounds)
            ids = {own, enemy}
            def tally(r, key, value=1):
                finite(value, key)
                totals[key] += value
                per_round[r][key] += value
            previous_round = 1
            for line in stream:
                event = json.loads(line)
                check(final is None, 'Data after final game endpoint')
                kind = event['type']
                if kind == 'TickEventForObserver':
                    r, turn = event['roundNumber'], event['turnNumber']
                    check(type(r) is int and 1 <= r <= rounds and r in (previous_round, previous_round + 1), 'Unexpected tick round order')
                    if r != previous_round:
                        check(previous_round in ended, 'New round before preceding endpoint')
                    previous_round = r
                    seen_rounds.add(r)
                    check(type(turn) is int and turn == last_turn.get(r, 0) + 1 and r not in ended,
                          f'Missing/duplicate/out-of-order tick in round {r}')
                    last_turn[r] = turn
                    check(len({b['id'] for b in event['botStates']}) == len(event['botStates']) and {b['id'] for b in event['botStates']}.issubset(ids), 'Unknown/duplicate official bot state')
                    per_round.setdefault(r, {k: 0 for k in COUNTS})
                    for e in event['events']:
                        et = e['type']
                        if et in ('BulletFiredEvent', 'BulletHitBotEvent', 'BulletHitBulletEvent'):
                            b = e['bullet']
                            check(b['ownerId'] in ids, 'Unknown bullet owner')
                        if et == 'BulletFiredEvent':
                            mine = b['ownerId'] == own
                            tally(r, 'shots' if mine else 'enemyShots')
                            tally(r, 'shotEnergy' if mine else 'enemyShotEnergy', b['power'])
                            (powers if mine else enemy_powers)[str(b['power'])] += 1
                        elif et == 'BulletHitBotEvent':
                            check(e['victimId'] in ids and e['victimId'] != b['ownerId'], 'Invalid duel bullet victim')
                            mine = b['ownerId'] == own
                            tally(r, 'hits' if mine else 'enemyHits')
                            tally(r, 'damageDealt' if mine else 'damageTaken', e['damage'])
                        elif et == 'BulletHitBulletEvent':
                            check(e['hitBullet']['ownerId'] in ids, 'Unknown collided bullet owner')
                            if own in (b['ownerId'], e['hitBullet']['ownerId']):
                                tally(r, 'bulletCollisions')
                        elif et == 'BotHitWallEvent' and e['victimId'] == own:
                            tally(r, 'wallContacts')
                        elif et == 'BotHitBotEvent' and e['botId'] == own:
                            tally(r, 'botContacts')
                elif kind == 'RoundEndedEventForObserver':
                    r = event['roundNumber']
                    seen_rounds.add(r)
                    check(r == len(ended) + 1 and r not in ended and last_turn.get(r) == event['turnNumber'],
                          'Missing/duplicate/misordered official round endpoint')
                    check(len(event['results']) == 2 and sorted((v['id'],v['name'],v['version']) for v in event['results']) == sorted((v['id'],v['name'],v['version']) for v in people), 'Round result participants differ')
                    ended[r] = event
                elif kind == 'GameEndedEventForObserver':
                    final = event
                else:
                    raise ValueError('Unexpected top-level observer message: ' + kind)
        check(final is not None and final['numberOfRounds'] == rounds, 'Missing complete official game endpoint')
        check(list(ended) == list(range(1, rounds + 1)), 'Incomplete registered endpoints')
        expected_people = sorted((p['id'], p['name'], p['version']) for p in people)
        check(sorted((p['id'], p['name'], p['version']) for p in final['results']) == expected_people, 'Final participants differ')
        check(final['results'] == ended[rounds]['results'], 'Game totals differ from last round totals')
        prior_score, prior_wins, rows = {k: 0 for k in ids}, {k: 0 for k in ids}, []
        for r, endpoint in ended.items():
            score = {v['id']: v['totalScore'] for v in endpoint['results']}
            wins = {v['id']: v['firstPlaces'] for v in endpoint['results']}
            check(all(type(v) is int and v >= 0 for v in [*score.values(), *wins.values()]), 'Invalid official score/wins')
            delta = {k: score[k] - prior_score[k] for k in ids}
            dw = {k: wins[k] - prior_wins[k] for k in ids}
            check(all(v >= 0 for v in delta.values()) and sum(delta.values()) > 0 and all(v in (0, 1) for v in dw.values()),
                  'Invalid cumulative round score/win deltas')
            rows.append({'round': r, 'turns': last_turn[r], 'ownScore': delta[own], 'totalScore': sum(delta.values()),
                         'share': float(Fraction(delta[own], sum(delta.values()))), 'win': dw[own], **per_round[r]})
            prior_score, prior_wins = score, wins
        timing = markers(marker_path, rounds, candidate)
        for row in rows:
            row.update(timing[row['round']])
        check(totals['enemyShots'] > 0, 'Incoming shots must be nonzero')
        own_total, total = prior_score[own], sum(prior_score.values())
        skip_count = sum(v['skipCount'] for v in timing.values())
        max_ms = max(v['maxMs'] for v in timing.values())
        result.update(valid=True, share=float(Fraction(own_total, total)), exactShare=ratio(Fraction(own_total, total)),
                      ownScore=own_total, totalScore=total, wins=prior_wins[own], rounds=rounds,
                      **totals, perRound=rows, powerMix=dict(powers), enemyPowerMix=dict(enemy_powers),
                      skips=skip_count, maxMs=max_ms, timingPass=(skip_count == 0 and max_ms < 75),
                      turns=sum(last_turn.values()), hitRate=totals['hits'] / totals['shots'] if totals['shots'] else None,
                      enemyHitRate=totals['enemyHits'] / totals['enemyShots'],
                      firingPer1000Turns=1000 * totals['shots'] / sum(last_turn.values()),
                      exposure={k: sum(v['exposure'][k] for v in timing.values()) for k in EXPOSURE} if candidate else None)
    except (ValueError, KeyError, TypeError, EOFError, OSError, StopIteration, json.JSONDecodeError, zlib.error) as exc:
        result.update(error=type(exc).__name__ + ': ' + str(exc), partialCounters=totals)
    result.update(partialMarkerObservation=marker_observation(marker_path), observedPhysicalRoundIds=sorted(seen_rounds), observedRoundEndpointIds=sorted(ended),
                  observedTickCounts=last_turn,
                  budgetLimit='Observed round IDs are a lower bound if the recording/process was interrupted; do not invent unrecorded rounds.')
    return result


def marker_observation(path):
    # Partial diagnostics cannot make an incomplete recording valid.
    if not Path(path).exists():
        return {'exists': False, 'skips': [], 'reportedTimingViolations': [], 'unparseableLines': []}
    skips, violations, unparseable = [], [], []
    try:
        lines = Path(path).read_text().splitlines()
    except (OSError, UnicodeError) as exc:
        return {'exists': True, 'skips': [], 'reportedTimingViolations': [], 'unparseableLines': [], 'readError': type(exc).__name__ + ': ' + str(exc)}
    for number, line in enumerate(lines, 1):
        m = re.fullmatch(r'([1-9][0-9]*) (start|end|stats(?: .*)?|skip(?: .*)?)', line)
        if not m:
            unparseable.append({'line': number, 'text': line});continue
        if m[2].split(' ', 1)[0] == 'skip':
            skips.append({'round': int(m[1]), 'line': number, 'text': line})
        if m[2].split(' ', 1)[0] == 'stats':
            values = re.findall(r'(?:^| )maxMs=([^ ]+)', m[2])
            if len(values) != 1 or re.fullmatch(r'0|[1-9][0-9]*', values[0]) is None or int(values[0]) >= 75:
                violations.append({'round': int(m[1]), 'line': number, 'text': line})
    return {'exists': True, 'skips': skips, 'reportedTimingViolations': violations, 'unparseableLines': unparseable}


def select_group(rows, arm, opponent):
    return [r for r in rows if r['game']['bot'] == arm and r['game']['opponent'] == opponent]


def group_summary(rows):
    out = {}
    for arm in ('baseline', 'candidate'):
        for opponent in ('diamond', 'retcon', 'walls', 'ais'):
            group = select_group(rows, arm, opponent)
            if not group:
                continue
            fractions = [Fraction(r['ownScore'], r['totalScore']) for r in group]
            mean = sum(fractions, Fraction()) / len(fractions)
            values = [float(v) for v in fractions]
            out[f'{arm}/{opponent}'] = dict(games=len(group), mean=ratio(mean),
                sampleSDPercentagePoints=statistics.stdev(values) * 100 if len(values) > 1 else None,
                minimum=min(values), maximum=max(values), wins=sum(r['wins'] for r in group),
                rounds=sum(r['game']['rounds'] for r in group), gamesAboveHalf=sum(v > Fraction(1, 2) for v in fractions))
    return out


def stage_gates(stage, rows):
    groups = group_summary(rows)
    get = lambda key: Fraction(groups[key]['mean']['numerator'], groups[key]['mean']['denominator'])
    dc, db, rc, rb = map(get, ('candidate/diamond', 'baseline/diamond', 'candidate/retcon', 'baseline/retcon'))
    gates = {'zeroSkipsAndMaxMsBelow75BothArms': all(r['timingPass'] for r in rows),
             'candidateDiamondMeanAboveHalf': dc > Fraction(1, 2),
             'candidateRetconMeanAtLeast90Percent': rc >= Fraction(9, 10)}
    if stage == 'screen':
        gates.update(diamondDeltaAtLeast3pp=dc - db >= Fraction(3, 100),
            diamondWinsAtLeast16=groups['candidate/diamond']['wins'] >= 16,
            diamondGamesAboveHalfAtLeast2=groups['candidate/diamond']['gamesAboveHalf'] >= 2,
            retconDeltaAtLeastMinus3pp=rc - rb >= Fraction(-3, 100),
            retconWinsAtLeast16=groups['candidate/retcon']['wins'] >= 16)
        for opponent in ('walls', 'ais'):
            gates[opponent + 'DeltaAtLeastMinus5pp'] = get('candidate/' + opponent) - get('baseline/' + opponent) >= Fraction(-5, 100)
            gates[opponent + 'WinsAtLeast4'] = groups['candidate/' + opponent]['wins'] >= 4
    else:
        gates.update(diamondDeltaAtLeast2pp=dc - db >= Fraction(2, 100),
            diamondWinsMoreThan60=groups['candidate/diamond']['wins'] > 60,
            diamondGamesAboveHalfAtLeast4=groups['candidate/diamond']['gamesAboveHalf'] >= 4,
            retconDeltaAtLeastMinus2pp=rc - rb >= Fraction(-2, 100),
            retconWinsAtLeast24=groups['candidate/retcon']['wins'] >= 24)
    return {'checks': gates, 'groups': groups, 'allPass': all(gates.values()),
            'diamondDifference': ratio(dc - db), 'retconDifference': ratio(rc - rb)}


def verify_exported_provenance(evidence, source_root, stage, expected_head, expected_run):
    """Offline binding audit, not a claim to rehash unexported live system bytes."""
    failures, checked = [], []
    def require(condition, name):
        (checked if condition else failures).append(name)
    try:
        profile_path = evidence / 'profile.json'
        if not profile_path.exists():
            profile_path = evidence / 'provenance/profile.json'
        profile = read_json(profile_path)
        provenance = profile_path.parent
        ident = profile['identity']
        require(ident['repository'] == 'stephanelee/robocode' and ident['headSHA'] == expected_head, 'profile exact repository/head')
        require(ident['runId'] == expected_run and ident['runAttempt'] == 1, 'profile exact run/attempt1')
        require(ident['image'] in ('20261026.180.1', '20261101.186.1') and ident['architecture'] == 'x86_64'
                and 'VERSION_ID="26.04"' in ident['os'], 'registered Ubuntu image/architecture')
        require(ident['pythonVersion'].startswith('3.13.17 '), 'registered Python3.13.17')
        require(bool(ident['bootId']) and bool(ident['workspace']), 'profile boot/workspace recorded')
        freeze_path = provenance / 'freeze.json'
        if not freeze_path.exists():
            freeze_path = source_root / 'benchmarks/keel-helm-v1/freeze.json'
        freeze = read_json(freeze_path)
        require(profile['sourceAssets'] == freeze['assets'], 'profile source assets equal frozen source assets')
        exported = provenance / 'source'
        if not exported.is_dir():
            exported = evidence / 'provenance/source'
        for rel, wanted in freeze['assets'].items():
            p = exported / rel
            require(p.is_file() and digest(p) == wanted, 'exported frozen source ' + rel)
        commands = read_json(provenance / 'commands.json')
        plan = read_json(source_root / 'benchmarks/keel-helm-v1/games.json')[stage]
        require([v['id'] for v in commands] == [v['id'] for v in plan], 'fixed command population order')
        for command_row, registered in zip(commands, plan):
            prepared = command_row['preparedGame']
            require(prepared['game'] == registered, 'command exact game ' + registered['id'])
            prepared_file = evidence / 'games' / registered['id'] / 'prepared-game.json'
            require(prepared_file.is_file() and digest(prepared_file) == command_row['preparedGameSHA256']
                    and read_json(prepared_file) == prepared, 'exact prepared command manifest ' + registered['id'])
            require(prepared['modelFilesBefore'] == [], 'no imported model file ' + registered['id'])
            require(prepared['persistentModelBefore'].startswith('N/A'), 'Keel persistent model N/A ' + registered['id'])
            for envkey in prepared['environment']:
                require(envkey not in {'GH_TOKEN','GITHUB_TOKEN','CLASSPATH','JAVA_TOOL_OPTIONS','_JAVA_OPTIONS','JDK_JAVA_OPTIONS','PYTHONPATH','PYTHONHOME'}, 'clean child environment ' + registered['id'] + ':' + envkey)
            require(str(registered['port']) in prepared['command'] and str(registered['rounds']) in prepared['command'],
                    'child registered port/rounds ' + registered['id'])
        participants = read_json(provenance / 'participant-manifest.json')
        for arm, name in [('baseline','Keel'),('candidate','KeelHelm')]:
            require(participants[arm]['name'] == name and participants[arm]['version'] == '0.1.0', 'participant identity ' + arm)
            require(participants[arm]['jarSHA256'] == freeze['assets']['experiments/' + name + '/' + name + '.jar'], 'participant frozen JAR ' + arm)
        approval_path = evidence / 'approval.json'
        approval = read_json(approval_path)
        require(set(approval) == APPROVAL_KEYS and approval['verdict'] == 'CLEAR', 'strict20-key runtime approval')
        for key, value in [('headSHA', expected_head), ('runId', expected_run), ('runAttempt', 1),
                           ('bootId', ident['bootId']), ('workspace', ident['workspace']), ('stage', stage)]:
            require(type(approval[key]) is type(value) and approval[key] == value, 'approval ' + key)
        for key, path in [('profileSHA256', profile_path), ('freezeSHA256', freeze_path),
                          ('gamesSHA256', source_root / 'benchmarks/keel-helm-v1/games.json'),
                          ('participantManifestSHA256', provenance / 'participant-manifest.json'),
                          ('opponentManifestSHA256', provenance / 'opponent-manifest.json'),
                          ('commandsSHA256', provenance / 'commands.json')]:
            require(path.is_file() and approval[key] == digest(path), 'approval ' + key)
        artifact = read_json(evidence / 'artifact.json')
        require(artifact['id'] == approval['artifactId'] and artifact['digest'] == 'sha256:' + approval['artifactDigestSHA256'],
                'approval finalized artifact ID/digest')
        require(artifact['workflow_run']['id'] == expected_run and artifact['workflow_run']['head_sha'] == expected_head,
                'artifact exact workflow run/head')
        receipt = read_json(evidence / 'receipt.json')
        payload = receipt['payload']
        require(payload['approval'] == approval, 'permanent receipt binds approval')
        require(payload['receiptRef'] == 'refs/tags/experiment-receipts/keel-helm-v1-' + stage, 'fixed stage receipt name')
        require(payload['jobRemainingSecondsAtAdmission'] >= 2100, 'receipt admission at least35minutes')
        require(receipt['createdRef']['object']['sha'] == receipt['annotatedTag']['sha'], 'created receipt binds annotated tag SHA')
        require(receipt['annotatedTag']['object']['sha'] == expected_head, 'annotated tag points at exact head')
        dispatch = payload['dispatchAssignment']
        require(hashlib.sha256(dispatch['body'].encode()).hexdigest() == approval['dispatchAssignmentSHA256'], 'receipt exact dispatch body hash')
        require(payload['acceptedComment']['body'].startswith('KEEL_HELM_RUNTIME_APPROVAL_V1\n'), 'receipt accepted approval API body')
        require(payload['reviewReport']['body'].startswith('KEEL_HELM_RUNTIME_REVIEW_V1\n'), 'receipt separate reasoned review API body')
        require(profile['runner'] == '02f6d1d8e9346a4aeae1b91e7abb278c98ac1f1d30c605f867758566f5f327cf', 'pinned runner hash')
        require('17.0.21' in profile['java17']['version'] and '25.0.5' in profile['java25']['version'], 'recorded Java17 build then25 battle identities')
    except (KeyError, TypeError, ValueError, OSError) as exc:
        failures.append(type(exc).__name__ + ': ' + str(exc))
    return {'checked': checked, 'failures': failures, 'pass': not failures,
            'scope': 'Archived identities/exported source and receipt bindings only. Live system files, API freshness, remote tag/run cardinality and process start/admission timing require independent API/log evidence review.'}


def audit_stage(evidence, source_root, stage, expected_head, expected_run):
    plan_path = source_root / 'benchmarks/keel-helm-v1/games.json'
    population = read_json(plan_path)[stage]
    rows, missing, partial, protocol_failures = [], [], [], []
    first_failure = None
    for game in population:
        d = evidence / 'games' / game['id']
        found = list(d.glob('*.battle.gz'))
        if not found:
            missing.append(game['id'])
            if any((d / name).exists() for name in ('attempt.json', 'runner.log', 'markers.txt')):
                partial.append({'game': game, 'valid': False, 'error': 'Attempt evidence exists without a recording',
                                'observedPhysicalRoundIds': [], 'budgetLimit': 'Physical round count unknown without recording; inspect runner/attempt/API evidence.'})
                first_failure = first_failure or game['id']
            continue
        if first_failure:
            protocol_failures.append('Recording exists after first invalid/timing failure: ' + game['id'])
        if missing:
            protocol_failures.append('Recording exists after an unplayed/missing preceding ID: ' + game['id'])
        if len(found) != 1:
            partial.append({'game': game, 'valid': False, 'error': 'Multiple recordings; no selection allowed',
                            'recordingFiles': [str(p) for p in found]})
            first_failure = first_failure or game['id']
            continue
        own_name = 'KeelHelm' if game['bot'] == 'candidate' else 'Keel'
        if game['opponent'] == 'walls':
            # Version must come from frozen exported official descriptor, not a guessed number.
            opponent = ('Walls', None)
            prepared = d / 'prepared-game.json'
            if prepared.exists():
                opponent = ('Walls', read_json(prepared)['opponentVersion'])
            check(opponent[1] is not None, 'Pinned Walls descriptor version required')
        else:
            opponent = OPPONENTS[game['opponent']]
        row = reduce_game(found[0], d / 'markers.txt', own_name, *opponent, game['rounds'], game['bot'] == 'candidate')
        row['game'] = game
        if row['valid']:
            if game['opponent'] == 'ais':
                ais = d / 'ais.jsonl';model = d / 'ais-model.dat'
                try:
                    metrics = [json.loads(l) for l in ais.read_text().splitlines()]
                    check([v['round'] for v in metrics] == list(range(1, game['rounds'] + 1)), 'AIS complete round telemetry')
                    check(all(v['mode'] == 'train' and v['seed'] == game['seed'] and v['checkpointLoaded'] is False for v in metrics), 'AIS cold TRAIN identity')
                    raw = model.read_bytes()
                    check(raw[:4] == b'AISV' and int.from_bytes(raw[4:8], 'big') == 1 and int.from_bytes(raw[20:24], 'big') == game['rounds'], 'AIS actual stored model completion')
                    row['actualAISFiles'] = {p.name: digest(p) for p in (ais, model)}
                except (ValueError, KeyError, OSError) as exc:
                    row.update(valid=False, error=type(exc).__name__ + ': ' + str(exc))
            if not row.get('timingPass'):
                first_failure = first_failure or game['id']
        if not row['valid']:
            partial.append(row);first_failure = first_failure or game['id']
        else:
            rows.append(row)
    complete = len(rows) == len(population) and not missing and not partial and not protocol_failures
    raw = stage_gates(stage, rows) if complete else None
    provenance = verify_exported_provenance(evidence, source_root, stage, expected_head, expected_run)
    return {'schemaVersion': 1, 'stage': stage, 'independentReducer': True, 'gamesSHA256': digest(plan_path),
            'expectedHead': expected_head, 'expectedRun': expected_run, 'registeredGames': len(population),
            'registeredRoundBudget': sum(g['rounds'] for g in population), 'validGames': rows,
            'invalidOrPartialGames': partial, 'unrecordedGameIds': missing, 'protocolFailures': protocol_failures,
            'observedPhysicalRoundCountLowerBound': sum(len(r.get('observedPhysicalRoundIds', [])) for r in rows + partial),
            'completeRawPopulation': complete, 'rawGates': raw, 'provenance': provenance,
            'observedTimingOrSkipFailures': {r['game']['id']:r['partialMarkerObservation'] for r in rows+partial if r.get('partialMarkerObservation',{}).get('skips') or r.get('partialMarkerObservation',{}).get('reportedTimingViolations')},
            'decision': ('PASS' if raw['allPass'] and provenance['pass'] else 'REJECT' if not raw['allPass'] else 'PROVENANCE-INCOMPLETE') if complete else 'INVALID/INCOMPLETE',
            'limits': ['No source/JAR/model/approval changes are authorized by this audit.',
                       'Do not pool historical games, screen or confirmation.',
                       'Round learning is correlated; equal-weight cold games are the comparison units.',
                       'Diamond private skipped turns are not observable.',
                       'Observed partial round counts are lower bounds, not permission to replace missing rounds.']}


def historical(root):
    """Read-only verification against already published, unrelated Keel confirmation."""
    published = read_json(root / 'decision-confirmation.json')
    rows, mismatches = [], []
    for old in published['games']:
        d = root / old['id'];recording = list(d.glob('*.battle.gz'))
        check(len(recording) == 1, 'Historical recording cardinality')
        row = reduce_game(recording[0], d / 'markers.txt', 'Keel', 'Diamond Port', '1.8.28', 20)
        check(row['valid'], row.get('error', 'Historical reduction failed'))
        keys = [*COUNTS, 'share', 'wins', 'hitRate', 'enemyHitRate', 'skips', 'recordingSHA256']
        for k in keys:
            a, b = row[k], old[k]
            same = math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-9) if type(a) is float else a == b
            if not same:
                mismatches.append({'game': old['id'], 'field': k, 'independent': a, 'published': b})
        check([r['turns'] for r in row['perRound']] == old['turnsByRound'], 'Historical per-round turns differ')
        check([r['share'] for r in row['perRound']] == old['roundShares'], 'Historical per-round shares differ')
        row['game'] = {'id': old['id'], 'bot': 'baseline', 'opponent': 'diamond', 'rounds': 20}
        rows.append(row)
    mean = sum((Fraction(r['ownScore'], r['totalScore']) for r in rows), Fraction()) / len(rows)
    return {'scope': 'Historical immutable Keel confirmation verification ONLY; no KeelHelm games or strength evidence.',
            'sourceDecisionSHA256': digest(root / 'decision-confirmation.json'), 'games': len(rows),
            'rounds': sum(r['rounds'] for r in rows), 'mean': ratio(mean), 'wins': sum(r['wins'] for r in rows),
            'recordingHashes': {r['game']['id']: r['recordingSHA256'] for r in rows},
            'mismatches': mismatches, 'pass': not mismatches}


def audit_archive(original_zip, expected_sha256, extracted_root):
    """Verify original bytes and every safe regular ZIP entry without rewriting either."""
    check(re.fullmatch('[0-9a-f]{64}', expected_sha256) is not None, 'Expected original ZIP SHA256 required')
    check(digest(original_zip) == expected_sha256, 'Original API artifact ZIP SHA256 differs')
    entries, seen = {}, set()
    with zipfile.ZipFile(original_zip) as archive:
        for item in archive.infolist():
            name = item.filename
            parts = Path(name).parts
            check(name not in seen, 'Duplicate ZIP member: ' + name);seen.add(name)
            check(not name.startswith(('/', '\\')) and '\\' not in name and '..' not in parts
                  and re.match('[A-Za-z]:', name) is None, 'Unsafe ZIP entry: ' + name)
            mode = item.external_attr >> 16
            check(mode & 0o170000 != 0o120000, 'Symlink ZIP entry is outside regular-evidence contract: ' + name)
            if item.is_dir():
                continue
            raw = archive.read(item)
            file = extracted_root / name
            check(file.is_file() and not file.is_symlink(), 'Missing/nonregular extracted original: ' + name)
            checksum = hashlib.sha256(raw).hexdigest()
            check(file.stat().st_size == len(raw) and digest(file) == checksum, 'Extracted byte mismatch: ' + name)
            entries[name] = {'bytes': len(raw), 'sha256': checksum}
    return {'scope': 'Original ZIP/API-digest and all regular extracted member bytes; no games executed.',
            'originalZIP': str(original_zip), 'originalZIPBytes': original_zip.stat().st_size,
            'originalZIPSHA256': expected_sha256, 'regularFiles': entries, 'fileCount': len(entries), 'pass': True}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    h = sub.add_parser('verify-historical');h.add_argument('--root', type=Path, required=True);h.add_argument('--output', type=Path, required=True)
    a = sub.add_parser('audit-stage');a.add_argument('--evidence', type=Path, required=True);a.add_argument('--source-root', type=Path, required=True)
    a.add_argument('--stage', choices=['screen', 'confirmation'], required=True);a.add_argument('--head', required=True);a.add_argument('--run', type=int, required=True);a.add_argument('--output', type=Path, required=True)
    z = sub.add_parser('verify-archive');z.add_argument('--zip', type=Path, required=True);z.add_argument('--sha256', required=True);z.add_argument('--extracted-root', type=Path, required=True);z.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.command == 'verify-historical':
        result = historical(args.root)
    elif args.command == 'verify-archive':
        result = audit_archive(args.zip, args.sha256, args.extracted_root)
    else:
        result = audit_stage(args.evidence, args.source_root, args.stage, args.head, args.run)
    result['auditorSourceSHA256'] = digest(Path(__file__))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: result[k] for k in ('scope', 'pass', 'decision', 'games', 'rounds', 'mismatches') if k in result}))
    return 0 if result.get('pass', result.get('decision') == 'PASS') else 1


if __name__ == '__main__':
    sys.exit(main())
