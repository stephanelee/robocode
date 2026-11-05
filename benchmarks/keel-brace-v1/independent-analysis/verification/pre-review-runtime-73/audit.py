#!/usr/bin/env python3
"""Independent official-recording audit; stdlib only, no bot/harness imports.

Reads existing evidence only. Never launches or imports gameplay or benchmark code.
Missing or partial evidence is retained as an invalid/incomplete stage, never filled.
"""
import argparse
import collections
from fractions import Fraction
import gzip
import io
import fnmatch
import datetime
from types import SimpleNamespace
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys
import zipfile
import zlib

EXPOSURE = ('rawRamEvents', 'eligibleRamCorrectionEvents', 'eligibleRamCorrectedScans', 'ramChangedPowerScans', 'ramChangedGateScans')
COUNTER_CAP = 1_000_000
ISSUE = 114
PR = 115
NAMESPACE = 'keel-brace-v1'
AUDIT_RELATIVE_PATH = 'benchmarks/keel-brace-v1/independent-analysis/audit.py'
PROTOCOL_SHA256 = 'cd1ec9f5bba78b47f487dc829b59d99ed52953266f15c1c3b5407186b2ad20a8'
GAMES_SHA256 = '9526d1a75d7ee99ac3d52e5affbf2487823b696c3d9b1e0f0497b1d42d65b3a3'
RUNTIME_PINS_SHA256 = '2b7b946b633ffc972bb34d98b91eeb1df897b775ea204d075ff8ae8a0a42f9e5'
BASELINE_MANIFEST_SHA256 = 'cf85147dbf2f9b4ae41495011f20a0850b5349cd86fff4133889ecaade77a623'
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
    with evidence_path(path).open('rb') as stream:
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
    return json.loads(evidence_path(path).read_bytes(), object_pairs_hook=unique,
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
    for line_number, line in enumerate(evidence_path(path).read_text().splitlines(), 1):
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
        if r>1:check(records[r-1]['end'][0][0]<row['start'][0][0],f'Round {r} starts before preceding round end')
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
        if candidate:
            check(all(v <= COUNTER_CAP for v in exposures.values()), f'Round {r} diagnostic cap exceeded')
            chain = [exposures[k] for k in ('ramChangedGateScans','ramChangedPowerScans','eligibleRamCorrectedScans','eligibleRamCorrectionEvents','rawRamEvents')]
            check(chain == sorted(chain), f'Round {r} diagnostic inequality violated')
        else:
            check(not any(re.search(r'(?:^| )' + k + '=', raw) for k in EXPOSURE), f'Round {r} baseline diagnostics must be N/A')
        out[r] = {'maxMs': int(timing[0]), 'skipCount': len(row['skip']), 'exposure': exposures}
    return out


def initial_identity(event, own_name, own_version, enemy_name, enemy_version, rounds):
    check(event.get('type') == 'GameStartedEventForObserver', 'Recording does not start with official game start')
    s = event['gameSetup']
    expected = {'gameType': '1v1', 'arenaWidth': 800, 'arenaHeight': 600,
                'numberOfRounds': rounds, 'turnTimeout': 100000}
    check(all(s.get(k) == v for k, v in expected.items()), 'Wrong official setup')
    people = event['participants']
    check(len(people) == 2 and all(type(p['id']) is int and p['id']>0 for p in people) and len({p['id'] for p in people}) == 2, 'Expected exactly two distinct integer-ID participants')
    check(sorted((p['name'], p['version']) for p in people) ==
          sorted([(own_name, own_version), (enemy_name, enemy_version)]), 'Wrong official participants/versions')
    own = next(p['id'] for p in people if p['name'] == own_name)
    return own, next(p['id'] for p in people if p['id'] != own), people


def reduce_game(recording, marker_path, own_name, enemy_name, enemy_version, rounds,
                candidate=False, own_version='0.1.0'):
    """Return raw reduction including partial counts on failure; never repairs input."""
    result = {'recording': str(recording), 'markers': str(marker_path), 'valid': False,
              'recordingSHA256': digest(recording), 'markerSHA256': digest(marker_path) if evidence_path(marker_path).exists() else None}
    last_turn, ended, seen_rounds = {}, {}, set()
    totals = {k: 0 for k in COUNTS}
    per_round = {}
    powers, enemy_powers = collections.Counter(), collections.Counter()
    final = None
    try:
        with evidence_path(recording).open('rb') as compressed, gzip.GzipFile(fileobj=compressed) as raw, io.TextIOWrapper(raw, encoding='utf-8') as stream:
            first_line = next(stream)
            first = json_data(first_line)
            own, enemy, people = initial_identity(first, own_name, own_version, enemy_name, enemy_version, rounds)
            ids = {own, enemy}
            def tally(r, key, value=1):
                finite(value, key)
                totals[key] += value
                per_round[r][key] += value
            previous_round = 1
            for line in stream:
                event = json_data(line)
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
        check(final is not None and type(final['numberOfRounds']) is int and final['numberOfRounds'] == rounds, 'Missing complete official game endpoint')
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
            if candidate:
                check(row['exposure']['rawRamEvents'] <= row['botContacts'], 'Private rawRamEvents exceed official own contact opportunities')
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
    if not evidence_path(path).exists():
        return {'exists': False, 'skips': [], 'reportedTimingViolations': [], 'unparseableLines': []}
    skips, violations, unparseable = [], [], []
    try:
        lines = evidence_path(path).read_text().splitlines()
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
    """Exact prospective score/timing/exposure gates; does not filter by contacts."""
    check(stage in ('screen','confirmation'), 'Unknown registered stage')
    groups = group_summary(rows)
    get = lambda key: Fraction(groups[key]['mean']['numerator'], groups[key]['mean']['denominator'])
    dc, db, rc, rb = map(get, ('candidate/diamond', 'baseline/diamond', 'candidate/retcon', 'baseline/retcon'))
    exposure = sum(r['exposure']['eligibleRamCorrectedScans'] for r in select_group(rows,'candidate','retcon'))
    check(type(exposure) is int and exposure >= 0, 'Invalid stage-local Retcon exposure')
    gates = {'zeroSkipsAndMaxMsBelow75BothArms': all(r['timingPass'] for r in rows),
             'candidateDiamondMeanAboveHalf': dc > Fraction(1,2),
             'candidateRetconMeanAtLeast90Percent': rc >= Fraction(9,10),
             'retconDeltaAtLeast1pp': rc-rb >= Fraction(1,100)}
    if stage == 'screen':
        gates.update(diamondDeltaAtLeastMinus3pp=dc-db >= Fraction(-3,100),
            diamondWinsAtLeast16=groups['candidate/diamond']['wins'] >= 16,
            diamondGamesAboveHalfAtLeast2=groups['candidate/diamond']['gamesAboveHalf'] >= 2,
            retconWinsAtLeast16=groups['candidate/retcon']['wins'] >= 16)
        for opponent in ('walls','ais'):
            gates[opponent+'DeltaAtLeastMinus5pp'] = get('candidate/'+opponent)-get('baseline/'+opponent) >= Fraction(-5,100)
            gates[opponent+'WinsAtLeast4'] = groups['candidate/'+opponent]['wins'] >= 4
    else:
        gates.update(diamondDeltaAtLeastMinus2pp=dc-db >= Fraction(-2,100),
            diamondWinsMoreThan60=groups['candidate/diamond']['wins'] > 60,
            diamondGamesAboveHalfAtLeast4=groups['candidate/diamond']['gamesAboveHalf'] >= 4,
            retconWinsAtLeast24=groups['candidate/retcon']['wins'] >= 24)
    exposure_pass = exposure > 0
    score_timing_pass = all(gates.values())
    checks = dict(gates, candidateRetconEligibleRamCorrectedScansAboveZero=exposure_pass)
    return {'checks':checks, 'groups':groups, 'allPass':score_timing_pass and exposure_pass,
            'scoreAndTimingPass':score_timing_pass, 'exposurePass':exposure_pass,
            'candidateRetconEligibleRamCorrectedScans':exposure,
            'rawDisposition':'INVALID/INCOMPLETE' if not gates['zeroSkipsAndMaxMsBelow75BothArms'] else 'INCONCLUSIVE' if not exposure_pass else 'PASS' if score_timing_pass else 'REJECT',
            'diamondDifference':ratio(dc-db), 'retconDifference':ratio(rc-rb)}


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
        require(ident['repository'] == 'stephanelee/robocode' and ident['headSHA'] == expected_head and ident['pullNumber'] == PR, 'profile exact repository/head')
        require(ident['runId'] == expected_run and ident['runAttempt'] == 1, 'profile exact run/attempt1')
        require(ident['image'] == QUALIFIED_ENV['ImageVersion'] and ident['architecture'] == 'x86_64'
                and 'VERSION_ID="26.04"' in ident['os'], 'profile DECLARES separate qualified environment/Ubuntu architecture; native snapshots and final reconciliation remain mandatory')
        require(ident['pythonVersion'].startswith('3.13.17 '), 'registered Python3.13.17')
        require(bool(ident['bootId']) and bool(ident['workspace']), 'profile boot/workspace recorded')
        freeze_path = provenance / 'freeze.json'
        if not freeze_path.exists():
            freeze_path = source_root / 'benchmarks/keel-brace-v1/freeze.json'
        freeze = read_json(freeze_path)
        baseline_path = source_root/'benchmarks/keel-brace-v1/baseline.json'
        require(digest(baseline_path)==BASELINE_MANIFEST_SHA256,'published immutable baseline manifest bytes')
        baseline = read_json(baseline_path)
        require(baseline['commit']=='24fb9a5f02065c0bf0b146b8387c2a9d8a272373' and baseline['sourceTree']=='4d76ac190e9d303ce1657cb2c3659961ef72f611','original immutable Keel baseline identity')
        for item in baseline['files']+baseline['productionAISurvivor']:
            if item['path'].startswith(('experiments/Keel/','AISurvivor/')):
                require(freeze['assets'].get(item['path'])==item['sha256'],'original baseline/production asset '+item['path'])
        for path,wanted in [('docs/improvements/keel-brace-v1.md',PROTOCOL_SHA256),('benchmarks/keel-brace-v1/games.json',GAMES_SHA256),('benchmarks/keel-brace-v1/runtime-pins.json',RUNTIME_PINS_SHA256)]:
            require(freeze['assets'].get(path)==wanted and digest(source_root/path)==wanted,'actual published prospective bytes '+path)
        require(freeze['assets'].get(AUDIT_RELATIVE_PATH) == digest(Path(__file__)), 'executed independent reducer matches frozen source')
        require(profile['sourceAssets'] == freeze['assets'], 'profile source assets equal frozen source assets')
        exported = provenance / 'source'
        if not exported.is_dir():
            exported = evidence / 'provenance/source'
        for rel, wanted in freeze['assets'].items():
            p = exported / rel
            require(p.is_file() and digest(p) == wanted, 'exported frozen source ' + rel)
        python_proof = read_json(provenance/'python-runtime.json')
        require(profile['pythonRuntime']==python_proof and profile['pythonRuntimeSHA256']==digest(provenance/'python-runtime.json'),'full archived Python fingerprint proof/identity binding')
        require(digest(provenance/'build.log')==profile['buildLogSHA256'],'original Java17 build log binding')
        commands = read_json(provenance / 'commands.json')
        plan = read_json(source_root / 'benchmarks/keel-brace-v1/games.json')[stage]
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
        for arm, name in [('baseline','Keel'),('candidate','KeelBrace')]:
            require(participants[arm]['name'] == name and participants[arm]['version'] == '0.1.0', 'participant identity ' + arm)
            require(participants[arm]['jarSHA256'] == freeze['assets']['experiments/' + name + '/' + name + '.jar'], 'participant frozen JAR ' + arm)
        approval_path = evidence / 'approval.json'
        approval = read_json(approval_path)
        require(set(approval) == APPROVAL_KEYS and approval['verdict'] == 'CLEAR' and approval['issue'] == ISSUE and approval['repository'] == 'stephanelee/robocode', 'strict20-key runtime approval')
        for key, value in [('headSHA', expected_head), ('runId', expected_run), ('runAttempt', 1),
                           ('bootId', ident['bootId']), ('workspace', ident['workspace']), ('stage', stage)]:
            require(type(approval[key]) is type(value) and approval[key] == value, 'approval ' + key)
        for key, path in [('profileSHA256', profile_path), ('freezeSHA256', freeze_path),
                          ('gamesSHA256', source_root / 'benchmarks/keel-brace-v1/games.json'),
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
        require(payload['receiptRef'] == 'refs/tags/experiment-receipts/keel-brace-v1-' + stage, 'fixed stage receipt name')
        require(payload['jobRemainingSecondsAtAdmission'] >= 2100, 'receipt admission at least35minutes')
        require(receipt['createdRef']['object']['sha'] == receipt['annotatedTag']['sha'], 'created receipt binds annotated tag SHA')
        require(receipt['annotatedTag']['object']['sha'] == expected_head, 'annotated tag points at exact head')
        dispatch = payload['dispatchAssignment']
        require(hashlib.sha256(dispatch['body'].encode()).hexdigest() == approval['dispatchAssignmentSHA256'], 'receipt exact dispatch body hash')
        require(payload['acceptedComment']['body'].startswith('KEEL_BRACE_RUNTIME_APPROVAL_V1\n'), 'receipt accepted approval API body')
        require(payload['reviewReport']['body'].startswith('KEEL_BRACE_RUNTIME_REVIEW_V1\n'), 'receipt separate reasoned review API body')
        require(profile['runner'] == '02f6d1d8e9346a4aeae1b91e7abb278c98ac1f1d30c605f867758566f5f327cf', 'pinned runner hash')
        require('17.0.21+9' in profile['java17']['version'] and 'Temurin' in profile['java17']['version'] and '25.0.5+8' in profile['java25']['version'] and 'Temurin' in profile['java25']['version'], 'recorded Java17 build then25 battle identities')
    except (KeyError, TypeError, ValueError, OSError) as exc:
        failures.append(type(exc).__name__ + ': ' + str(exc))
    return {'checked': checked, 'failures': failures, 'pass': not failures,
            'scope': 'Archived identities/exported source and receipt bindings only. Live system files, API freshness, remote tag/run cardinality and process start/admission timing require independent API/log evidence review.'}


def canonical_gate_checks(stage, rows, raw):
    groups=raw['groups'];c=raw['checks']
    out=dict(allGamesValid=all(r['valid'] for r in rows),zeroBothArmSkips=all(r['skips']==0 for r in rows),bothArmTiming=all(r['maxMs']<75 for r in rows),candidateCountersValid=True,candidateRetconEligibleRamExposure=raw['exposurePass'],diamondAbove50=c['candidateDiamondMeanAboveHalf'],diamondGamesAbove50=groups['candidate/diamond']['gamesAboveHalf']>=(2 if stage=='screen' else 4),retconAtLeast90=c['candidateRetconMeanAtLeast90Percent'],retconDeltaAtLeast1pp=c['retconDeltaAtLeast1pp'])
    if stage=='screen':
        out.update(diamondDeltaAtLeastMinus3pp=c['diamondDeltaAtLeastMinus3pp'],diamondWinsAtLeast16=c['diamondWinsAtLeast16'],retconWinsAtLeast16=c['retconWinsAtLeast16'])
        for name in ('walls','ais'):out[name+'Within5pp']=c[name+'DeltaAtLeastMinus5pp'];out[name+'WinsAtLeast4']=c[name+'WinsAtLeast4']
    else:out.update(diamondDeltaAtLeastMinus2pp=c['diamondDeltaAtLeastMinus2pp'],diamondWinsAbove60=c['diamondWinsMoreThan60'],retconWinsAtLeast24=c['retconWinsAtLeast24'])
    return out


def reconcile_canonical(evidence, stage, head, population, rows, raw, complete):
    """Never waive producer acquisition failures because observer scores look complete."""
    failures=[];checked=[];declared=None
    def same(a,b,label):
        if type(b) is float:check(type(a) in (int,float) and type(a) is not bool and math.isfinite(a) and math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12),label)
        else:check(type(a) is type(b) and a==b,label)
    try:
        declared=read_json(evidence/'decision.json')
        check(declared['stage']==stage and declared['headSHA']==head,'Canonical exact stage/head')
        check(complete and raw is not None,'Full independently valid population required for canonical acceptance')
        check('integrityFailure' not in declared and 'infrastructureError' not in declared,'Canonical integrity/infrastructure failure precludes acceptance')
        for name in ('failure-prepare.json','failure-approve.json','failure-run.json'):check(not (evidence/name).exists(),'Retained stage acquisition failure '+name)
        check(declared['unplayedIDs']==[],'Canonical no unplayed IDs')
        check([r['id'] for r in declared['games']]==[g['id'] for g in population],'Canonical exact ordered population no duplicate/omission')
        budget=sum(g['rounds'] for g in population)
        for key in ('observedPhysicalRounds','completedPhysicalRounds'):check(type(declared[key]) is int and declared[key]==budget,'Canonical exact full physical budget '+key)
        fields=['ownScore','totalScore','wins',*COUNTS,'turns','hitRate','enemyHitRate','firingPer1000Turns','powerMix','enemyPowerMix','skips','maxMs','exposure','recordingSHA256']
        for registered,independent,original in zip(population,rows,declared['games']):
            check(all(type(original[k]) is type(v) and original[k]==v for k,v in registered.items()),'Canonical exact registered game identity '+registered['id'])
            check(original['status']=='valid' and not original.get('error') and not original.get('partialRecordingError'),'Canonical acquisition status valid '+registered['id'])
            check(not (evidence/'games'/registered['id']/'failure.json').exists(),'Retained game acquisition failure '+registered['id'])
            for key in fields:same(original[key],independent[key],'Canonical independent game metric '+registered['id']+':'+key)
            same(original['share'],independent['share'],'Canonical game score share')
            ids=list(range(1,registered['rounds']+1))
            for key in ('observedPhysicalRoundIDs','completedPhysicalRoundIDs'):check(original[key]==ids and all(type(n) is int for n in original[key]),'Canonical full round population '+key)
            for key in ('observedPhysicalRounds','completedPhysicalRounds'):check(type(original[key]) is int and original[key]==registered['rounds'],'Canonical game round budget '+key)
            check(len(original['perRound'])==len(independent['perRound']),'Canonical complete per-round metrics')
            for a,b in zip(original['perRound'],independent['perRound']):
                for key in ('round','turns','ownScore','totalScore','share','win',*COUNTS,'maxMs','exposure'):same(a[key],b[key],'Canonical independent per-round metric '+key)
        expected_checks=canonical_gate_checks(stage,rows,raw)
        check(set(declared['checks'])==set(expected_checks) and all(type(declared['checks'][k]) is bool and declared['checks'][k]==v for k,v in expected_checks.items()),'Canonical all registered independent gates match')
        for key,wanted in [('diamondMean',raw['groups']['candidate/diamond']['mean']['decimal']),('diamondDelta',raw['diamondDifference']['decimal']),('retconMean',raw['groups']['candidate/retcon']['mean']['decimal']),('retconDelta',raw['retconDifference']['decimal'])]:same(declared[key],wanted,'Canonical exact game-weighted metric '+key)
        same(declared['candidateRetconEligibleRamScans'],raw['candidateRetconEligibleRamCorrectedScans'],'Canonical stage-local Retcon exposure')
        wanted_sd={k:v['sampleSDPercentagePoints']/100 if v['sampleSDPercentagePoints'] is not None else None for k,v in raw['groups'].items() if k.rsplit('/',1)[1] in ('diamond','retcon')}
        check(set(declared['betweenGameShareSD'])==set(wanted_sd),'Canonical complete variation groups')
        for key,value in wanted_sd.items():same(declared['betweenGameShareSD'][key],value,'Canonical game variation '+key)
        expected='inconclusive' if not raw['exposurePass'] else ('progress; confirmation open' if stage=='screen' else 'confirmed') if raw['allPass'] else 'reject' if stage=='screen' else 'not confirmed'
        check(declared['decision']==expected,'Canonical decision reconciles independent full raw gates')
        checked.append('Original canonical decision agrees with independent raw population/budgets/metrics/gates and has no retained acquisition failure')
    except (KeyError,TypeError,ValueError,OSError) as exc:failures.append(type(exc).__name__+': '+str(exc))
    return dict(checked=checked,failures=failures,canonicalDecision=declared.get('decision') if isinstance(declared,dict) else None,**{'pass':not failures})


def audit_stage(evidence, source_root, stage, expected_head, expected_run, provider_evidence=None, transport=None):
    plan_path = source_root / 'benchmarks/keel-brace-v1/games.json'
    check(digest(plan_path)==GAMES_SHA256,'Source-root registry differs from prospectively published immutable bytes')
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
        own_name = 'KeelBrace' if game['bot'] == 'candidate' else 'Keel'
        check(game['participantName']==own_name and game['participantVersion']=='0.1.0','Registered arm identity differs')
        opponent=(game['opponentName'],game['opponentVersion'])
        row = reduce_game(found[0], d/'markers.txt', own_name, *opponent, game['rounds'], game['bot']=='candidate', game['participantVersion'])
        row['game'] = game
        if row['valid']:
            if game['opponent'] == 'ais':
                ais = d / 'ais.jsonl';model = d / 'ais-model.dat'
                try:
                    metrics = [json_data(l) for l in ais.read_text().splitlines()]
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
    authority = audit_authority(evidence, source_root, stage, expected_head, expected_run, provider_evidence)
    storage = audit_storage(evidence, stage, expected_run, provider_evidence, transport)
    runner_image = audit_runner_image(evidence, expected_head, expected_run, provider_evidence)
    upstream_screen = audit_upstream_screen(source_root, stage, provider_evidence)
    canonical=reconcile_canonical(evidence,stage,expected_head,population,rows,raw,complete)
    raw_disposition = raw['rawDisposition'] if complete else 'INVALID/INCOMPLETE'
    decision = 'INVALID/INCOMPLETE' if not canonical['pass'] or raw_disposition=='INVALID/INCOMPLETE' else 'PROVENANCE-INCOMPLETE' if not provenance['pass'] or not authority['pass'] or not storage['pass'] or not upstream_screen['pass'] else runner_image.get('status','PROVENANCE-INCOMPLETE') if not runner_image['pass'] else raw_disposition
    return {'schemaVersion': 1, 'stage': stage, 'independentReducer': True, 'gamesSHA256': digest(plan_path),
            'expectedHead': expected_head, 'expectedRun': expected_run, 'registeredGames': len(population),
            'registeredRoundBudget': sum(g['rounds'] for g in population), 'validGames': rows,
            'invalidOrPartialGames': partial, 'unrecordedGameIds': missing, 'protocolFailures': protocol_failures,
            'observedPhysicalRoundCountLowerBound': sum(len(r.get('observedPhysicalRoundIds', [])) for r in rows + partial),
            'completeRawPopulation': complete, 'rawGates': raw, 'provenance': provenance, 'authority':authority, 'storage':storage, 'runnerImage':runner_image, 'upstreamScreen':upstream_screen,
            'observedTimingOrSkipFailures': {r['game']['id']:r['partialMarkerObservation'] for r in rows+partial if r.get('partialMarkerObservation',{}).get('skips') or r.get('partialMarkerObservation',{}).get('reportedTimingViolations')},
            'rawDisposition':raw_disposition, 'decision':decision, 'canonicalReconciliation':canonical,
            'limits': ['No source/JAR/model/approval changes are authorized by this audit.',
                       'Do not pool historical games, screen or confirmation.',
                       'Round learning is correlated; equal-weight cold games are the comparison units.',
                       'Diamond private skipped turns are not observable.',
                       'Candidate exposure is source/runtime-bound aggregate activation, not an independently reconstructed private callback trace or score causality.',
                       'Raw INCONCLUSIVE due to zero stage-local Retcon eligible exposure never opens confirmation or promotion.',
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
    return {'scope': 'Historical immutable Keel confirmation verification ONLY; no KeelBrace games or strength evidence.',
            'sourceDecisionSHA256': digest(root / 'decision-confirmation.json'), 'games': len(rows),
            'rounds': sum(r['rounds'] for r in rows), 'mean': ratio(mean), 'wins': sum(r['wins'] for r in rows),
            'recordingHashes': {r['game']['id']: r['recordingSHA256'] for r in rows},
            'mismatches': mismatches, 'pass': not mismatches}



def json_data(raw):
    def unique(pairs):
        out = {}
        for k,v in pairs:
            check(k not in out, 'Duplicate JSON key: '+k);out[k]=v
        return out
    return json.loads(raw,object_pairs_hook=unique,parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Nonfinite JSON: '+x)))


def evidence_path(path):
    return path if isinstance(path, ZipPath) else Path(path)


def safe_zip_name(name):
    check(type(name) is str and name and not name.startswith(('/', '\\')) and '\\' not in name
          and re.match('[A-Za-z]:',name) is None, 'Unsafe ZIP member: '+str(name))
    parts=name.rstrip('/').split('/')
    check(all(p not in ('','.', '..') for p in parts),'Unsafe/aliased ZIP member: '+name)


class ZipStore:
    """Read-only virtual paths over an original ZIP; never extracts/re-ZIPs data."""
    def __init__(self,path):
        self.path=Path(path) if isinstance(path,(str,Path)) else "retained-original-ZIP-parts";self.archive=zipfile.ZipFile(path);self.files={}
        for item in self.archive.infolist():
            safe_zip_name(item.filename)
            check(item.filename not in self.files,'Duplicate ZIP member: '+item.filename)
            check((item.external_attr >> 16)&0o170000 not in (0o120000,0o010000,0o060000,0o020000,0o140000),'Nonregular ZIP member: '+item.filename)
            check(not item.flag_bits&1,'Encrypted ZIP member not supported')
            self.files[item.filename]=item
    def root(self,prefix=''):
        if prefix:safe_zip_name(prefix.rstrip('/'))
        return ZipPath(self,prefix.strip('/'))
    def close(self):self.archive.close()


class ZipPath:
    def __init__(self,store,name):self.store,self.member=store,name
    def __truediv__(self,name):
        name=str(name);safe_zip_name(name)
        return ZipPath(self.store,self.member+'/'+name if self.member else name)
    def __str__(self):return str(self.store.path)+'!/'+self.member
    @property
    def name(self):return self.member.rsplit('/',1)[-1]
    @property
    def parent(self):return ZipPath(self.store,self.member.rsplit('/',1)[0] if '/' in self.member else '')
    def exists(self):return self.is_file() or self.is_dir()
    def is_file(self):return self.member in self.store.files and not self.store.files[self.member].is_dir()
    def is_dir(self):return any(x.startswith(self.member.rstrip('/')+'/') for x in self.store.files) if self.member else bool(self.store.files)
    def is_symlink(self):return False # ZipStore rejects them before exposing any path.
    def stat(self):return SimpleNamespace(st_size=self.store.files[self.member].file_size)
    def open(self,mode='r',encoding='utf-8'):
        check(mode in ('r','rb','rt'),'ZIP evidence is read-only')
        raw=self.store.archive.open(self.member,'r')
        return raw if mode=='rb' else io.TextIOWrapper(raw,encoding=encoding)
    def read_bytes(self):
        with self.open('rb') as f:return f.read()
    def read_text(self):return self.read_bytes().decode('utf-8')
    def glob(self,pattern):
        check('/' not in pattern,'Only direct-child evidence globs allowed')
        prefix=self.member.rstrip('/')+'/' if self.member else ''
        return [ZipPath(self.store,n) for n in self.store.files if n.startswith(prefix)
                and '/' not in n[len(prefix):] and fnmatch.fnmatchcase(n[len(prefix):],pattern)]


def audit_archive(original_zip, expected_sha256, extracted_root=None, original_size_limit=True):
    """Stream every original member including CRC, hashes and ZIP metadata, bounded memory."""
    original_zip=Path(original_zip)
    check(re.fullmatch('[0-9a-f]{64}',expected_sha256) is not None,'Expected original ZIP SHA256 required')
    if original_size_limit:check(original_zip.stat().st_size <= 2*1024**3,'Original ZIP exceeds registered 2GiB custody limit')
    check(digest(original_zip)==expected_sha256,'Original API artifact ZIP SHA256 differs')
    entries={};seen=set();metadata=[]
    with zipfile.ZipFile(original_zip) as z:
        for item in z.infolist():
            name=item.filename;safe_zip_name(name)
            check(name not in seen,'Duplicate ZIP member: '+name);seen.add(name)
            mode=item.external_attr >> 16
            check(mode&0o170000 in (0,0o040000,0o100000),'Nonregular ZIP member: '+name)
            check(not item.flag_bits&1,'Encrypted ZIP member not supported')
            metadata.append(dict(path=name,bytes=item.file_size,compressedBytes=item.compress_size,CRC=item.CRC,
                                 compression=item.compress_type,flagBits=item.flag_bits,externalAttributes=item.external_attr,
                                 createSystem=item.create_system,dateTime=list(item.date_time),directory=item.is_dir()))
            if item.is_dir():continue
            h=hashlib.sha256();g=hashlib.sha1(b'blob '+str(item.file_size).encode()+b'\0');size=0
            with z.open(item) as f:
                for b in iter(lambda:f.read(1024*1024),b''):h.update(b);g.update(b);size+=len(b)
            check(size==item.file_size,'Original member length mismatch: '+name)
            if extracted_root is not None:
                out=Path(extracted_root)/name
                check(out.is_file() and not out.is_symlink(),'Missing/nonregular extracted original: '+name)
                check(out.stat().st_size==size and digest(out)==h.hexdigest(),'Extracted byte mismatch: '+name)
            entries[name]=dict(bytes=size,sha256=h.hexdigest(),gitBlobSHA=g.hexdigest())
    recordings={n:v for n,v in entries.items() if n.endswith('.battle.gz')}
    return dict(scope='Original API-digest and every whole original ZIP member streamed/CRC checked; no games or extraction.',
                originalZIP=str(original_zip),originalZIPBytes=original_zip.stat().st_size,originalZIPSHA256=expected_sha256,
                regularFiles=entries,fileCount=len(entries),zipMetadata=metadata,wholeRecordings=recordings,**{'pass':True})


def timestamp(value):
    check(type(value) is str,'Timestamp must be a string')
    d=datetime.datetime.fromisoformat(value.replace(' UTC','+00:00').replace('Z','+00:00'))
    check(d.tzinfo is not None,'Timestamp lacks UTC offset')
    return d


def comment_identity(comment):
    return {k:comment[k] for k in ('id','node_id','url','issue_url','html_url','created_at','updated_at','body')} | {'author':{k:comment['user'][k] for k in ('id','login','type')},'bodySHA256':hashlib.sha256(comment['body'].encode('utf-8')).hexdigest()}


def comment_payload(comment,prefix):
    check(comment['user']['login']=='stephanelee' and comment['user']['id']==30698997,'Wrong approval account')
    check(comment['body'].startswith(prefix+'\n'),'Wrong comment prefix')
    return json_data(comment['body'][len(prefix)+1:])


def targeted_revocation(comment, approval, accepted_comment):
    """Only assigned-account, accepted-approval-target revocations remove authority."""
    if not comment.get('body','').startswith('KEEL_BRACE_RUNTIME_REVOCATION_V1\n'):return False
    user=comment.get('user',{})
    if user.get('login')!='stephanelee' or user.get('id')!=30698997:return False
    revoke=json_data(comment['body'].split('\n',1)[1])
    check(type(revoke) is dict,'Assigned-account revocation must be JSON object')
    if revoke.get('approvalCommentId')!=accepted_comment['id']:return False
    check(set(revoke)=={'schemaVersion','verdict','repository','issue','stage','headSHA','runId','runAttempt','bootId','artifactId','reviewerTask','approvalCommentId','reason'},'Targeted revocation exact13 keys')
    check(all(type(revoke[k]) is int for k in ('schemaVersion','issue','runId','runAttempt','artifactId','approvalCommentId')),'Targeted revocation integer identities')
    check(revoke['schemaVersion']==1 and revoke['verdict']=='REVOKE' and revoke['approvalCommentId']==accepted_comment['id'],'Targeted revocation schema/verdict/approval')
    check(all(revoke[k]==approval[k] for k in ('repository','issue','stage','headSHA','runId','runAttempt','bootId','artifactId','reviewerTask')),'Targeted revocation exact accepted worker/reviewer')
    check(type(revoke['reason']) is str and bool(revoke['reason'].strip()) and timestamp(comment['created_at'])>timestamp(accepted_comment['created_at']),'Targeted revocation reason and later provider timestamp')
    return True


def audit_authority(evidence, source_root, stage, expected_head, expected_run, provider=None):
    """Independent archived authority/API audit. Missing provider originals fails closed.

    provider is a read-only directory supplied by the independent collector, not a network client.
    Required run.json/jobs.json/workflow-runs.json/receipt-ref.json/receipt-tag.json and
    comments/<accepted-ID>.json are documented in README. Live worker bytes remain a limit.
    """
    checked,failures=[],[]
    def require(ok,label):(checked if ok else failures).append(label)
    try:
        authority=read_json(evidence/'accepted-authority.json');approval=read_json(evidence/'approval.json')
        require(authority['approvalPayload']==approval,'accepted authority exact original approval payload')
        require(authority['approvalPayloadSHA256']==hashlib.sha256(canonical(approval)).hexdigest(),'accepted payload canonical digest')
        require(set(approval)==APPROVAL_KEYS and all(type(approval[k]) is int for k in ('schemaVersion','issue','runId','runAttempt','artifactId')) and approval['schemaVersion']==1 and approval['issue']==ISSUE and approval['runAttempt']==1,'new issue114 strict20-key typed approval')
        require(approval['verdict']=='CLEAR' and approval['repository']=='stephanelee/robocode' and approval['stage']==stage and approval['headSHA']==expected_head and approval['runId']==expected_run,'approval exact registered authority identity')
        comments=authority['comments'];accepted={}
        for file,prefix in [('approval-comment.json','KEEL_BRACE_RUNTIME_APPROVAL_V1'),('review-report-comment.json','KEEL_BRACE_RUNTIME_REVIEW_V1'),('dispatch-assignment-comment.json','KEEL_BRACE_DISPATCH_ASSIGNMENT_V1')]:
            original=read_json(evidence/file);require(comments[file]==comment_identity(original),'exact normalized accepted comment pin '+file)
            require(authority['originalAcceptedCopiesSHA256'][file]==digest(evidence/file),'accepted original bytes '+file)
            accepted[file]=(original,comment_payload(original,prefix))
        ac,a=accepted['approval-comment.json'];rc,report=accepted['review-report-comment.json'];dc,dispatch=accepted['dispatch-assignment-comment.json']
        require(a==approval,'approval comment exactly strict payload')
        dispatch_keys={'schemaVersion','repository','issue','stage','headSHA','reviewerTask','reviewerAccount','implementationTasks','assignmentToolRecord','assignmentToolRecordSHA256'}
        require(set(dispatch)==dispatch_keys and type(dispatch['schemaVersion']) is int and dispatch['schemaVersion']==1 and dispatch['issue']==ISSUE,'strict10-key fresh dispatch')
        require(dispatch['stage']==stage and dispatch['headSHA']==expected_head and dispatch['repository']=='stephanelee/robocode','dispatch stage/head/repository')
        require(dispatch['reviewerTask']==approval['reviewerTask'] and dispatch['reviewerTask'] not in dispatch['implementationTasks'] and dispatch['reviewerAccount']=='stephanelee','separate actual reviewer identity')
        record=dispatch['assignmentToolRecord']
        require(record['reviewerTask']==dispatch['reviewerTask'] and record['activeBeforeDispatch'] is True,'recorded active-before-dispatch task')
        require(hashlib.sha256(canonical(record)).hexdigest()==dispatch['assignmentToolRecordSHA256'],'actual assignment record canonical digest')
        require(hashlib.sha256(dc['body'].encode()).hexdigest()==approval['dispatchAssignmentSHA256'],'exact dispatch comment body digest')
        require(set(report)=={'approvalPayloadSHA256','reviewerTask','reviewReport','agentToolProvenance'},'strict4-key reasoned report')
        require(report['approvalPayloadSHA256']==authority['approvalPayloadSHA256'] and report['reviewerTask']==approval['reviewerTask'] and type(report['reviewReport']) is str and bool(report['reviewReport'].strip()),'separate reasoned review binds approval')
        require(report['agentToolProvenance']['reviewerTask']==approval['reviewerTask'] and report['agentToolProvenance']['reportGeneratedBeforeGames'] is True,'actual independent pregame tool provenance')
        artifact=read_json(evidence/'artifact.json');finalized=timestamp(artifact['updated_at'])
        require(finalized < timestamp(rc['created_at']) < timestamp(ac['created_at']) <= finalized+datetime.timedelta(seconds=1800),'artifact/report/approval ordered inside30minute deadline')
        require(rc['id']<ac['id'],'provider report ID precedes approval ID')
        receipt=read_json(evidence/'receipt.json');payload=receipt['payload'];tag=receipt['annotatedTag']
        require(payload['acceptedAuthority']==authority,'permanent receipt exact original authority chain')
        require(payload['acceptedAuthoritySHA256']==hashlib.sha256(canonical(authority)).hexdigest(),'receipt canonical authority digest')
        tag_payload=json_data(tag['message']);require(tag_payload==payload,'annotated tag message equals exact receipt payload')
        require(tag['object']['sha']==expected_head and tag['object']['type']=='commit','permanent receipt commit binding')
        require(receipt['createdRef']['object']['sha']==tag['sha'] and receipt['createdRef']['object']['type']=='tag','fixed annotated receipt object')
        population=read_json(source_root/'benchmarks/keel-brace-v1/games.json')[stage]
        phases=['before-receipt']+['before-'+g['id'] for g in population]
        require(sorted(p.name for p in (evidence/'approval-revalidations').glob('*.json'))==sorted(p+'.json' for p in phases),'exact pre-receipt/per-game revalidation population')
        previous=timestamp(ac['created_at'])
        starts=[]
        for phase in phases:
            v=read_json(evidence/'approval-revalidations'/(phase+'.json'))
            require(v['status']=='valid' and v['stage']==stage and v['phase']==phase,'successful live revalidation '+phase)
            require(v['acceptedAuthoritySHA256']==payload['acceptedAuthoritySHA256'],'immutable authority at '+phase)
            now=timestamp(v['checkedAtUTC']);require(now>=previous,'serial revalidation timestamps '+phase);previous=now
            for file,(c,_) in accepted.items():require(comment_identity(v['currentComments'][file])==comment_identity(c),'unchanged live accepted comment '+phase+':'+file)
            require(v['currentArtifact']==artifact,'unchanged live finalized artifact '+phase)
            require(v['currentPullRequest']['head']['sha']==expected_head and ('KEEL-BRACE-SOURCE-CLEARED '+expected_head) in v['currentPullRequest']['body'],'live source clearance '+phase)
            require(len(v['scannedComments'])<=2000,'bounded comment scan '+phase)
            require(all(any(comment_identity(x)==comment_identity(c) for x in v['scannedComments']) for c,_ in accepted.values()),'accepted originals included in full comment scan '+phase)
            for c in v['scannedComments']:
                if targeted_revocation(c,approval,ac):require(False,'authorized exact-target runtime revocation present '+phase)
            if phase!='before-receipt':
                require(v['currentReceiptRef']['object']['sha']==tag['sha'] and v['currentReceiptTag']==tag,'unchanged permanent remote receipt '+phase)
                game=next(g for g in population if 'before-'+g['id']==phase);folder=evidence/'games'/game['id']
                attempt=read_json(folder/'attempt.json');prepared=read_json(folder/'prepared-game.json')
                require(attempt['game']==game and attempt['headSHA']==expected_head,'attempt registered exact game/head '+game['id'])
                require(all(attempt[k]==prepared[k] for k in ('command','environment','cwd')),'attempt approved command/env/cwd '+game['id'])
                require(attempt['freshProcess'] is True and attempt['freshWritableDirectory'] is True and attempt['importedModels']==[],'fresh owned process/cold directory '+game['id'])
                start=timestamp(attempt['startUTC']);require(start>=now,'admission before child '+game['id'])
                teardown=read_json(folder/'teardown.json')
                require(teardown.get('status')!='failed' and type(teardown['ownedProcessGroup']) is int and teardown['ownedProcessGroup']>0 and type(teardown['termSent']) is bool and type(teardown['killSent']) is bool,'recorded bounded owned teardown '+game['id'])
                elapsed=teardown['elapsedMonotonicSeconds'];ended=timestamp(teardown['endUTC'])
                require(type(elapsed) in (int,float) and math.isfinite(elapsed) and 0<=elapsed<=1800 and teardown['hardDeadlineRespected'] is True,'actual monotonic child+drain+teardown<=1800seconds '+game['id'])
                require(type(teardown['hardLimitSeconds']) is int and teardown['hardLimitSeconds']==1800 and type(teardown['runStopReserveSeconds']) is int and teardown['runStopReserveSeconds']==16,'exact hard child limit/reserve '+game['id'])
                require(ended>=start,'child UTC end follows start '+game['id']);starts.append((game['id'],start,ended));previous=ended

        if provider is None:
            failures.append('Independent current provider API bundle not supplied; see README. No full acceptance claimed.')
        else:
            provider=evidence_path(provider);run=read_json(provider/'run.json');jobs=read_json(provider/'jobs.json');runs=read_json(provider/'workflow-runs.json')
            require(run['id']==expected_run and run['head_sha']==expected_head and run['run_attempt']==1 and run['status']=='completed' and run['conclusion']=='success','current completed/success run attempt1/head')
            require(timestamp(dc['created_at'])<timestamp(run['created_at']),'actual assignment predates actual workflow run')
            matching=[x for x in runs['workflow_runs'] if x['head_sha']==expected_head]
            require(len(matching)==1 and matching[0]['id']==expected_run and matching[0]['run_attempt']==1,'sole stage workflow run, no duplicate/rerun')
            ref=read_json(provider/'receipt-ref.json');current_tag=read_json(provider/'receipt-tag.json')
            require(ref['object']['sha']==tag['sha'] and current_tag==tag,'current permanent remote receipt exact original')
            for file,(c,_) in accepted.items():require(comment_identity(read_json(provider/'comments'/(str(c['id'])+'.json')))==comment_identity(c),'current provider accepted comment '+file)
            executing=[j for j in jobs['jobs'] if any('run fixed population' in s['name'].lower() for s in j['steps'])]
            require(len(executing)==1,'one registered execution job')
            if executing:
                job=executing[0];require(job['conclusion']=='success','registered job succeeded')
                step=next(s for s in job['steps'] if 'run fixed population' in s['name'].lower());end=timestamp(step['completed_at']);job_start=timestamp(job['started_at'])
                require(timestamp(job['completed_at'])-job_start<=datetime.timedelta(minutes=340),'full job registered340minute cap including export')
                for i,(gid,start,ended) in enumerate(starts):
                    require(ended<=end,'recorded child completes before provider execution step '+gid)
                    if i:require(start>=starts[i-1][2],'serial children do not overlap '+gid)
                    require(job_start+datetime.timedelta(minutes=340)-start>=datetime.timedelta(minutes=35),'at least35minutes remaining before child '+gid)
    except (KeyError,TypeError,ValueError,OSError,StopIteration) as exc:failures.append(type(exc).__name__+': '+str(exc))
    return dict(checked=checked,failures=failures,**{'pass':not failures},scope='Archived original/live-revalidation bindings plus separately collected current provider APIs. Does not remotely rehash unexported system binaries or reconstruct every private callback. Controller storage/byte-limit proofs must also be reviewed against actual protocol/source and provider job logs before final approval.')


def audit_storage(evidence, stage, expected_run, provider=None, transport=None):
    """Agreed premeasurement artifact interface; no claim to remote disk remeasurement."""
    checked,failures=[],[]
    def require(ok,label):(checked if ok else failures).append(label)
    try:
        dispatch=read_json(evidence/'dispatch-assignment.json');comment=read_json(evidence/'dispatch-assignment-comment.json')
        storage=dispatch['assignmentToolRecord']['storagePreflight']
        require(set(storage)=={'host','volume','observedAtUTC','freeBytes','toolEvidence'},'exact root storage-preflight fields')
        require(type(storage['freeBytes']) is int and storage['freeBytes']>=8*1024**3,'audit volume recorded >=8GiB')
        require(bool(storage['host']) and bool(storage['volume']) and bool(storage['toolEvidence']),'actual audit host/volume/tool evidence')
        require(timestamp(storage['observedAtUTC'])<=timestamp(comment['created_at']),'audit-storage sample precedes assignment')
        sample_keys={'schemaVersion','phase','observedAtUTC','volume','freeBytes','minimumFreeBytes','stageOutputBytes','stageOutputLimitBytes'}
        def sample(name,minimum):
            v=read_json(evidence/'storage-admissions'/(name+'.json'))
            require(set(v)==sample_keys and type(v['schemaVersion']) is int and v['schemaVersion']==1 and v['phase']==name,'typed exact storage sample '+name)
            require(type(v['freeBytes']) is int and type(v['minimumFreeBytes']) is int and v['minimumFreeBytes']==minimum and v['freeBytes']>=minimum,'registered free-space admission '+name)
            require(type(v['stageOutputBytes']) is int and 0<=v['stageOutputBytes']<=4*1024**3 and type(v['stageOutputLimitBytes']) is int and v['stageOutputLimitBytes']==4*1024**3,'registered owned-output admission '+name)
            require(bool(v['volume']),'worker volume recorded '+name)
            return v
        prep=sample('before-preparation',20*1024**3);receipt=sample('before-receipt',8*1024**3)
        require(timestamp(prep['observedAtUTC'])<=timestamp(receipt['observedAtUTC']),'storage preparation before receipt')
        commands=read_json(evidence/'commands.json')
        for command in commands:
            gid=command['id'];v=sample('before-'+gid,8*1024**3);attempt=read_json(evidence/'games'/gid/'attempt.json')
            require(timestamp(receipt['observedAtUTC'])<=timestamp(v['observedAtUTC'])<=timestamp(attempt['startUTC']),'storage sample before child '+gid)
        monitor=read_json(evidence/'storage-monitor.json')
        require(set(monitor)=={'schemaVersion','firstObservedAtUTC','lastObservedAtUTC','observations','maximumStageOutputBytes','limitBytes','limitExceeded'},'exact storage monitor fields')
        require(type(monitor['schemaVersion']) is int and monitor['schemaVersion']==1 and type(monitor['observations']) is int and monitor['observations']>0,'storage monitor recorded observations')
        require(timestamp(monitor['firstObservedAtUTC'])<=timestamp(monitor['lastObservedAtUTC']),'storage monitor chronology')
        require(type(monitor['maximumStageOutputBytes']) is int and 0<=monitor['maximumStageOutputBytes']<=4*1024**3 and type(monitor['limitBytes']) is int and monitor['limitBytes']==4*1024**3 and monitor['limitExceeded'] is False,'recorded continuous output monitor<=4GiB')
        if provider is None or transport is None:failures.append('Original ZIP, separate custody-validation ZIP and external provider metadata are all required; missing is not acceptance')
        else:
            provider=evidence_path(provider);run=read_json(provider/'run.json')
            require(timestamp(run['created_at'])<=timestamp(prep['observedAtUTC']),'worker preparation storage sample follows workflow creation')
            age=timestamp(run['created_at'])-timestamp(storage['observedAtUTC'])
            require(datetime.timedelta()<=age<=datetime.timedelta(minutes=60),'audit-storage sample<=60minutes before actual workflow creation')
            artifact=read_json(provider/'evidence-artifact.json');companion=read_json(provider/'custody-artifact.json')
            require(type(artifact['size_in_bytes']) is int and artifact['size_in_bytes']==transport['originalZIPBytes'] and 0<artifact['size_in_bytes']<=2*1024**3,'actual original provider ZIP size<=2GiB')
            require(artifact['digest']=='sha256:'+transport['originalZIPSHA256'],'actual original provider ZIP digest')
            require(companion['size_in_bytes']==transport['custodyZIPBytes'] and companion['digest']=='sha256:'+transport['custodyZIPSHA256'],'actual separate companion ZIP size/digest')
            require(artifact['id']!=companion['id'],'original and companion artifacts are distinct')
            for x in (artifact,companion):require(x['workflow_run']['id']==expected_run and x['workflow_run']['head_sha']==run['head_sha'],'artifact exact workflow run/head '+str(x['id']))
            require(companion['name']=='keel-brace-v1-'+stage+'-custody-validation-'+str(expected_run),'registered separate companion artifact name')
            custody=transport['custodyRoot'];captured=read_json(custody/'evidence-artifact.json');export=read_json(custody/'export-validation.json')
            require(captured==artifact and export['artifact']==artifact,'companion captures exact ORIGINAL provider artifact API')
            base={'schemaVersion','stage','status','observedAtUTC','originalZIPMaximumBytes','rawDecisionSHA256','artifact'}
            require(base<=set(export)<=base|{'error'} and type(export['schemaVersion']) is int and export['schemaVersion']==1,'postexport validation exact fields')
            require(export['stage']==stage and export['status']=='valid' and not export.get('error'),'postexport validation is valid without error')
            require(type(export['originalZIPMaximumBytes']) is int and export['originalZIPMaximumBytes']==2*1024**3,'postexport registered ZIP cap')
            require(export['rawDecisionSHA256']==digest(evidence/'decision.json'),'original raw decision not rewritten for custody')
            require(timestamp(artifact['updated_at'])<=timestamp(export['observedAtUTC'])<=timestamp(run['updated_at']),'postexport observation after original finalization before completed API checkpoint')
            require(run['status']=='completed' and run['conclusion']=='success','completed/success includes postexport validation')
    except (KeyError,TypeError,ValueError,OSError) as exc:failures.append(type(exc).__name__+': '+str(exc))
    return dict(checked=checked,failures=failures,**{'pass':not failures},scope='Recorded controller observations/monitor plus exact original and separate companion artifact bytes and external final APIs. Source review must prove sampling/wiring and first-fault enforcement. Archived samples are not an independent remote live-disk scan or byte-level proof of unobserved transient peaks.')


AMENDMENT_POLICY_SHA256='7429753f1d72c888718bd8965a674c7d7f18003f7f9edf3dee309bbb30976edf'
AMENDMENT_DOCUMENT_SHA256='85229f18285ce4dabbd076d5f8e56d0a6ec93174c5ab98bfac56ff6553dabb84'
QUALIFIED_POLICY_ID='keel-brace-qualified-native-observations-v1'
QUALIFIED_NATIVE=dict(image='ubuntu-26.04',version='20261025.179.1',includedSoftwareURL='https://github.com/actions/runner-images/blob/ubuntu26/20261025.179/images/ubuntu/Ubuntu2604-Readme.md',imageReleaseURL='https://github.com/actions/runner-images/releases/tag/ubuntu26%2F20261025.179')
QUALIFIED_ENV=dict(ImageOS='ubuntu26',ImageVersion='20261026.180.1')


def strict_native_claim(raw, source):
    check(len(raw)<=1048576,'Native metadata cap; no prefixes accepted')
    data=json_data(raw);check(type(data) is list,'Native metadata group/detail array required')
    group,detail=('Group','Detail') if source=='setup_info' else ('group','detail') if source=='imagedata' else (None,None)
    check(group is not None,'Unregistered native source')
    matches=[]
    for entry in data:
        check(type(entry) is dict and set(entry)=={group,detail} and type(entry[group]) is str and type(entry[detail]) is str,'Native exact source-backed keys; aliases/ambiguity rejected')
        if entry[group]=='Runner Image':matches.append(entry[detail])
    check(len(matches)==1,'Native exactly one Runner Image group')
    fields={}
    for line in matches[0].split('\n'):
        key,separator,value=line.partition(': ');check(separator and key not in fields and bool(value),'Native malformed/duplicate detail')
        fields[key]=value
    names={'Image':'image','Version':'version','Included Software':'includedSoftwareURL','Image Release':'imageReleaseURL'}
    check(set(fields)==set(names),'Native exact four fields')
    claim={names[k]:v for k,v in fields.items()};check(claim==QUALIFIED_NATIVE,'Native qualified tuple literal mismatch')
    return claim


def strict_environment_claim(raw):
    check(len(raw)<=1048576,'Environment metadata cap; no prefixes accepted')
    text=raw.decode('utf-8');values={}
    for line in text.split('\n'):
        if not line or line.startswith('#'):continue
        match=re.fullmatch(r'([A-Za-z_][A-Za-z0-9_]*)=(.*)',line);check(match is not None,'Literal environment assignment required')
        key,value=match.groups()
        if key not in QUALIFIED_ENV:continue
        check(key not in values,'Duplicate required environment name')
        if value.startswith(('"',"'")):
            check(len(value)>=2 and value[-1]==value[0],'Environment unterminated quote');value=value[1:-1]
        check(value and not any(x in value for x in ('$', '`', '\n', '\r', '"',"'")),'Environment expansion/empty/ambiguous required value')
        values[key]=value
    check(values==QUALIFIED_ENV,'Both literal environment fields must match separate qualified tuple')
    return values


def qualified_provider_header(raw):
    """Original raw job-log bytes only; stop before actual workflow output."""
    check(type(raw) is bytes,'Original provider log bytes required')
    text=raw.decode('utf-8');groups=[];systems=[];active=None;lines=[];boundary=False;prologue=False
    for index,line in enumerate(text.split('\n')):
        check(index<2048,'Startup parser cap reached without bounded scope')
        line=line[:-1] if line.endswith('\r') else line
        if index==0:
            check(re.fullmatch(r"(?:\ufeff)?(?:\d{4}-\d{2}-\d{2}T[0-9:.]+Z )?Current runner version: '[0-9]+\.[0-9]+\.[0-9]+'",line) is not None,'Original complete provider log must begin with actual runner startup prologue');prologue=True
        line=re.sub(r'^\d{4}-\d{2}-\d{2}T[0-9:.]+Z ','',line)
        if line.startswith(('Complete job name: ','##[group]Run ','##[group]Complete job')):
            check(active is None,'Unclosed startup group at workflow boundary');boundary=True;break
        if line.startswith('##[group]'):
            check(active is None,'Nested startup group');active=line[len('##[group]'):];lines=[]
        elif line=='##[endgroup]':
            check(active is not None,'Unmatched startup group end')
            if active=='Runner Image':groups.append(lines)
            elif active=='Operating System':systems.append(lines)
            active=None;lines=[]
        elif active is not None:lines.append(line)
    check(prologue and boundary and active is None and len(groups)==len(systems)==1,'Complete unambiguous startup image/OS scope required')
    claim=strict_native_claim(json.dumps([dict(group='Runner Image',detail='\n'.join(groups[0]))]).encode(),'imagedata')
    check(len(systems[0])==3 and systems[0][0]=='Ubuntu' and re.fullmatch(r'26\.04(?:\.[0-9]+)?',systems[0][1]) is not None and systems[0][2]=='LTS','Qualified startup Ubuntu26.04 OS')
    return claim


def runner_image_header(lines):
    """Parse only actual Setup-job groups; never provisioner version or later shell ENV."""
    image_groups=[];os_groups=[];active=None;values=[]
    for index,line in enumerate(lines):
        if index>=2048:break
        line=line.rstrip('\r\n')
        if '\t' in line:
            parts=line.split('\t',2)
            if len(parts)!=3 or parts[1]!='Set up job':continue
            line=parts[2]
        line=line.lstrip('\ufeff')
        line=re.sub(r'^\d{4}-\d{2}-\d{2}T[0-9:.]+Z\s+','',line)
        if line in ('Prepare workflow directory','Prepare all required actions'):break
        if line.startswith('##[group]'):
            check(active is None,'Nested/unfinished Setup-job image group')
            active=line[len('##[group]'):];values=[]
        elif line=='##[endgroup]':
            if active=='Runner Image':image_groups.append(values)
            if active=='Operating System':os_groups.append(values)
            active=None;values=[]
        elif active is not None:values.append(line)
    check(len(image_groups)==1 and len(os_groups)==1,'Missing/ambiguous authoritative Setup-job Runner Image/OS group')
    fields={}
    for line in image_groups[0]:
        key,separator,value=line.partition(': ')
        check(separator and key not in fields,'Malformed/duplicate Runner Image field')
        fields[key]=value
    check(all(k in fields for k in ('Image','Version','Included Software','Image Release')),'Incomplete authoritative Runner Image header')
    check(re.fullmatch(r'[0-9]{8}\.[0-9]+\.[0-9]+',fields['Version']) is not None,'Unrecognized exact Runner Image version')
    return dict(image=fields['Image'],version=fields['Version'],includedSoftwareURL=fields['Included Software'],imageReleaseURL=fields['Image Release'],operatingSystem=os_groups[0])


class PartReader(io.RawIOBase):
    """Seekable bounded-memory original ZIP ranges; never reconstructs a ZIP file."""
    def __init__(self, root, original, retained):
        self.parts=[];self.position=0;offset=0;whole=hashlib.sha256()
        check(set(original)=={'artifactId','bytes','sha256','parts'},'Original ZIP exact schema')
        check(type(original['artifactId']) is int and original['artifactId']>0,'Original artifact ID')
        check(type(original['bytes']) is int and 0<original['bytes']<=2*1024**3,'Original ZIP size')
        for part in original['parts']:
            check(set(part)=={'path','offset','bytes','sha256','gitBlobSHA1'},'Original ZIP part exact schema')
            safe_zip_name(part['path']);path=root/part['path']
            check(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root.resolve()),'Retained part regular safe path')
            check(type(part['offset']) is int and part['offset']==offset and type(part['bytes']) is int and 0<part['bytes']<=2*1024**2,'Contiguous bounded original ranges')
            b=path.read_bytes();check(len(b)==part['bytes'] and hashlib.sha256(b).hexdigest()==part['sha256']==retained[part['path']],'Retained original part SHA/length')
            check(hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==part['gitBlobSHA1'],'Retained original part Git blob')
            whole.update(b);self.parts.append((offset,path,len(b)));offset+=len(b)
        check(offset==original['bytes'] and whole.hexdigest()==original['sha256'],'Complete original ZIP byte custody')
        self.size=offset
    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.position
    def seek(self,offset,whence=0):
        target=offset if whence==0 else self.position+offset if whence==1 else self.size+offset
        check(0<=target<=self.size,'Original ZIP seek range');self.position=target;return target
    def read(self,n=-1):
        n=self.size-self.position if n<0 else min(n,self.size-self.position);out=[]
        for start,path,size in self.parts:
            if n and start<=self.position<start+size:
                count=min(n,start+size-self.position)
                with path.open('rb') as f:f.seek(self.position-start);b=f.read(count)
                check(len(b)==count,'Original ZIP range changed during read');out.append(b);self.position+=count;n-=count
        return b''.join(out)


def audit_upstream_screen(source_root, stage, provider=None):
    if stage=='screen':return {'pass':True,'scope':'Not applicable to initial registered screen'}
    failures=[];checked=[];store=None;raw=None
    try:
        gate=read_json(source_root/'benchmarks/keel-brace-v1/screen-evidence.json')
        check(set(gate)=={'schemaVersion','sourceFreezeSHA256','screenRunId','receiptTagSHA','decisionFile','decisionSHA256','retainedEvidenceFiles','custodyManifestFile','independentOutcomeReviewCommentId'},'Upstream screen exact gate keys')
        check(type(gate['schemaVersion']) is int and gate['schemaVersion']==1,'Upstream gate schema')
        retained=gate['retainedEvidenceFiles']
        for rel,sha in retained.items():
            safe_zip_name(rel);path=source_root/rel
            check(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(source_root.resolve()) and digest(path)==sha,'Upstream retained file '+rel)
        check(gate['decisionFile'] in retained and gate['custodyManifestFile'] in retained,'Retained decision and custody manifest required')
        m=read_json(source_root/gate['custodyManifestFile'])
        check(set(m)=={'schemaVersion','headSHA','sourceFreezeSHA256','originalZIP','members','decisionMember','freezeMember','gameRecordingMembers','sourceBindings'},'Upstream exact custody manifest keys')
        check(type(m['schemaVersion']) is int and m['schemaVersion']==1 and m['sourceFreezeSHA256']==gate['sourceFreezeSHA256'],'Upstream manifest freeze')
        stream=PartReader(source_root,m['originalZIP'],retained);store=ZipStore(stream)
        check(set(store.files)==set(m['members']),'All original ZIP members retained without omission')
        for name,item in store.files.items():
            row=m['members'][name];check(set(row)=={'bytes','sha256','gitBlobSHA1','crc32','compression','compressedBytes','flags','externalAttr','timestamp','directory'},'Original member exact keys')
            check(row['bytes']==item.file_size and row['crc32']==item.CRC and row['compression']==item.compress_type and row['compressedBytes']==item.compress_size and row['flags']==item.flag_bits and row['externalAttr']==item.external_attr and row['timestamp']==list(item.date_time) and row['directory']==item.is_dir(),'Original member metadata '+name)
            sha=hashlib.sha256();git=hashlib.sha1(b'blob '+str(item.file_size).encode()+b'\0');size=0
            with store.archive.open(item) as f:
                for b in iter(lambda:f.read(1024*1024),b''):sha.update(b);git.update(b);size+=len(b)
            check(size==item.file_size and sha.hexdigest()==row['sha256'] and git.hexdigest()==row['gitBlobSHA1'],'Original complete member bytes '+name)
        suffix='provenance/freeze.json';check(m['freezeMember'].endswith(suffix),'Upstream exact freeze member suffix')
        prefix=m['freezeMember'][:-len(suffix)];check(m['decisionMember']==prefix+'decision.json','Upstream exact decision member')
        evidence=store.root(prefix);freeze=read_json(store.root()/m['freezeMember'])
        check(digest(store.root()/m['freezeMember'])==gate['sourceFreezeSHA256'],'Upstream original frozen bytes')
        check(set(freeze['assets'])==set(m['sourceBindings']),'All upstream source bindings')
        for rel,sha in freeze['assets'].items():
            binding=m['sourceBindings'][rel];check(set(binding)=={'member','commit','path','gitBlobSHA1','sha256'},'Source binding exact schema')
            member=prefix+'provenance/source/'+rel
            check(binding==dict(member=member,commit=m['headSHA'],path=rel,gitBlobSHA1=m['members'][member]['gitBlobSHA1'],sha256=sha) and m['members'][member]['sha256']==sha,'Exact original commit/path/blob/source bytes '+rel)
        decision_bytes=(store.root()/m['decisionMember']).read_bytes();check(decision_bytes==(source_root/gate['decisionFile']).read_bytes() and hashlib.sha256(decision_bytes).hexdigest()==gate['decisionSHA256'],'Unrewritten original raw screen decision')
        decision=json_data(decision_bytes);check(decision['stage']=='screen','Original decision screen stage')
        decision_rows=decision.get('games',decision.get('validGames',[]));original_rows={r.get('id',r.get('game',{}).get('id')):r for r in decision_rows}
        plan=read_json(source_root/'benchmarks/keel-brace-v1/games.json')['screen'];check(set(m['gameRecordingMembers'])=={g['id'] for g in plan} and len(set(m['gameRecordingMembers'].values()))==len(plan),'Exact distinct upstream screen population')
        rows=[]
        for game in plan:
            member=m['gameRecordingMembers'][game['id']];check(member.startswith(prefix+'games/'+game['id']+'/') and member.endswith('.battle.gz'),'Original recording correct game path')
            row=reduce_game(store.root()/member,evidence/'games'/game['id']/'markers.txt',game['participantName'],game['opponentName'],game['opponentVersion'],game['rounds'],game['bot']=='candidate',game['participantVersion'])
            check(row['valid'],'Upstream full raw recording invalid '+game['id']);check(original_rows[game['id']]['recordingSHA256']==row['recordingSHA256'],'Original decision whole recording SHA '+game['id']);row['game']=game;rows.append(row)
        raw=stage_gates('screen',rows);check(raw['allPass'],'Upstream independently replayed full screen does not PASS')
        canonical=reconcile_canonical(evidence,'screen',m['headSHA'],plan,rows,raw,True);check(canonical['pass'],'Upstream canonical acquisition/metrics disagree with full independent raw replay')
        check(provider is not None,'External upstream provider originals required');prior=evidence_path(provider)/'screen'
        run=read_json(prior/'run.json');jobs=read_json(prior/'jobs.json');artifact=read_json(prior/'evidence-artifact.json')
        check(run['id']==gate['screenRunId'] and run['head_sha']==m['headSHA'] and run['run_attempt']==1 and run['status']=='completed' and run['conclusion']=='success','Upstream original completed successful run')
        check(jobs['jobs'] and all(j['status']=='completed' and j['conclusion']=='success' for j in jobs['jobs']),'Upstream completed successful jobs including finalized custody')
        check(artifact['id']==m['originalZIP']['artifactId'] and artifact['size_in_bytes']==m['originalZIP']['bytes'] and artifact['digest']=='sha256:'+m['originalZIP']['sha256'] and artifact['workflow_run']['id']==run['id'] and artifact['workflow_run']['head_sha']==m['headSHA'],'Original upstream finalized provider artifact')
        receipt=read_json(prior/'receipt-tag.json');ref=read_json(prior/'receipt-ref.json')
        check(receipt['sha']==gate['receiptTagSHA']==ref['object']['sha'] and receipt['object']['sha']==m['headSHA'],'Permanent upstream receipt ref/tag/head')
        image=audit_runner_image(evidence,m['headSHA'],run['id'],prior);check(image['pass'],'Upstream authoritative runner-image identity unresolved/conflicting')
        comment=read_json(prior/'outcome-comment.json');check(comment['id']==gate['independentOutcomeReviewCommentId'] and comment['user']['login']=='stephanelee' and comment['user']['id']==30698997,'Actual independently published upstream outcome API')
        prefix_comment='KEEL_BRACE_OUTCOME_REVIEW_V1\n';check(comment['body'].startswith(prefix_comment),'Upstream outcome prefix');review=json_data(comment['body'][len(prefix_comment):])
        check(set(review)=={'verdict','stage','sourceFreezeSHA256','runId','headSHA','decisionSHA256','reviewerTask','agentToolProvenance'},'Upstream exact outcome keys')
        check(review['verdict']=='SCREEN-PASS' and review['stage']=='screen' and review['sourceFreezeSHA256']==gate['sourceFreezeSHA256'] and review['runId']==run['id'] and review['headSHA']==m['headSHA'] and review['decisionSHA256']==gate['decisionSHA256'],'Independent upstream outcome exact identity')
        published=review['agentToolProvenance']['imageIdentityReconciliation'];check(set(published)==IMAGE_FINAL_KEYS,'Upstream exact published mandatory final image reconciliation')
        derived=image['finalReconciliation'];check(derived is not None,'Upstream independently recomputed final image reconciliation')
        for key in IMAGE_FINAL_KEYS-{'runAPI','jobsAPI','jobLog','reviewerTask'}:check(type(published[key]) is type(derived[key]) and published[key]==derived[key],'Upstream exact independent final reconciliation '+key)
        for key,filename in [('runAPI','run.json'),('jobsAPI','jobs.json'),('jobLog','job.log')]:
            ref=published[key];check(set(ref)=={'path','bytes','sha256'},'Upstream exact final provider evidence reference');safe_zip_name(ref['path']);path=source_root/ref['path']
            check(ref['path'] in retained and path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(source_root.resolve()) and type(ref['bytes']) is int and ref['bytes']==path.stat().st_size and digest(path)==ref['sha256']==retained[ref['path']]==digest(prior/filename),'Upstream actual retained full original provider bytes '+key)
        check(published['reviewerTask']==review['reviewerTask'],'Actual separate outcome reviewer owns final reconciliation')
        dispatch=json_data(read_json(evidence/'receipt.json')['payload']['dispatchAssignment']['body'].split('\n',1)[1])
        check(type(review['reviewerTask']) is str and bool(review['reviewerTask']) and type(review['agentToolProvenance']) is dict and bool(review['agentToolProvenance']) and review['agentToolProvenance']['reviewerTask']==review['reviewerTask'] and review['reviewerTask'] not in dispatch['implementationTasks'] and comment['updated_at']==comment['created_at'] and timestamp(comment['created_at'])>max(timestamp(j['completed_at']) for j in jobs['jobs']),'Unedited strictly post-completion separate reviewer/tool task provenance recorded')
        checked.append('All original upstream ZIP ranges/members/source bindings, raw screen gates, provider run/receipt/image and published independent outcome matched')
    except (KeyError,TypeError,ValueError,OSError,zipfile.BadZipFile,EOFError,zlib.error) as exc:failures.append(type(exc).__name__+': '+str(exc))
    finally:
        if store:store.close()
    return dict(checked=checked,failures=failures,rawGates=raw,**{'pass':not failures},scope='Independent full original screen byte custody and raw score/timing/exposure replay. Published separate review supplies upstream source-policy/runtime approval judgment; current confirmation independently needs its own complete authority. Collector must corroborate actual reviewer task provenance; no self-certified runtime clearance.')


IMAGE_SNAPSHOT_KEYS={'schemaVersion','policyID','policySHA256','stage','phase','runId','runAttempt','headSHA','workspace','bootId','architecture','kernel','kernelVersion','logicalJobName','collectorSourceSHA256','snapshotManifestSHA256','reportSHA256','stable'}
IMAGE_STABLE_KEYS={'rawSources','claims','runnerRoot','setupInfoPath','bootId','architecture','kernel','kernelVersion','collectorSourceSHA256','sourceIdentity'}
IMAGE_SOURCE_KEYS={'stage','runId','runAttempt','headSHA','workspace','logicalJobName'}
IMAGE_WRAPPER_KEYS={'schemaVersion','policyID','policySHA256','runId','runAttempt','headSHA','logicalJobName','preparedNumericJobId','beforeToolsSnapshotFile','beforeToolsSnapshotSHA256','preparedSnapshotFile','preparedSnapshotSHA256','jobsAPIFile','jobsAPISHA256'}
IMAGE_REVIEW_KEYS={'schemaVersion','policyID','policySHA256','beforeToolsSnapshotSHA256','preparedSnapshotSHA256','preparedWrapperSHA256','primaryPregameAuthority','earlyHeaderEvidence','finalExternalReconciliationRequired'}
IMAGE_EARLY_KEYS={'status','endpoint','runId','runAttempt','jobId','headSHA','observedAtUTC','httpStatus','responseBytes','responseSHA256','responseFile','scope','runAccessVerified','jobsAccessVerified','artifactAccessVerified','jobInProgress','nativeClaim'}
IMAGE_FINAL_KEYS={'schemaVersion','status','policyID','policySHA256','runId','runAttempt','jobId','headSHA','beforeToolsSnapshotSHA256','preparedSnapshotSHA256','preparedWrapperSHA256','nativeClaim','runAPI','jobsAPI','jobLog','completedAtUTC','reviewerTask'}
IMAGE_LABEL_PATHS={'imagedata':'/imagegeneration/imagedata.json','etc_environment':'/etc/environment','etc_os_release':'/etc/os-release','usr_lib_os_release':'/usr/lib/os-release','etc_lsb_release':'/etc/lsb-release','boot_id':'/proc/sys/kernel/random/boot_id'}


def strict_int(value,label,minimum=0,maximum=None):
    check(type(value) is int and value>=minimum and (maximum is None or value<=maximum),'Strict bounded integer '+label)
    return value


def image_policy_binding(source_root,freeze):
    for rel,wanted in [('docs/improvements/keel-brace-runtime-identity-amendment.md',AMENDMENT_DOCUMENT_SHA256),('benchmarks/keel-brace-v1/runtime-identity-amendment.json',AMENDMENT_POLICY_SHA256)]:
        check(digest(source_root/rel)==wanted and freeze['assets'][rel]==wanted,'Exact prospectively published future image policy '+rel)
    check('tools/keel_brace_image_probe.py' in freeze['assets'],'Actual reviewed collector included in current source freeze')


def audit_image_snapshot(root,expected,collector_phase,freeze):
    """Independently reconstruct stable claims from complete captured raw bytes."""
    snapshot=read_json(root/'snapshot.json');check(set(snapshot)==IMAGE_SNAPSHOT_KEYS,'Image snapshot exact keys')
    strict_int(snapshot['schemaVersion'],'snapshot schema',1,1)
    check(snapshot['policyID']==QUALIFIED_POLICY_ID and snapshot['policySHA256']==AMENDMENT_POLICY_SHA256,'Snapshot exact qualified policy')
    for k,v in expected.items():check(type(snapshot[k]) is type(v) and snapshot[k]==v,'Snapshot actual identity '+k)
    for k in ('workspace','bootId','architecture','kernel','kernelVersion','logicalJobName'):check(type(snapshot[k]) is str and bool(snapshot[k]),'Snapshot nonempty observed '+k)
    check(snapshot['architecture']=='x86_64','Snapshot qualified architecture')
    stable=snapshot['stable'];check(set(stable)==IMAGE_STABLE_KEYS,'Snapshot stable exact keys')
    collector=root/'collector'/collector_phase;status=read_json(collector/'capture-status.json');report=read_json(collector/'report.json')
    check(snapshot['snapshotManifestSHA256']==digest(collector/'capture-status.json') and snapshot['reportSHA256']==digest(collector/'report.json'),'Immutable whole snapshot/report byte binding')
    for path in (root/'snapshot.json',collector/'capture-status.json'):check(path.stat().st_size<=1048576,'Snapshot/status metadata cap')
    check(status['schemaVersion']==1 and type(status['schemaVersion']) is int and report['schemaVersion']==1 and type(report['schemaVersion']) is int and report['phase']==collector_phase and status['phase']==collector_phase,'Original collector phase/schema')
    check(report['errors']==[],'No global collector errors');strict_int(status['diagnosticErrorCount'],'diagnostic error count',0,1)
    check(report['collectorPython']['executable']=='/usr/bin/python3','Frozen collector stock Python interpreter')
    strict_int(report['durationNs'],'capture duration',0,120*10**9)
    check(image_timestamp(report['utcStart'])<=image_timestamp(report['utcEnd']) and image_timestamp(report['utcEnd'])-image_timestamp(report['utcStart'])<=datetime.timedelta(seconds=120),'Bounded actual capture UTC')
    check(report['limits']['eachFileBytes']==1048576 and report['limits']['wholeTwoPhaseArtifactBytes']==10485760 and report['limits']['procEntries']==512 and report['limits']['runnerMatches']==16 and report['limits']['collectionSeconds']==120,'Registered raw collector bounds')
    payload=0
    expected_files={'collector.py','report.json'}|{'raw/'+k+'.bin' for k,v in report['captures'].items() if v.get('exists') is True}|{'parsed/'+k+'.json' for k in ('imagedata','setup_info_0','etc_environment')}
    check(set(status['files'])==expected_files,'Exact source-backed capture/parsed/code/report file inventory')
    for rel,row in status['files'].items():
        safe_zip_name(rel);path=collector/rel
        check(row['artifactPath']==rel and row['complete'] is True,'Whole original capture file '+rel)
        strict_int(row['bytes'],'capture file bytes',0,1048576);strict_int(row['suppliedBytes'],'supplied complete bytes',0,1048576);check(row['bytes']==row['suppliedBytes']==path.stat().st_size and digest(path)==row['sha256'],'Original complete capture bytehash/size '+rel)
        if rel.endswith('.json'):read_json(path)
        payload+=row['bytes']
    check(payload==status['retainedBytesBeforeStatus'] and payload+65536<=5210112,'Per-snapshot payload/archive reserve')
    check(status['report']==status['files']['report.json'] and report['collectorSource']==status['files']['collector.py'],'Original report/source capture manifest bindings')
    check(report['rawArtifactIndex']=={k:v for k,v in status['files'].items() if k!='report.json'},'Complete report raw/source/parsed artifact inventory')
    collector_sha=digest(collector/'collector.py');check(collector_sha==snapshot['collectorSourceSHA256']==freeze['assets']['tools/keel_brace_image_probe.py'],'Actual current frozen executed collector bytes, no probe-code hash substitution')
    discovery=report['runnerDiscovery'];check(discovery['procRoot']=='/proc' and discovery['errors']==[] and discovery['truncated'] is False,'Error-free bounded real process discovery')
    strict_int(discovery['scannedEntries'],'scanned proc entries',1,512);strict_int(discovery['nonmatchingCount'],'nonmatching proc entries',0,512)
    check(0<len(discovery['matches'])<16 and discovery['scannedEntries']==discovery['nonmatchingCount']+len(discovery['matches']),'Complete nontruncated process accounting')
    roots=set();pids=set()
    for match in discovery['matches']:
        pid=strict_int(match['pid'],'observed runner PID',1);check(pid not in pids,'Distinct runner PID');pids.add(pid)
        comm=match['comm'];check(comm in ('Runner.Worker','Runner.Listener') and match['commRawASCII']==comm+'\n' and hashlib.sha256(match['commRawASCII'].encode('ascii')).hexdigest()==match['commSHA256'],'Whole observed runner comm bytes')
        check(match['commPath']=='/proc/'+str(pid)+'/comm' and match['errors']==[],'Observed PID/comm source binding')
        exe=match['exeReadlink'];check(exe==match['exePath'] and hashlib.sha256(exe.encode()).hexdigest()==match['exeReadlinkSHA256'],'Observed executable link bytes')
        check(exe.endswith('/bin/'+comm) and match['runnerRoot']==exe[:-len('/bin/'+comm)] and match['setupInfoPath']==match['runnerRoot']+'/.setup_info','Runner root derived exclusively from observed matching executable layout')
        roots.add(match['runnerRoot'])
    check(len(roots)==1 and discovery['roots']==sorted(roots),'Unique actually observed runner root')
    runner_root=next(iter(roots));paths=dict(IMAGE_LABEL_PATHS,setup_info_0=runner_root+'/.setup_info')
    check(set(report['captures'])==set(paths) and set(stable['rawSources'])==set(paths),'Exact allowed raw capture population')
    raw_sources={};raw_bytes={};allowed_absence_count=0
    for label,path in paths.items():
        record=report['captures'][label];check(record['requestedPath']==path,'Exact observed raw source path '+label)
        if label=='etc_lsb_release' and record['exists'] is False:
            errors=record['errors'];check(len(errors)==1 and set(errors[0])=={'operation','path','type','message','messageWasTruncated'} and errors[0]['operation']=='lstat' and errors[0]['path']=='/etc/lsb-release' and errors[0]['type']=='FileNotFoundError' and type(errors[0]['message']) is str and bool(errors[0]['message']) and errors[0]['messageWasTruncated'] is False and record['raw'] is None and record['bytesComplete'] is False and record['resolvedPath'] is None,'Only exact optional lsb lstat-not-found absence allowed')
            allowed_absence_count+=1;raw_sources[label]=dict(sourcePath=path,absent=True);check(stable['rawSources'][label]==raw_sources[label],'Optional absence exact stable record');continue
        check(record['exists'] is True and record['errors']==[] and record['bytesComplete'] is True and record.get('parseErrors',[])==[] and record['rawSHA256Scope']=='whole-observed-file','Complete error-free untruncated raw source '+label)
        raw=record['raw'];check(raw==status['files'][raw['artifactPath']] and raw['sourceBytesComplete'] is True,'Raw source complete capture manifest '+label)
        check(raw['artifactPath']=='raw/'+label+'.bin','Fixed source-backed raw filename')
        for field in ('size','mode','uid','gid','device','inode','mtimeNs'):
            value=record['statBefore'][field];strict_int(value,'raw stat '+field)
            strict_int(record['openedStat'][field],'opened raw stat '+field);strict_int(record['statAfter'][field],'after raw stat '+field)
            check(record['openedStat'][field]==value==record['statAfter'][field],'No source mutation during individual read '+label+':'+field)
        check(record['openedStat']['regular'] is True,'Opened metadata source regular file')
        strict_int(record['durationNs'],'individual read duration',0,120*10**9)
        check(image_timestamp(report['utcStart'])<=image_timestamp(record['utcStart'])<=image_timestamp(record['utcEnd'])<=image_timestamp(report['utcEnd']),'Actual raw read bounded inside snapshot')
        raw_bytes[label]=(collector/raw['artifactPath']).read_bytes()
        check(label=='boot_id' or record['statBefore']['size']==len(raw_bytes[label]),'Raw whole source size '+label)
        raw_sources[label]=dict(sourcePath=path,resolvedPath=record['resolvedPath'],size=record['statBefore']['size'],wholeSHA256=raw['sha256'],mode=record['statBefore']['mode'],uid=record['statBefore']['uid'],gid=record['statBefore']['gid'])
        check(type(record['resolvedPath']) is str and record['resolvedPath'].startswith('/'),'Observed resolved source path')
        for key in ('size','mode','uid','gid'):strict_int(stable['rawSources'][label][key],'stable raw '+key)
        check(stable['rawSources'][label]==raw_sources[label] and set(stable['rawSources'][label])=={'sourcePath','resolvedPath','size','wholeSHA256','mode','uid','gid'},'Exact reconstructed stable raw source '+label)
    check(status['diagnosticErrorCount']==allowed_absence_count,'Only exact known optional absence contributes diagnostic count')
    check(raw_bytes['boot_id'].decode('ascii').strip()==snapshot['bootId'],'Actual raw boot ID binding')
    for label in ('etc_os_release','usr_lib_os_release'):
        text=raw_bytes[label].decode();check('ID=ubuntu' in text.splitlines() and 'VERSION_ID="26.04"' in text.splitlines(),'Both actual raw Ubuntu26.04 release files')
    process=report['processImageEnvironment'];check(set(process)==set(QUALIFIED_ENV) and all(set(process[k])=={'present','value'} and process[k]['present'] is True and process[k]['value']==v for k,v in QUALIFIED_ENV.items()),'Separate observed process ImageOS/ImageVersion')
    claims=dict(imagedataNative=strict_native_claim(raw_bytes['imagedata'],'imagedata'),setupInfoNative=strict_native_claim(raw_bytes['setup_info_0'],'setup_info'),processEnvironment=dict(QUALIFIED_ENV),etcEnvironment=strict_environment_claim(raw_bytes['etc_environment']))
    for label,key in [('imagedata','imagedataNative'),('setup_info_0','setupInfoNative')]:
        record=report['captures'][label];check(record['parsedClaims']==claims[key] and record['parsedArtifact']==status['files']['parsed/'+label+'.json'],'Independently reparsed native claims and complete parsed bytehash record')
    check(stable['claims']==claims and stable['runnerRoot']==runner_root and stable['setupInfoPath']==runner_root+'/.setup_info','Reconstructed separate claims/root equal immutable stable snapshot')
    source={k:snapshot[k] for k in IMAGE_SOURCE_KEYS};check(stable['sourceIdentity']==source and set(stable['sourceIdentity'])==IMAGE_SOURCE_KEYS and all(type(stable['sourceIdentity'][k]) is type(v) for k,v in source.items()),'Exact stable actual source/run/head/workspace/job')
    for key in ('bootId','architecture','kernel','kernelVersion','collectorSourceSHA256'):check(stable[key]==snapshot[key],'Stable snapshot observation '+key)
    check(report['platform']['architecture']==snapshot['architecture'] and report['platform']['kernel']==snapshot['kernel'] and report['platform']['kernelVersion']==snapshot['kernelVersion'],'Observed platform/kernel binding')
    context=report['context'];check(context['GITHUB_RUN_ID']==str(snapshot['runId']) and context['GITHUB_RUN_ATTEMPT']==str(snapshot['runAttempt']) and context['GITHUB_JOB']==snapshot['logicalJobName'] and context['GITHUB_REPOSITORY']=='stephanelee/robocode','Original logical job/run/attempt context; no fabricated early numeric job ID')
    envelope=(root/'snapshot.json').stat().st_size+(collector/'capture-status.json').stat().st_size
    check(payload+envelope+65536<=5210112,'Complete bounded snapshot payload including immutable manifests/reserve')
    return dict(snapshot=snapshot,report=report,payloadBytes=payload,envelopeBytes=envelope,root=root)


def image_timestamp(value):
    result=timestamp(value);check(result.utcoffset()==datetime.timedelta(0),'Qualified image evidence timestamp must be actual UTC')
    return result


def provider_step_contains(step,start,end):
    # GitHub step timestamps record whole seconds: completed T denotes [T,T+1s).
    return image_timestamp(step['started_at'])<=image_timestamp(start)<=image_timestamp(end)<image_timestamp(step['completed_at'])+datetime.timedelta(seconds=1)


def actual_file_ref(path):
    return dict(path=str(path),bytes=path.stat().st_size,sha256=digest(path))


def audit_runner_image(evidence, expected_head, expected_run, provider=None):
    """FUTURE Brace qualified observations; never a legacy fallback/VM attestation."""
    checked=[];failures=[];final=None;status='PROVENANCE-PENDING';qualified=None
    try:
        profile_path=evidence/'profile.json'
        if not profile_path.exists():profile_path=evidence/'provenance/profile.json'
        provenance=profile_path.parent;profile=read_json(profile_path);freeze=read_json(provenance/'freeze.json');image=provenance/'image-identity'
        image_policy_binding(provenance/'source',freeze)
        approval=read_json(evidence/'approval.json');stage=approval['stage'];identity=profile['identity']
        expected=dict(stage=stage,runId=expected_run,runAttempt=1,headSHA=expected_head,workspace=identity['workspace'])
        before=audit_image_snapshot(image/'before-tools',dict(expected,phase='before-tools'),'before-tools',freeze)
        prepared=audit_image_snapshot(image/'prepared',dict(expected,phase='prepared'),'after-tools',freeze)
        bs,ps=before['snapshot'],prepared['snapshot'];qualified=ps['stable']['claims']
        check(bs['stable']==ps['stable'],'Before-tools and prepared whole raw stable/image/boot/source/collector equality; installed graph first frozen only at prepared')
        check(before['payloadBytes']+prepared['payloadBytes']+before['envelopeBytes']+prepared['envelopeBytes']+(image/'prepared-wrapper.json').stat().st_size+(image/'prepared-jobs-api.json').stat().st_size+65536<=10485760,'Two immutable snapshot payloads plus archive reserve cap')
        check(bs['bootId']==ps['bootId']==identity['bootId'] and identity['image']==QUALIFIED_ENV['ImageVersion'],'Original profile anchors same-worker qualified native/separate ENV observations')
        check((image/'prepared-wrapper.json').stat().st_size<=1048576 and (image/'prepared-jobs-api.json').stat().st_size<=1048576,'Wrapper/jobs API metadata caps')
        wrapper=read_json(image/'prepared-wrapper.json');check(set(wrapper)==IMAGE_WRAPPER_KEYS,'Prepared wrapper exact schema')
        strict_int(wrapper['schemaVersion'],'prepared wrapper schema',1,1);job_id=strict_int(wrapper['preparedNumericJobId'],'actual numeric prepared job ID',1)
        for k in ('runId','runAttempt','headSHA','logicalJobName'):check(type(wrapper[k]) is type(ps[k]) and wrapper[k]==ps[k],'Wrapper exact actual '+k)
        check(wrapper['policyID']==QUALIFIED_POLICY_ID and wrapper['policySHA256']==AMENDMENT_POLICY_SHA256,'Wrapper exact policy')
        check(wrapper['beforeToolsSnapshotFile']=='before-tools/snapshot.json' and wrapper['preparedSnapshotFile']=='prepared/snapshot.json' and wrapper['jobsAPIFile']=='prepared-jobs-api.json','Fixed immutable wrapper reference paths')
        check(wrapper['beforeToolsSnapshotSHA256']==digest(image/'before-tools/snapshot.json') and wrapper['preparedSnapshotSHA256']==digest(image/'prepared/snapshot.json') and wrapper['jobsAPISHA256']==digest(image/'prepared-jobs-api.json'),'Actual immutable snapshot/API bytehashes')
        prepared_jobs=read_json(image/'prepared-jobs-api.json');matches=[j for j in prepared_jobs['jobs'] if j['id']==job_id]
        check(len(matches)==1,'Exactly one original prepared numeric job');strict_int(matches[0]['id'],'prepared API job ID',1);strict_int(matches[0]['run_id'],'prepared API run ID',1)
        check(matches[0]['run_id']==expected_run and matches[0]['status']=='in_progress','Prepared numeric job comes from actual same-run in-progress jobs API')
        hashes=dict(beforeToolsSnapshotSHA256=wrapper['beforeToolsSnapshotSHA256'],preparedSnapshotSHA256=wrapper['preparedSnapshotSHA256'],preparedWrapperSHA256=digest(image/'prepared-wrapper.json'))
        anchored=profile['imageIdentity'];check(set(anchored)=={'policyID','policySHA256',*hashes,'preparedNumericJobId','stable'},'Profile exact imageIdentity schema')
        check(anchored==dict(policyID=QUALIFIED_POLICY_ID,policySHA256=AMENDMENT_POLICY_SHA256,**hashes,preparedNumericJobId=job_id,stable=ps['stable']),'Immutable prepared profile exact policy/snapshot/wrapper/stable anchor')
        report_comment=read_json(evidence/'review-report-comment.json');report=comment_payload(report_comment,'KEEL_BRACE_RUNTIME_REVIEW_V1');review=report['agentToolProvenance']['imageIdentityReview']
        check(set(review)==IMAGE_REVIEW_KEYS and type(review['schemaVersion']) is int and review['schemaVersion']==1,'Accepted runtime review nested exact schema')
        check(review['policyID']==QUALIFIED_POLICY_ID and review['policySHA256']==AMENDMENT_POLICY_SHA256 and all(review[k]==v for k,v in hashes.items()),'Actual immutable runtime report binds qualified snapshots/policy')
        check(review['primaryPregameAuthority']=='native-only-qualified-observation' and review['finalExternalReconciliationRequired'] is True,'Explicit native-only pregame authority with mandatory independent final reconciliation')
        early=review['earlyHeaderEvidence'];check(set(early)==IMAGE_EARLY_KEYS,'Exact early actual-job header record')
        for key,value in [('runId',expected_run),('runAttempt',1),('jobId',job_id),('headSHA',expected_head)]:check(type(early[key]) is type(value) and early[key]==value,'Actual early job identity '+key)
        check(early['endpoint']=='https://api.github.com/repos/stephanelee/robocode/actions/jobs/'+str(job_id)+'/logs' and early['responseFile']=='early-header-response.bin','Exact original actual-job log endpoint/response path')
        check(all(early[k] is True for k in ('runAccessVerified','jobsAccessVerified','artifactAccessVerified')),'Early access declarations; independently corroborated below')
        check(provider is not None,'External early originals/final provider reconciliation not supplied')
        provider=evidence_path(provider);earlyroot=provider/'early';earlyrun=read_json(earlyroot/'run.json');earlyjobs=read_json(earlyroot/'jobs.json');earlyartifact=read_json(earlyroot/'artifact.json');response=earlyroot/'early-header-response.bin'
        check(type(early['responseBytes']) is int and 0<=early['responseBytes']<=1048576 and response.stat().st_size==early['responseBytes'] and digest(response)==early['responseSHA256'],'Actual original early HTTP response bytes/length/hash')
        strict_int(earlyrun['id'],'actual early run ID',1);strict_int(earlyrun['run_attempt'],'actual early run attempt',1,1)
        check(earlyrun['id']==expected_run and earlyrun['head_sha']==expected_head and earlyrun['run_attempt']==1 and earlyrun['status']=='in_progress','Actual independently retained early same-run/head/attempt in-progress access')
        earlyjob=[j for j in earlyjobs['jobs'] if j['id']==job_id];check(len(earlyjob)==1,'Exactly one original early numeric job');strict_int(earlyjob[0]['id'],'early API job ID',1);strict_int(earlyjob[0]['run_id'],'early API run ID',1);check(len(earlyjob)==1 and earlyjob[0]['run_id']==expected_run,'Actual early jobs API binds original numeric job')
        artifact=read_json(evidence/'artifact.json');check(earlyartifact['id']==artifact['id'] and earlyartifact['digest']==artifact['digest'] and earlyartifact['workflow_run']['id']==expected_run and earlyartifact['workflow_run']['head_sha']==expected_head,'Actual independently retained original finalized artifact access')
        observed=image_timestamp(early['observedAtUTC']);check(image_timestamp(artifact['updated_at'])<=observed<=image_timestamp(report_comment['created_at']),'Early HTTP attempt after actual finalized artifact and before immutable independent report')
        check(type(early['httpStatus']) is int,'Original HTTP status strict integer')
        if early['status']=='unavailable-404-deferred':
            check(early['httpStatus']==404 and early['scope']=='actual-job-log-request' and early['jobInProgress'] is True and earlyjob[0]['status']=='in_progress' and early['nativeClaim'] is None,'Sole explicitly registered correctly bound HTTP404 in-progress deferral')
        else:
            check(early['status']=='available-matched' and early['httpStatus']==200 and early['scope']=='complete-startup-header' and early['nativeClaim']==QUALIFIED_NATIVE and qualified_provider_header(response.read_bytes())==QUALIFIED_NATIVE,'Available original complete startup header must exactly match native tuple')
        phases=['before-receipt']+['before-'+g['id'] for g in read_json(provenance/'source/benchmarks/keel-brace-v1/games.json')[stage]]
        revalidation_files=[]
        for phase in phases:
            authority=read_json(evidence/'approval-revalidations'/(phase+'.json'));binding=authority['imageIdentityRevalidation']
            check(set(binding)=={'snapshotFile','snapshotSHA256','preparedSnapshotSHA256','runtimeProfileSHA256'},'Exact per-authority image/prepared-graph revalidation binding')
            rel=binding['snapshotFile'];safe_zip_name(rel);check(re.fullmatch(r'image-identity-revalidations/revalidation-[0-9]+/snapshot\.json',rel) is not None,'Fresh image revalidation source path')
            path=evidence/rel;check(binding['snapshotSHA256']==digest(path) and binding['preparedSnapshotSHA256']==hashes['preparedSnapshotSHA256'] and binding['runtimeProfileSHA256']==digest(profile_path),'Each receipt/child binds fresh image snapshot and original prepared installed graph/profile')
            current=audit_image_snapshot(path.parent,expected,'after-tools',freeze)
            check(current['snapshot']['phase']==rel.split('/')[1] and current['snapshot']['stable']==ps['stable'],'Fresh actual stable image/source/boot equals original prepared anchor; no overwrite')
            check(image_timestamp(prepared['report']['utcEnd'])<=image_timestamp(current['report']['utcStart'])<=image_timestamp(current['report']['utcEnd'])<=image_timestamp(authority['checkedAtUTC']),'Image/installed-graph live check before authority admission')
            check(rel not in revalidation_files,'Distinct immutable per-admission snapshots');revalidation_files.append(rel)
        # Final provider records are mandatory even when early access succeeded.
        run=read_json(provider/'run.json');jobs=read_json(provider/'jobs.json')
        if run['status']=='completed':status='PROVENANCE-INCOMPLETE'
        strict_int(run['id'],'actual final run ID',1);strict_int(run['run_attempt'],'actual final run attempt',1,1)
        check(run['id']==expected_run and run['head_sha']==expected_head and run['run_attempt']==1 and run['status']=='completed' and run['conclusion']=='success','Actual completed successful same-run/head/attempt provider API')
        selected=[j for j in jobs['jobs'] if j['id']==job_id];check(len(selected)==1,'Actual completed original numeric job')
        job=selected[0];strict_int(job['id'],'final API job ID',1);strict_int(job['run_id'],'final API run ID',1);check(job['run_id']==expected_run and job['status']=='completed' and job['conclusion']=='success','Actual successful same-run completed job')
        steps=job['steps'];names=['Capture original qualified image observations before tools','Prepare exact sources/runtime/commands on this worker (zero games)','Finalize immutable zero-game provenance before independent review'];ordered=[]
        for name in names:
            found=[(i,x) for i,x in enumerate(steps) if x['name']==name];check(len(found)==1 and found[0][1]['status']=='completed' and found[0][1]['conclusion']=='success','Exact provider completed capture/prepare/finalize step '+name);ordered.append(found[0])
        check(ordered[0][0]<ordered[1][0]<ordered[2][0],'Actual provider before-tools/prepared/finalize UTC step ordering')
        for capture,(_,step) in zip((before,prepared),ordered):check(provider_step_contains(step,capture['report']['utcStart'],capture['report']['utcEnd']),'Actual capture UTC inside exact provider recorded-second interval [T,T+1s)')
        check(all(not re.search(r'java|python|temurin|jdk|cpython',step['name'],re.I) for step in steps[:ordered[0][0]]),'Native original capture precedes any registered Java/Python setup')
        check(image_timestamp(prepared['report']['utcEnd'])<=image_timestamp(artifact['updated_at'])<=image_timestamp(report_comment['created_at'])<image_timestamp(job['completed_at']),'Prepared/provenance/review/actual completion chronology')
        meta=read_json(provider/'job-log-api.json');log=provider/'job.log'
        check(set(meta)=={'schemaVersion','jobId','runId','headSHA','logSHA256','sourceURL','retrievedAtUTC','collectorTask','toolEvidence'},'Original independent full-provider-log custody record')
        check(type(meta['schemaVersion']) is int and meta['schemaVersion']==1 and type(meta['jobId']) is int and meta['jobId']==job_id and meta['runId']==expected_run and meta['headSHA']==expected_head,'Actual final log exact original job/run/head')
        check(meta['sourceURL']==early['endpoint'] and meta['logSHA256']==digest(log) and bool(meta['collectorTask']) and bool(meta['toolEvidence']) and image_timestamp(meta['retrievedAtUTC'])>=image_timestamp(job['completed_at']),'Full original provider log bytehash/source/task/post-completion custody')
        native=qualified_provider_header(log.read_bytes());check(native==ps['stable']['claims']['setupInfoNative']==ps['stable']['claims']['imagedataNative'],'Mandatory independent actual-job final native four-field reconciliation')
        final=dict(schemaVersion=1,status='PASS',policyID=QUALIFIED_POLICY_ID,policySHA256=AMENDMENT_POLICY_SHA256,runId=expected_run,runAttempt=1,jobId=job_id,headSHA=expected_head,**hashes,nativeClaim=native,runAPI=actual_file_ref(provider/'run.json'),jobsAPI=actual_file_ref(provider/'jobs.json'),jobLog=actual_file_ref(log),completedAtUTC=job['completed_at'],reviewerTask='/root/brace_independent_outcomes')
        status='PASS';checked.append('Qualified observed native/ENV tuple, immutable before-tools/prepared anchors, original early actual-job evidence, every explicit live image/prepared-graph binding and mandatory independent final same-job provider reconciliation')
    except (KeyError,TypeError,ValueError,OSError,StopIteration,UnicodeError) as exc:
        failures.append(type(exc).__name__+': '+str(exc))
        if provider is not None:
            try:
                if read_json(evidence_path(provider)/'run.json').get('status')=='completed':status='PROVENANCE-INCOMPLETE'
            except (KeyError,TypeError,ValueError,OSError):status='PROVENANCE-INCOMPLETE'
    return dict(checked=checked,failures=failures,status=status,qualifiedClaims=qualified,finalReconciliation=final,**{'pass':not failures},scope='Prospectively registered FUTURE Brace qualified observations only. Native20261025.179.1 and separate ENV20261026.180.1 remain distinct; this is NOT canonical VM/build attestation, a version translation, old Helm rescue or gameplay clearance. Unexported actual executable bytes and source-wired prepared-graph equality still require independent live runtime/source review. External original HTTP/API/log bytes are retained separately, never retro-added to immutable stage provenance.')


def audit_legacy_runner_image(evidence, expected_head, expected_run, provider=None):
    checked,failures=[],[];header=None;declared=None
    def require(ok,label):(checked if ok else failures).append(label)
    try:
        profile_path=evidence/'profile.json'
        if not profile_path.exists():profile_path=evidence/'provenance'/'profile.json'
        declared=read_json(profile_path)['identity']['image']
        if provider is None:failures.append('Independent authoritative runner job log/API-origin metadata not supplied; ENV alone is insufficient')
        else:
            provider=evidence_path(provider);meta=read_json(provider/'job-log-api.json');jobs=read_json(provider/'jobs.json');run=read_json(provider/'run.json');log=provider/'job.log'
            keys={'schemaVersion','jobId','runId','headSHA','logSHA256','sourceURL','retrievedAtUTC','collectorTask','toolEvidence'}
            require(set(meta)==keys and type(meta['schemaVersion']) is int and meta['schemaVersion']==1 and type(meta['jobId']) is int and type(meta['runId']) is int,'exact typed job-log collector record')
            require(meta['runId']==expected_run and meta['headSHA']==expected_head and any(j['id']==meta['jobId'] for j in jobs['jobs']),'raw job log exact independently collected job/run/head identity')
            require(meta['logSHA256']==digest(log),'complete original decoded job-log byte digest')
            require(type(meta['sourceURL']) is str and meta['sourceURL'].startswith('https://api.github.com/repos/stephanelee/robocode/actions/') and '?' not in meta['sourceURL'] and bool(meta['collectorTask']) and bool(meta['toolEvidence']),'permanent provider API source and actual collector/tool provenance recorded')
            require(timestamp(meta['retrievedAtUTC'])>=timestamp(run['created_at']),'job-log collection chronology')
            with log.open('r') as lines:header=runner_image_header(lines)
            require(header['image']=='ubuntu-26.04' and len(header['operatingSystem'])>=2 and header['operatingSystem'][0]=='Ubuntu' and re.fullmatch(r'26\.04(?:\.[0-9]+)?',header['operatingSystem'][1]) is not None,'independent authoritative Ubuntu26.04 image/OS header')
            require(header['version'] in ('20261026.180.1','20261101.186.1'),'authoritative Runner Image version is prospectively allowed')
            require(declared==header['version'],'NO conflict between ENV/profile and independent authoritative Runner Image header')
    except (KeyError,TypeError,ValueError,OSError) as exc:failures.append(type(exc).__name__+': '+str(exc))
    return dict(checked=checked,failures=failures,profileDeclaredImage=declared,authoritativeHeader=header,**{'pass':not failures},scope='Post-outcome independently collected original Setup-job header versus archived ENV/profile and unchanged allowlist. ENV alone or majority voting is never authority. This does not supply pre-execution image-resolution or runtime clearance; root must resolve any additional authoritative contradictions prospectively before dispatch.')


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('audit-stage');a.add_argument('--evidence',type=Path);a.add_argument('--evidence-zip',type=Path);a.add_argument('--prefix',default='');a.add_argument('--zip-sha256');a.add_argument('--custody-zip',type=Path);a.add_argument('--custody-sha256');a.add_argument('--source-root',type=Path,required=True);a.add_argument('--provider-evidence',type=Path);a.add_argument('--stage',choices=['screen','confirmation'],required=True);a.add_argument('--head',required=True);a.add_argument('--run',type=int,required=True);a.add_argument('--output',type=Path,required=True)
    z=sub.add_parser('verify-archive');z.add_argument('--zip',type=Path,required=True);z.add_argument('--sha256',required=True);z.add_argument('--extracted-root',type=Path);z.add_argument('--output',type=Path,required=True)
    args=p.parse_args();store=None;companion_store=None
    try:
        if args.command=='verify-archive':result=audit_archive(args.zip,args.sha256,args.extracted_root)
        else:
            check(bool(args.evidence)!=bool(args.evidence_zip),'Exactly one evidence directory or original ZIP required')
            if args.evidence_zip:
                check(bool(args.zip_sha256),'Original provider ZIP digest required');custody=audit_archive(args.evidence_zip,args.zip_sha256);store=ZipStore(args.evidence_zip);evidence=store.root(args.prefix)
            else:evidence=args.evidence;custody=None
            transport=None;companion_proof=None
            if args.custody_zip:
                check(bool(args.custody_sha256),'Companion provider ZIP digest required');check(custody is not None,'Original ZIP proof required with companion')
                companion_proof=audit_archive(args.custody_zip,args.custody_sha256,original_size_limit=False);companion_store=ZipStore(args.custody_zip)
                candidates=[n for n in companion_store.files if n.rsplit('/',1)[-1]=='evidence-artifact.json' and (n.rsplit('/',1)[0]+'/' if '/' in n else '')+'export-validation.json' in companion_store.files]
                check(len(candidates)==1,'Exactly one companion original-artifact/export-validation pair required')
                root=candidates[0].rsplit('/',1)[0] if '/' in candidates[0] else ''
                transport=dict(originalZIPBytes=custody['originalZIPBytes'],originalZIPSHA256=custody['originalZIPSHA256'],custodyZIPBytes=companion_proof['originalZIPBytes'],custodyZIPSHA256=companion_proof['originalZIPSHA256'],custodyRoot=companion_store.root(root))
            result=audit_stage(evidence,args.source_root,args.stage,args.head,args.run,args.provider_evidence,transport)
            if companion_proof:result['companionArchiveAudit']=companion_proof
            if custody:result['originalArchiveAudit']=custody
        result['auditorSourceSHA256']=digest(Path(__file__));args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(json.dumps({k:result[k] for k in ('scope','pass','decision','rawDisposition') if k in result}));return 0 if result.get('pass',result.get('decision')=='PASS') else 1
    except (ValueError,OSError,zipfile.BadZipFile,KeyError,TypeError,EOFError,zlib.error) as exc:
        result={'decision':'AUDIT-CUSTODY-INCOMPLETE','error':type(exc).__name__+': '+str(exc),'limit':'Local custody/parser failure is not evidence of a new battle result and never opens replacement rounds','auditorSourceSHA256':digest(Path(__file__))}
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result));return 1
    finally:
        if store:store.close()
        if companion_store:companion_store.close()


if __name__=='__main__':sys.exit(main())
