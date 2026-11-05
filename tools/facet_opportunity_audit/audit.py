#!/usr/bin/env python3
"""Fixed-population own-Facet firing audit. No battles, models or training."""
import argparse
from collections import Counter
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
import zipfile

ARCHIVE_SHA = 'ef24b160d461008783fb7bb293c114addbcaa066ebcd40ca48ea383d74143ac4'
REGISTRATION_SHA = '1713338d6b6cd9d0d271173a4243f238642913ebe3e717d72b471feaf07ecb58'
JAR_SHA = '9b489b8429b01ff96caba378fbb481ddac6811a9a48b522649b3e4ea0eccb174'
SOURCE_SHA = '22c032253036c936116046f939b0196c8291d45597c9e6ee2387055f64be4f4b'
PREFIX = 'build/facet-prepared-cohort-100/'
MAX_LINES = 100000
NUMBER = r'([-+0-9.eE]+)'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def numeric(text, name):
    match = re.search(r'(?<![A-Za-z])' + re.escape(name) + '=' + NUMBER + r'(?=[ ,\]]|$)', text)
    if not match:
        raise ValueError('Missing numeric field: ' + name)
    value = float(match[1])
    if not math.isfinite(value):
        raise ValueError('Nonfinite field: ' + name)
    return value


def vector(text, name):
    match = re.search(re.escape(name) + r'=V\[x=' + NUMBER + ', y=' + NUMBER + r'\]', text)
    if not match:
        raise ValueError('Missing vector: ' + name)
    return tuple(map(float, match.groups()))


def parse_plan(text):
    if 'plan=null' in text:
        return None
    if 'plan=Plan[' not in text:
        raise ValueError('Full serialized plan required')
    text = text.split('plan=Plan[', 1)[1]
    plan = {key: numeric(text, key) for key in
            ('angle', 'power', 'target', 'created', 'expires', 'tolerance', 'observationTurn')}
    for key in ('target', 'created', 'expires', 'observationTurn'):
        if int(plan[key]) != plan[key]:
            raise ValueError('Noninteger plan identity')
        plan[key] = int(plan[key])
    plan['origin'] = vector(text, 'origin')
    plan['targetPosition'] = vector(text, 'targetPosition')
    body = text.split('targetBody=Body[', 1)[1].split('], policy=', 1)[0]
    plan['targetBody'] = dict(position=vector(body, 'p'), heading=numeric(body, 'heading'), speed=numeric(body, 'speed'))
    policy = text.split('policy=Policy[', 1)[1].split(']', 1)[0]
    plan['policy'] = {key: numeric(policy, key) for key in ('population', 'range', 'targetEnergy', 'observed')}
    plan['gun'] = re.search(r'gun=(analytic|long|recent|combinedmelee)\b', text)[1]
    return plan


class Evidence:
    """Only exact own members can be opened. Own model exports are excluded."""
    def __init__(self, root):
        root = root.resolve()
        manifest = json.loads((root / 'artifact-provenance.json').read_text())
        if manifest['artifactSHA256'] != ARCHIVE_SHA or len(manifest['parts']) != 25:
            raise ValueError('Fixed archive identity required')
        archive = io.BytesIO()
        for index, part in enumerate(manifest['parts']):
            name = f'raw-artifact.zip.part{index:03d}'
            if part['path'] != name:
                raise ValueError('Part order/name')
            path = (root / name).resolve()
            if not path.is_relative_to(root) or path.stat().st_size > 1048576:
                raise ValueError('Part boundary/cap')
            data = path.read_bytes()
            if len(data) != part['bytes'] or sha(data) != part['sha256']:
                raise ValueError('Part identity')
            archive.write(data)
        if archive.tell() != 26010585 or sha(archive.getvalue()) != ARCHIVE_SHA:
            raise ValueError('Archive identity')
        self.archive = zipfile.ZipFile(archive)
        self.allowed = {'registration.json'} | {
            f'diamond-port-g{i:02d}/{leaf}' for i in range(1, 6)
            for leaf in ('FACET_trace.jsonl', 'FACET_metrics.jsonl', 'FACET_metrics.jsonl.critical.jsonl', 'battle.json')}
        for name in self.allowed:
            info = self.archive.getinfo(PREFIX + name)
            cap = 67108864 if name.endswith('/FACET_trace.jsonl') else 2097152
            if info.file_size > cap:
                raise ValueError('Own member cap')
        registration = self.read('registration.json')
        if sha(registration) != REGISTRATION_SHA:
            raise ValueError('Registration identity')
        self.registration = json.loads(registration)
        if self.registration['facetSourceSHA256'] != SOURCE_SHA or self.registration['facetJarSHA256'] != JAR_SHA:
            raise ValueError('Candidate identity')

    def open(self, name):
        if name not in self.allowed:
            raise ValueError('Not an allowed own/public member: ' + name)
        return self.archive.open(PREFIX + name)

    def read(self, name):
        with self.open(name) as stream:
            return stream.read()

    def rows(self, name, cap=MAX_LINES):
        with self.open(name) as stream:
            for number, line in enumerate(stream, 1):
                if number > cap or len(line) > 65536:
                    raise ValueError('Record count/size cap')
                yield number, json.loads(line)


class Replay:
    def __init__(self, java_home, jar, directory):
        if jar.stat().st_size > 5242880 or sha(jar.read_bytes()) != JAR_SHA:
            raise ValueError('Hash-pinned own measured JAR required')
        java = str(java_home / 'bin/java')
        version = subprocess.run([java, '-version'], check=True, capture_output=True, text=True)
        if 'version "17.' not in version.stderr + version.stdout:
            raise ValueError('Java17 required')
        self.version = version.stderr + version.stdout
        self.env = {key: value for key, value in os.environ.items()
                    if not key.startswith(('FACET_', 'AISURVIVOR_', 'RETCON_'))
                    and key not in ('JAVA_TOOL_OPTIONS', '_JAVA_OPTIONS', 'JDK_JAVA_OPTIONS')}
        source = Path(__file__).with_name('GateReplay.java')
        subprocess.run([str(java_home / 'bin/javac'), '--release', '17', '-cp', str(jar.resolve()),
                        '-d', str(directory), str(source)], check=True, env=self.env)
        self.process = subprocess.Popen([java, '-Xmx128m', '-cp', str(directory) + os.pathsep + str(jar.resolve()),
                                         'GateReplay'], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        text=True, env=self.env)
        self.count = 0

    def check(self, own, plan, target, pending):
        body = plan['targetBody']; policy = plan['policy']
        fields = [own['turn'], own['x'], own['y'], own['gun'], own['heat'], own['energy'], own['population'], int(pending),
                  plan['angle'], plan['power'], plan['target'], plan['created'], plan['expires'], plan['tolerance'],
                  *plan['origin'], plan['observationTurn'], *plan['targetPosition'], *body['position'], body['heading'], body['speed'],
                  int(policy['population']), policy['range'], policy['targetEnergy'], int(policy['observed']),
                  target['turn'], target['x'], target['y'], target['heading'], target['speed'], target['energy'], int(target['alive'])]
        if len(fields) != 34 or any(not math.isfinite(v) for v in fields):
            raise ValueError('Replay schema')
        self.count += 1
        self.process.stdin.write('\t'.join(map(str, fields)) + '\n'); self.process.stdin.flush()
        result = self.process.stdout.readline().rstrip('\n').split('\t')
        if len(result) != 5:
            raise ValueError('Own-JAR gate helper failed')
        return dict(valid=result[0] == 'true', drift=float(result[1]), half=float(result[2]),
                    policyPower=float(result[3]), failed=result[4].split(',') if result[4] else [])

    def close(self):
        self.process.stdin.close()
        self.process.stdout.close()
        if self.process.wait(timeout=30):
            raise ValueError('Gate replay exit')


def analyze_game(evidence, index, replay):
    base = f'diamond-port-g{index:02d}/'
    counts = Counter(); blockers = Counter(); cold_blockers = Counter(); aligned_blockers = Counter()
    combinations = Counter(); kinds = Counter(); guns = Counter(); examples = {}; rounds = {}
    own = None; active_plan = None; pending = None; scans = {}; alive = {}; frame = None
    prior_turn = None; active_round = None; prepared_intent = None
    shots = {}; outcomes = {}; native_fired = set(); submitted = {}; states = {}
    trace_digest = hashlib.sha256(); last = None
    last_prepared = None
    # Hash and parse the same bounded stream; no second trace pass.
    with evidence.open(base + 'FACET_trace.jsonl') as stream:
        for line_number, line in enumerate(stream, 1):
            if line_number > MAX_LINES or len(line) > 65536:
                raise ValueError('Trace row cap')
            trace_digest.update(line); row = json.loads(line)
            kind = row['kind']; text = row['value']; key = (row['round'], row['turn'])
            kinds[kind] += 1; last = key
            if kind == 'own.state':
                if active_round != row['round']:
                    active_plan = None; pending = None; scans = {}; alive = {}; prior_turn = None
                    active_round = row['round']
                if prior_turn is not None and row['turn'] not in (prior_turn, prior_turn + 1):
                    active_plan = None; pending = None
                prior_turn = row['turn']; frame = key
                own = {k: numeric(text, k) for k in ('x', 'y', 'heading', 'speed', 'gun', 'heat', 'energy', 'population')}
                own['turn'] = row['turn']; states[key] = dict(own)
                if pending is not None and row['turn'] > pending + 1:
                    pending = None
            elif kind == 'BotDeathEvent':
                identity = int(text.split(' callbackSequence=')[0]); alive[identity] = False
                if active_plan and active_plan['target'] == identity:
                    active_plan = None
            elif kind == 'SkippedTurnEvent':
                active_plan = None; pending = None
            elif kind == 'scan.observed':
                scan = {k: numeric(text, k) for k in ('id', 'x', 'y', 'directionDeg', 'speed', 'energy')}
                identity = int(scan.pop('id')); scan['turn'] = row['turn']; scan['heading'] = math.radians(scan.pop('directionDeg'))
                if own is not None and scan['turn'] <= own['turn'] and alive.get(identity, True) and scan['turn'] > scans.get(identity, {}).get('turn', -1):
                    scan['alive'] = True; scans[identity] = scan
            elif kind == 'command.prepared':
                if frame != key or own is None:
                    raise ValueError('Command without matching observed frame')
                actual = ' fire=true ' in text
                counts['preparedDecisions'] += 1
                round_info = rounds.setdefault(key[0], dict(decisions=0, coolDecisions=0, submissions=0, firstRequest=None))
                round_info['decisions'] += 1
                cool = own['heat'] == 0
                counts['coolDecisions' if cool else 'hotDecisions'] += 1
                round_info['coolDecisions'] += int(cool)
                gate = None; failed = []; aligned = False
                if active_plan is None:
                    counts['noPriorActivePlan'] += 1; failed = ['noPriorActivePlan']
                elif active_plan['target'] not in scans:
                    counts['missingTargetScan'] += 1; failed = ['missingTargetScan']
                else:
                    target = dict(scans[active_plan['target']]); target['alive'] = alive.get(active_plan['target'], True)
                    gate = replay.check(own, active_plan, target, pending is not None)
                    failed = gate['failed']
                    if active_plan['policy']['population'] != own['population']:
                        failed = failed + ['populationPolicy']
                    if own['population'] != 1:
                        failed = failed + ['nonDuelUnreplayedCoreGate']
                    counts['mechanicallyReplayedDecisions'] += 1
                    expected = not failed
                    counts['replayMatchesIntent' if expected == actual else 'replayDiffersIntent'] += 1
                    if expected != actual:
                        examples.setdefault('replayDiffersIntent', [])
                        if len(examples['replayDiffersIntent']) < 3:
                            examples['replayDiffersIntent'].append(dict(line=line_number, round=key[0], turn=key[1], actual=actual, replay=gate))
                    aligned = 'alignment' not in failed and own['energy'] > active_plan['power'] and cool
                    if aligned:
                        counts['coolFundedAlignedWithActivePlan'] += 1
                        if not actual:
                            counts['coolFundedAlignedHeld'] += 1; aligned_blockers.update(failed)
                    guns[active_plan['gun']] += int(actual)
                    if cool and not actual and gate['failed'] == ['analyticDrift']:
                        counts['soleAnalyticDriftBlocker'] += 1
                        counts['soleAnalyticDriftGun.' + active_plan['gun']] += 1
                        group = examples.setdefault('soleAnalyticDriftBlocker', [])
                        if len(group) < 5:
                            group.append(dict(line=line_number, round=key[0], turn=key[1], driftDegrees=math.degrees(gate['drift']),
                                              allowedDegrees=math.degrees(gate['half']), own=own, plan=active_plan, target=target))
                if not actual:
                    blockers.update(failed); combinations['+'.join(sorted(failed)) or 'unexplainedHold'] += 1
                    if cool:
                        cold_blockers.update(failed)
                prepared_intent = actual
                last_prepared = dict(round=key[0], turn=key[1], line=line_number,
                                     fireIntent=actual, cool=cool, fundedAligned=aligned,
                                     failedPredicates=list(failed))
                # Serialized replacement becomes next frame's active plan. A submission below clears it.
                active_plan = parse_plan(text)
            elif kind == 'fire.submitted':
                if prepared_intent is not True or frame != key:
                    raise ValueError('Fire acceptance outside prepared intent')
                if 'accepted=true' in text:
                    pending = row['turn']; counts['acceptedPrefixRequests'] += 1
                else:
                    counts['rejectedPrefixRequests'] += 1
                counts['prefixRequests'] += 1; active_plan = None
                round_info = rounds[key[0]]; round_info['submissions'] += 1
                if round_info['firstRequest'] is None:
                    round_info['firstRequest'] = key[1]
            elif kind == 'command.submitted':
                actual = ' fire=true ' in text
                if key != frame or actual != prepared_intent:
                    raise ValueError('Prepared/submitted command disagreement')
                submitted[key] = actual; counts['submittedDecisions'] += 1
                counts['submittedFireCommands'] += int(actual)
            elif kind == 'BulletFiredEvent':
                native_fired.add((key[0], int(text.split(' callbackSequence=')[0]), key[1]))
            elif kind == 'fire.confirmed':
                pending = None
                identity = int(numeric(text, 'id')); request = int(numeric(text, 'request'))
                gun = re.search(r'gun=(analytic|long|recent|combinedmelee)\b', text)[1]
                shot_key = (key[0], identity)
                if shot_key in shots:
                    raise ValueError('Duplicate confirmation')
                counts['confirmedPrefixShots'] += 1
                request_key = (key[0], request); state = states.get(request_key)
                passed = False
                if state and submitted.get(request_key) and (key[0], identity, key[1]) in native_fired:
                    point = re.search(r'position=' + NUMBER + ',' + NUMBER, text)
                    heading_error = abs(math.remainder(math.radians(numeric(text, 'heading')) - state['gun'], 2 * math.pi))
                    distance = math.hypot(float(point[1])-state['x'], float(point[2])-state['y'])
                    passed = heading_error <= 1e-7 and distance <= 1e-6 and key[1] == request + 1
                counts['nativeHeadingOriginNextTickPass' if passed else 'nativeTimingEvidenceGapOrMismatch'] += 1
                shots[shot_key] = dict(gun=gun, request=request, confirmation=key[1])
            elif kind in ('BulletHitBotEvent', 'BulletHitWallEvent', 'BulletHitBulletEvent'):
                identity = text.split(' callbackSequence=')[0].split(':')
                candidates = {int(identity[1])} if kind == 'BulletHitBotEvent' else {int(identity[0])}
                if kind == 'BulletHitBulletEvent': candidates.add(int(identity[2]))
                for bullet in candidates:
                    shot_key = (key[0], bullet)
                    if shot_key in shots:
                        if shot_key in outcomes:
                            counts['multipleOutcomeEvidenceRecords'] += 1
                            group = examples.setdefault('multipleOutcomeEvidence', [])
                            if len(group) < 3:
                                group.append(dict(round=key[0], bullet=bullet, prior=list(outcomes[shot_key]),
                                                  additional=dict(kind=kind, turn=key[1], line=line_number)))
                        outcomes.setdefault(shot_key, []).append(dict(kind=kind, turn=key[1], line=line_number))
    if line_number != MAX_LINES:
        raise ValueError('Fixed capped input count differs')
    if counts['preparedDecisions'] != counts['submittedDecisions']:
        counts['unsubmittedPreparedAtCap'] = counts['preparedDecisions'] - counts['submittedDecisions']
        if counts['unsubmittedPreparedAtCap'] != 1 or (last_prepared['round'], last_prepared['turn']) in submitted:
            raise ValueError('Unsupported command evidence gap')
        counts['unsubmittedCoolAtCap'] = int(last_prepared['cool'])
        counts['unsubmittedCoolFundedAlignedAtCap'] = int(last_prepared['fundedAligned'])
        counts['unsubmittedSoleAnalyticDriftAtCap'] = int(last_prepared['failedPredicates'] == ['analyticDrift'])
    else:
        last_prepared = None
    outcome_counts = Counter(); by_gun = {}
    for key, shot in shots.items():
        events = outcomes.get(key, [])
        outcome = '+'.join(sorted({event['kind'] for event in events})) if events else 'notObservedBeforeCapOrTerminal'
        outcome_counts[outcome] += 1; by_gun.setdefault(shot['gun'], Counter())[outcome] += 1
    metric_rows = [row for _, row in evidence.rows(base + 'FACET_metrics.jsonl', 20)]
    if [row['round'] for row in metric_rows] != list(range(1, 21)):
        raise ValueError('Completion identity')
    final = metric_rows[-1]
    return dict(game=index, traceSHA256=trace_digest.hexdigest(), traceRows=line_number,
                lastIdentity=dict(round=last[0], turn=last[1]), recordKinds=dict(kinds), counts=dict(counts),
                observedFailingPredicatesForHolds=dict(blockers), coolHoldFailingPredicates=dict(cold_blockers),
                coolFundedAlignedHoldFailingPredicates=dict(aligned_blockers), holdCombinations=dict(combinations),
                requestedByGun=dict(guns), observedOutcomes=dict(outcome_counts),
                observedOutcomesByGun={k: dict(v) for k, v in by_gun.items()}, examples=examples,
                unsubmittedPreparedAtCapEvidence=last_prepared,
                coveredRounds=rounds, fullGameConfirmed=final['shotsConfirmed'], fullGameHits=final['counters'].get('shot.hit', 0))


def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    evidence = Evidence(args.parts_root)
    with tempfile.TemporaryDirectory(prefix='facet-opportunity-') as directory:
        replay = Replay(args.java_home, args.jar, Path(directory))
        try:
            games = [analyze_game(evidence, index, replay) for index in range(1, 6)]
        finally:
            replay.close()
        totals = Counter(); blockers = Counter(); aligned = Counter(); combinations = Counter(); outcomes = Counter()
        for game in games:
            totals.update(game['counts']); blockers.update(game['coolHoldFailingPredicates'])
            aligned.update(game['coolFundedAlignedHoldFailingPredicates']); combinations.update(game['holdCombinations'])
            outcomes.update(game['observedOutcomes'])
        result = dict(schema=1, purpose='Fixed own-trace offline diagnostic; zero new games/training/parameter edits',
                      artifactSHA256=ARCHIVE_SHA, registrationSHA256=REGISTRATION_SHA, sourceSHA256=SOURCE_SHA,
                      jarSHA256=JAR_SHA, traceRows=500000, totals=dict(totals), coolHoldFailingPredicates=dict(blockers),
                      coolFundedAlignedHoldFailingPredicates=dict(aligned), holdCombinations=dict(combinations),
                      observedOutcomes=dict(outcomes), games=games,
                      limitations=['Overlapping failed predicates are computed checks, not internally logged first rejection reasons.',
                                   'Gate replay uses the frozen own JAR and reconstructed serialized state, not a new engine battle.',
                                   'Prefixes are selected/capped; missing later or terminal outcomes are not misses or independent observations.',
                                   'No complete target-crossing/prediction-label trace exists; hit/wall counts do not identify aiming-error causes.',
                                   'No opponent files/logs or own exported models opened. No training data emitted.'])
        (args.output / 'analysis.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        (args.output / 'runtime.txt').write_text(replay.version)
        print(json.dumps({k: result[k] for k in ('traceRows', 'totals', 'coolHoldFailingPredicates',
                                                'coolFundedAlignedHoldFailingPredicates', 'observedOutcomes')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parts-root', type=Path, required=True)
    parser.add_argument('--jar', type=Path, required=True)
    parser.add_argument('--java-home', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
