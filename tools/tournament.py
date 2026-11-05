"""Bounded local tournament execution over a frozen, prebuilt source snapshot.
Official pairing scores are sums over two cold 20-round games. Never reruns a game.
"""
import argparse, datetime, hashlib, json, os, random, shutil, subprocess, sys, time, platform, zipfile, gzip
from functools import lru_cache
from pathlib import Path
NAMES = ['AISurvivor', 'Aegis', 'Breakwater', 'DiamondChallenger', 'Keel', 'KeelHelm', 'Meridian', 'Prism', 'Retcon', 'Diamond Port', 'Adaptive Prime', 'Walls', 'SpinBot', 'RamFire']
PREFIXES = ('AISURVIVOR_', 'AEGIS_', 'BREAKWATER_', 'CHALLENGER_', 'KEEL_', 'KEEL_HELM_', 'MERIDIAN_', 'PRISM_', 'FACET_', 'ROBOCODE_', 'PYTHON', 'PIP_')

def stamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def save(path, data):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temp.replace(path)

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def draw(seed):
    names = NAMES.copy()
    random.Random(seed).shuffle(names)
    return [dict(a=names[i], b=names[i + 1], status='scheduled', games=[]) for i in range(0, len(names), 2)]

def swiss_standings(reports):
    table = {n: dict(name=n, played=0, points=0, buchholz=0, scoreShare=0) for n in NAMES}
    pairs = [p for r in reports for p in r['draw'] if p['status'] == 'complete']
    for pair in pairs:
        total = pair['scoreA'] + pair['scoreB']
        for n, pk, sk in [(pair['a'], 'pointsA', 'scoreA'), (pair['b'], 'pointsB', 'scoreB')]:
            table[n]['played'] += 1
            table[n]['points'] += pair[pk]
            table[n]['scoreShare'] += pair[sk] / total if total else 0.5
    for pair in pairs:
        table[pair['a']]['buchholz'] += table[pair['b']]['points']
        table[pair['b']]['buchholz'] += table[pair['a']]['points']
    for row in table.values():
        if row['played']:
            row['scoreShare'] /= row['played']
    return sorted(table.values(), key=lambda r: (-r['points'], -r['buchholz'], -r['scoreShare'], r['name']))

def swiss_draw(output, number):
    if number == 1:
        return (draw(202611041), dict(algorithm='seeded opening draw', seed=202611041))
    past = []
    for previous in range(1, number):
        report = json.loads((output / f'swiss-{previous}.json').read_text())
        if report['status'] != 'complete' or report['completedGames'] != 14:
            raise ValueError('Every prior Swiss round must be complete')
        past.append(report)
    ranked = swiss_standings(past)
    seen = {frozenset((p['a'], p['b'])) for r in past for p in r['draw']}

    @lru_cache(None)
    def solve(mask):
        if not mask:
            return (0, 0, ())
        i = (mask & -mask).bit_length() - 1
        remaining = mask & ~(1 << i)
        best = None
        for j in range(i + 1, len(ranked)):
            if not remaining & 1 << j or frozenset((ranked[i]['name'], ranked[j]['name'])) in seen:
                continue
            tail = solve(remaining & ~(1 << j))
            if tail is None:
                continue
            candidate = (abs(ranked[i]['points'] - ranked[j]['points']) + tail[0], j - i + tail[1], ((i, j),) + tail[2])
            if best is None or candidate < best:
                best = candidate
        return best
    result = solve((1 << len(ranked)) - 1)
    if result is None:
        raise ValueError('No complete rematch-free matching')
    pairs = [dict(a=ranked[i]['name'], b=ranked[j]['name'], status='scheduled', games=[]) for i, j in result[2]]
    return (pairs, dict(algorithm='Exact minimum-cost perfect matching; lexicographically minimize total point gap, total rank gap, then rank-index pair tuple; no previous Swiss rematches', totalPointGap=result[0], totalRankGap=result[1], inputStandings=ranked))

def validate_result(data, identities):
    if data.get('numberOfRounds') != 20 or len(data.get('results', [])) != 2:
        raise ValueError('Official result coverage differs')
    if sorted(((r['name'], r['version']) for r in data['results'])) != sorted(identities):
        raise ValueError('Official participant identity mismatch')
    for r in data['results']:
        for key in ['totalScore', 'firstPlaces', 'bulletDamage', 'survival']:
            if type(r[key]) is not int or r[key] < 0:
                raise ValueError('Invalid official ' + key)
        if r['firstPlaces'] > 20:
            raise ValueError('First-place count exceeds round budget')

def collect_entry_manifests(source, bases):
    manifests = {}
    for name in NAMES:
        paths = list(bases[name].rglob('*'))
        if name in NAMES[1:8]:
            paths += list((source / 'experiments' / name / 'src').rglob('*')) + [source / 'experiments' / name / 'build.gradle', source / 'experiments' / name / (name + '.jar')]
        if name == 'AISurvivor':
            paths += list((source / 'src/main').rglob('*')) + [source / 'build.gradle']
        manifests[name] = {str(path.relative_to(source)): digest(path) for path in paths if path.is_file()}
    return manifests

def freeze_manifest(path, current, prepared=None):
    if prepared is not None and prepared.is_file() and json.loads(prepared.read_text()) != current:
        raise ValueError('Inputs differ from preparation: ' + str(prepared))
    if path.exists():
        if json.loads(path.read_text()) != current:
            raise ValueError('Frozen tournament inputs changed: ' + str(path))
    else:
        if prepared is not None and not prepared.is_file():
            raise ValueError('Run prepare_tournament.py before starting a new edition')
        save(path, current)

def collect_shared_inputs(source, java, python):
    cache = source / '.benchmark-cache'
    paths = [source / 'tools' / name for name in ['BenchmarkBattle.java', 'benchmark.py', 'retcon.py', 'retcon_stats.py', 'strong_melee_opponents.py', 'verify_jar.py', 'opponents/retcon.lock.json', 'opponents/strong-melee.lock.json']] + list((source / 'samples/lib').rglob('*')) + list((cache / 'retcon-api').glob('*.jar'))
    files = {str(path.relative_to(source)): digest(path) for path in paths if path.is_file()}
    # Verify extracted shared/sample bytes against the already hash-pinned release ZIP.
    with zipfile.ZipFile(cache / 'sample-bots-Java-1.4.0.zip') as archive:
        prefixes = tuple(name + '/' for name in NAMES[-3:]) + ('lib/',)
        for entry in archive.infolist():
            if not entry.is_dir() and entry.filename.startswith(prefixes):
                actual = source / 'samples' / entry.filename
                if not actual.is_file() or digest(actual) != hashlib.sha256(archive.read(entry)).hexdigest():
                    raise ValueError('Extracted official sample/library differs: ' + entry.filename)
    python_runtime = json.loads(subprocess.check_output([str(python), '-c', 'import sys,json,importlib.metadata as m; print(json.dumps(dict(version=sys.version,packages=sorted((d.metadata["Name"],d.version) for d in m.distributions()))))'], text=True))
    return dict(files=files, javaSHA256=digest(java), javaVersion=subprocess.check_output([str(java), '-version'], stderr=subprocess.STDOUT, text=True), pythonSHA256=digest(python), pythonRuntime=python_runtime)

def recording_harness_inputs(harness):
    return dict(sourceSHA256=digest(harness), policy='Observer recording only; original battle settings and bot sources unchanged')

def collect_recording(directory, output):
    files = list(directory.glob('*.battle.gz'))
    if len(files) != 1 or files[0].stat().st_size == 0:
        raise ValueError('Expected one nonempty .battle.gz recording')
    # Read to EOF to check gzip framing/CRC without loading the whole replay into RAM.
    with gzip.open(files[0], 'rb') as stream:
        while stream.read(1024 * 1024):
            pass
    with files[0].open('rb') as stream:
        sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(path=files[0].relative_to(output).as_posix(), sha256=sha, sizeBytes=files[0].stat().st_size)

def main(args):
    R = args.source.resolve()
    O = args.output.resolve()
    O.mkdir(parents=True, exist_ok=True)
    if args.site and (not (args.site / 'data.json').is_file()):
        raise ValueError('Site export requires an initialized data.json; copy website/ first')
    sys.path.insert(0, str(R / 'tools'))
    import benchmark, retcon, retcon_stats, strong_melee_opponents
    statefile = O / (args.stage + '.json')
    if statefile.exists() and (not args.resume_unplayed):
        raise ValueError('Stage already exists; use explicit unplayed-only continuation after a recorded ruling')
    java = args.java.resolve()
    java_version = subprocess.check_output([str(java), '-version'], stderr=subprocess.STDOUT, text=True)
    if 'version "25.' not in java_version:
        raise ValueError('Battle runtime must be JDK25')
    cache = R / '.benchmark-cache'
    runner = cache / 'robocode-tankroyale-runner-1.4.0.jar'
    if benchmark.digest(runner) != benchmark.ASSETS[runner.name]:
        raise ValueError('Runner pin mismatch')
    ret = cache / 'retcon' / '28981c7a469ac064194b41001cd7183ec5ecf189' / 'Retcon'
    if retcon.source_hash(ret) != retcon_stats.SOURCE_SHA256:
        raise ValueError('Retcon pin mismatch')
    guest = R / 'guest-source'
    lock = strong_melee_opponents.manifest()
    for e in lock['files']:
        if digest(guest / e['path']) != e['sha256']:
            raise ValueError('Guest source pin mismatch: ' + e['path'])
    bases = {n: R / 'AISurvivor' if n == 'AISurvivor' else R / 'experiments' / n / 'bot' for n in NAMES[:8]}
    bases['Retcon'] = ret
    for n, dirname in [('Diamond Port', 'ports/diamond-port'), ('Adaptive Prime', 'adaptive-prime')]:
        bases[n] = guest / 'bots' / dirname
    for n in NAMES[-3:]:
        bases[n] = R / 'samples' / n
    descriptors = {n: next(bases[n].glob('*.json')) for n in NAMES}
    identities = {n: (json.loads(p.read_text())['name'], json.loads(p.read_text())['version']) for n, p in descriptors.items()}
    manifests = collect_entry_manifests(R, bases)
    freeze_manifest(O / 'entry-manifests.json', manifests, O / 'prepared-entry-manifests.json')
    retcon_lock = json.loads(retcon.LOCK.read_text())
    dependency = retcon_lock['botApi']
    if digest(cache / 'retcon-api' / ('robocode-tankroyale-bot-api-' + dependency['version'] + '.jar')) != dependency['sha256']:
        raise ValueError('Retcon API pin mismatch')
    freeze_manifest(O / 'shared-inputs.json', collect_shared_inputs(R, java, args.python.absolute()), O / 'prepared-shared-inputs.json' if (O / 'prepared-entry-manifests.json').exists() else None)
    recording_enabled = not args.no_recordings
    harness = Path(__file__).resolve().with_name('TournamentBattle.java') if recording_enabled else R / 'tools/BenchmarkBattle.java'
    if recording_enabled:
        prepared_recording = O / 'prepared-recording-harness.json' if (O / 'prepared-entry-manifests.json').exists() else None
        freeze_manifest(O / 'recording-harness.json', recording_harness_inputs(harness), prepared_recording)
    playoff = args.stage.startswith('playoff-')
    number = int(args.stage.split('-')[1]) if args.stage.startswith(('swiss-', 'playoff-')) else 0
    if playoff:
        import tournament_playoffs
        if not recording_enabled:
            raise ValueError('Playoffs require the approved recording policy')
        proposed, pairing = tournament_playoffs.context(O, number)
    else:
        proposed, pairing = (draw(202611040), dict(algorithm='seeded practice draw', seed=202611040)) if not number else swiss_draw(O, number)
    seed = pairing.get('seed')
    if args.resume_unplayed:
        report = json.loads(statefile.read_text())
        pairs = report['draw']
        expected = proposed
        if [(p['a'], p['b']) for p in pairs] != [(p['a'], p['b']) for p in expected]:
            raise ValueError('Registered draw differs')
        if playoff and ([p['matchId'] for p in pairs] != [p['matchId'] for p in expected] or report['pairing'] != pairing):
            raise ValueError('Registered playoff bracket differs')
        if any((g['status'] != 'complete' for p in pairs for g in p['games'])):
            raise ValueError('Cannot resume an attempted incomplete game')
        for pi, pair in enumerate(pairs, 1):
            for game in ([1, 2, 3] if playoff else [1, 2]):
                if game == 3 and not tournament_playoffs.decision(pair)['needsExtra']:
                    continue
                if any((g['game'] == game for g in pair['games'])):
                    continue
                if (O / args.stage / f'p{pi:02d}-g{game}').exists():
                    raise ValueError('Unplayed directory exists; retain and relocate prelaunch staging first')
        if report.get('recordingsEnabled', False) != recording_enabled:
            raise ValueError('Continuation must preserve its recording policy; use --no-recordings for an older unrecorded stage')
        report['status'] = 'running'
        report['continuedAt'] = stamp()
    else:
        pairs = proposed
        report = dict(stage=args.stage, status='running', started=stamp(), drawSeed=seed, draw=pairs, plannedGames=2*len(pairs), completedGames=0, completedRounds=0, failures=[], mainCommit='1f6e6a38582838fe891330a0b3e7a7d9f1668279', host=platform.node(), pointsApplied=args.stage != 'practice' and not playoff)
        report['recordingsEnabled'] = recording_enabled
        report['harnessSHA256'] = digest(harness)
        report['pairing'] = pairing
        report['orchestrationSHA256'] = {Path(__file__).name: digest(Path(__file__))}
        if playoff:
            report['orchestrationSHA256']['tournament_playoffs.py'] = digest(Path(tournament_playoffs.__file__))
        save(O / (args.stage + '-prospective-draw.json'), dict(pairing=pairing, draw=pairs, registeredAt=stamp(), gamesPerPair=2, roundsPerGame=20))
    save(statefile, report)
    print(json.dumps(dict(stage=args.stage, draw=[(m['a'], m['b']) for m in pairs])), flush=True)
    if args.register_only:
        report['status'] = 'scheduled'
        save(statefile, report)
        update_site(args.site, report, O)
        print('STAGE REGISTERED; no games launched', flush=True)
        return
    for pi, pair in enumerate(pairs, 1):
        if pair['status'] == 'complete':
            continue
        for game in ([1, 2, 3] if playoff else [1, 2]):
            if game == 3:
                if not tournament_playoffs.decision(pair)['needsExtra']:
                    break
                report['plannedGames'] = 2*len(pairs) + sum(any(g['game'] == 3 for g in p['games']) for p in pairs) + 1
            if any((g['game'] == game for g in pair['games'])):
                continue
            start = time.monotonic()
            run = O / args.stage / f'p{pi:02d}-g{game}'
            run.mkdir(parents=True)
            env = {k: v for k, v in os.environ.items() if not k.startswith(PREFIXES) and k not in ['RETCON_STATS_DIR', 'RUMBLE_JAVA_CACHE', 'JAVA_TOOL_OPTIONS', 'JDK_JAVA_OPTIONS', 'JAVA_HOME', 'SERVER_URL', 'BOT_SECRET', 'SERVER_SECRET']}
            env['JAVA_HOME'] = str(java.parent.parent)
            env['PATH'] = str(java.parent) + os.pathsep + env.get('PATH', '')
            env['RUMBLE_JAVA_CACHE'] = str(cache / 'retcon-api')
            env['ROBOCODE_PYTHON_BIN'] = str(args.python.absolute())
            env['ROBOCODE_ENV_FILE'] = str(run / 'absent-developer.env')
            env['ROBOCODE_GUN_ENV_FILE'] = str(run / 'absent-guns.env')
            env['PYTHONNOUSERSITE'] = '1'
            env['PYTHONDONTWRITEBYTECODE'] = '1'
            participants = [pair['a'], pair['b']]
            if game == 2:
                participants.reverse()
            dirs = []
            for name in participants:
                folder = name if name in NAMES[:8] else bases[name].name
                target = run / 'bots' / folder
                benchmark.isolate_opponent(bases[name], target)
                if name in NAMES[1:8]:
                    shutil.copy2(R / 'experiments' / name / (name + '.jar'), target / (name + '.jar'))
                if name not in NAMES[-3:] and (not (target / (folder + '.sh')).is_file()):
                    raise ValueError('Canonical launcher missing: ' + str(target))
                if name in NAMES[-3:] and (not (target / (folder + '.java')).is_file()):
                    raise ValueError('Official Java source missing: ' + str(target))
                for sh in target.glob('*.sh'):
                    sh.chmod(493)
                dirs.append(target)
            if set(participants) & {'Diamond Port', 'Adaptive Prime'}:
                shutil.copytree(guest / 'bots/bot_core', run / 'bots/bot_core')
                shutil.copy2(guest / 'LICENSE', run / 'bots/LICENSE.shared-Apache-2.0')
            for prefix in ['AISURVIVOR', 'AEGIS', 'BREAKWATER', 'CHALLENGER', 'KEEL', 'KEEL_HELM', 'MERIDIAN', 'PRISM']:
                env[prefix + '_SEED'] = '20261104'
            if 'AISurvivor' in participants:
                env.update(AISURVIVOR_MODE='train', AISURVIVOR_MODEL_PATH=str(run / 'AISurvivor-model.dat'), AISURVIVOR_METRICS_PATH=str(run / 'AISurvivor-metrics.jsonl'))
            if 'DiamondChallenger' in participants:
                env.update(CHALLENGER_MODEL_PATH=str(run / 'DiamondChallenger-model.json'), CHALLENGER_METRICS_PATH=str(run / 'DiamondChallenger-metrics.jsonl'))
            if 'Breakwater' in participants:
                env.update(BREAKWATER_MODEL_PATH=str(run / 'Breakwater-model.json'), BREAKWATER_METRICS_PATH=str(run / 'Breakwater-metrics.jsonl'))
            for name, key in [('Aegis', 'AEGIS_METRICS_PATH'), ('Keel', 'KEEL_METRICS'), ('KeelHelm', 'KEEL_HELM_METRICS'), ('Meridian', 'MERIDIAN_METRICS'), ('Prism', 'PRISM_TELEMETRY_PATH')]:
                if name in participants:
                    env[key] = str(run / (name + '-metrics.jsonl'))
            capture = retcon_stats.Capture(retcon_stats.java_temp_directory(str(java))) if 'Retcon' in participants else None
            if capture:
                env['RETCON_STATS_DIR'] = str(capture.path)
            row = dict(game=game, participants=participants, started=stamp(), directory=str(run.relative_to(O)), status='running')
            pair['games'].append(row)
            save(statefile, report)
            recording_directory = run / 'recordings'
            recording_options = []
            if recording_enabled:
                recording_directory.mkdir()
                recording_options = ['-Dtournament.recordingDir=' + str(recording_directory)]
            command = [str(java), *recording_options, '-cp', str(runner), str(harness), '20', '900', str(run / 'battle.json'), *map(str, dirs)]
            save(run / 'invocation.json', dict(command=command, configuredEnvironment={k: v for k, v in env.items() if k.startswith(PREFIXES) or k in ['RETCON_STATS_DIR', 'RUMBLE_JAVA_CACHE', 'JAVA_HOME']}, started=row['started']))
            print(f"START {args.stage} p{pi:02d} g{game}: {' vs '.join(participants)}", flush=True)
            try:
                benchmark.run_process(command, R, env, run / 'runner.log', 960)
                data = json.loads((run / 'battle.json').read_text())
                validate_result(data, [identities[n] for n in participants])
                if 'AISurvivor' in participants:
                    metrics = [json.loads(l) for l in (run / 'AISurvivor-metrics.jsonl').read_text().splitlines() if l.strip()]
                    benchmark.validate_metrics(metrics, 20, 'train', 20261104, 0)
                row.update(status='complete', official=data)
                if recording_enabled:
                    row['recording'] = collect_recording(recording_directory, O)
                report['completedGames'] += 1
                report['completedRounds'] += 20
            except Exception as error:
                row.update(status='failed', error=str(error))
                report['failures'].append(dict(pair=pi, game=game, error=str(error)))
                if (run / 'battle.json').exists():
                    row['unvalidatedOfficial'] = json.loads((run / 'battle.json').read_text())
            finally:
                if capture:
                    retid = next((r['id'] for r in row.get('official', row.get('unvalidatedOfficial', {})).get('results', []) if r['name'] == 'Retcon'), None)
                    row['retconStats'] = capture.collect(run / 'retcon-stats', 20, retid, require=False)
                    capture.close()
            row['elapsedSeconds'] = round(time.monotonic() - start, 3)
            row['finished'] = stamp()
            save(statefile, report)
            update_site(args.site, report, O)
            print(f"END p{pi:02d} g{game}: {row['status']} {row['elapsedSeconds']}s", flush=True)
            if playoff and row['status'] != 'complete':
                report.update(status='incomplete', finished=stamp())
                pair['status'] = 'incomplete'
                save(statefile, report)
                update_site(args.site, report, O)
                print('STAGE INCOMPLETE; no automatic rerun or advancement', flush=True)
                return
        if all((g['status'] == 'complete' for g in pair['games'])):
            scores = {n: sum((next((r['totalScore'] for r in g['official']['results'] if r['name'] == identities[n][0])) for g in pair['games'])) for n in [pair['a'], pair['b']]}
            wins = {n: sum((next((r['firstPlaces'] for r in g['official']['results'] if r['name'] == identities[n][0])) for g in pair['games'])) for n in scores}
            a, b = (pair['a'], pair['b'])
            pair.update(status='complete', scoreA=scores[a], scoreB=scores[b], roundWinsA=wins[a], roundWinsB=wins[b], winner=a if scores[a] > scores[b] else b if scores[b] > scores[a] else None)
            if playoff:
                resolution = tournament_playoffs.decision(pair)
                if not resolution['winner']:
                    raise ValueError('Playoff match has no resolved winner')
                pair.update({k: v for k, v in resolution.items() if k != 'needsExtra'})
            elif args.stage != 'practice':
                pair.update(pointsA=3 if scores[a] > scores[b] else 0 if scores[a] < scores[b] else 1, pointsB=3 if scores[b] > scores[a] else 0 if scores[b] < scores[a] else 1)
        else:
            pair['status'] = 'incomplete'
        save(statefile, report)
        update_site(args.site, report, O)
    report['status'] = 'complete' if not report['failures'] else 'incomplete'
    report['finished'] = stamp()
    save(statefile, report)
    update_site(args.site, report, O)
    print('STAGE ' + report['status'].upper(), flush=True)

def public_pair(pair):
    out = {k: v for k, v in pair.items() if k in ['a', 'b', 'status', 'scoreA', 'scoreB', 'roundWinsA', 'roundWinsB', 'winner', 'pointsA', 'pointsB', 'matchId', 'resolution', 'lotterySHA256']}
    out['games'] = []
    for game in pair.get('games', []):
        public = {k: v for k, v in game.items() if k in ['game', 'participants', 'status']}
        if game.get('status') == 'complete':
            official = game['official']
            public['official'] = dict(numberOfRounds=official['numberOfRounds'], results=[{k: v for k, v in row.items() if k in ['id', 'name', 'version', 'rank', 'totalScore', 'survival', 'bulletDamage', 'firstPlaces']} for row in official['results']])
        out['games'].append(public)
    return out

def public_stage(report):
    out = {k: v for k, v in report.items() if k in ['stage', 'status', 'drawSeed', 'plannedGames', 'completedGames', 'completedRounds', 'pointsApplied']}
    out['draw'] = [public_pair(p) for p in report['draw']]
    return out

def update_site(site, report, output=None):
    if site is None:
        return
    path = site / 'data.json'
    data = json.loads(path.read_text())
    data['updated'] = stamp()
    practice = report if report['stage'] == 'practice' else None
    if output is not None and (output / 'practice.json').exists():
        practice = json.loads((output / 'practice.json').read_text())
    if practice is not None:
        data['practice'] = [public_pair(pair) for pair in practice['draw']]
        data['completedGames'] = practice['completedGames']
    if report['stage'] == 'practice':
        data['status'] = 'Practice ' + report['status']
    elif report['stage'].startswith('playoff-'):
        import tournament_playoffs
        if output is None:
            raise ValueError('Canonical output directory required for playoff exports')
        number = int(report['stage'].split('-')[1])
        registration = json.loads((output / 'playoff-registration.json').read_text())
        rounds = [public_stage(json.loads((output / f'swiss-{i}.json').read_text())) for i in range(1, 5)]
        data['rounds'] = rounds
        data['standings'] = swiss_standings(rounds)
        data['qualified'] = registration['seeds']
        playoffs = [public_stage(report if i == number else json.loads((output / f'playoff-{i}.json').read_text())) for i in range(1, number+1)]
        data['playoffs'] = playoffs
        data['playoffStandings'], data['champion'] = tournament_playoffs.standings(registration['seeds'], playoffs)
        data['status'] = ('Champion: ' + data['champion']) if data['champion'] else ('Playoff wave ' + str(number) + ' ' + report['status'])
    else:
        number = int(report['stage'].split('-')[1])
        if number > 1 and output is None:
            raise ValueError('Canonical output directory is required for cumulative exports')
        rounds = []
        for prior in range(1, number + 1):
            canonical = report if prior == number else json.loads((output / f'swiss-{prior}.json').read_text())
            rounds.append(public_stage(canonical))
        data['rounds'] = rounds
        data['standings'] = swiss_standings(rounds)
        data['status'] = 'Swiss round ' + str(number) + ' ' + report['status']
    save(path, data)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--java', type=Path, required=True)
    p.add_argument('--python', type=Path, required=True)
    p.add_argument('--site', type=Path)
    p.add_argument('--stage', choices=['practice', *[f'swiss-{i}' for i in range(1, 5)], *[f'playoff-{i}' for i in range(1, 8)]], required=True)
    p.add_argument('--resume-unplayed', action='store_true')
    p.add_argument('--register-only', action='store_true', help='Freeze and export this draw without launching games')
    p.add_argument('--no-recordings', action='store_true', help='Explicitly opt out; future games record .battle.gz by default')
    main(p.parse_args())
