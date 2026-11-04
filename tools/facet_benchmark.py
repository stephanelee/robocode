#!/usr/bin/env python3
"""Frozen Facet duel acquisition; official runner, black-box pinned opponents."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import statistics
import subprocess
import time
import zipfile
import benchmark
import retcon
import strong_melee_opponents as native

ROOT = Path(__file__).resolve().parents[1]
COHORTS = ('AISurvivor', 'Retcon', 'diamond-port')
SEEDS = (49667, 49668, 49669, 49670, 49671)


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def clean_env():
    prefixes = ('FACET_', 'AISURVIVOR_', 'ROBOCODE_', 'PYTHON', 'PIP_', 'RETCON_', 'CHALLENGER_')
    return {k: v for k, v in os.environ.items() if not k.startswith(prefixes)
            and k not in ('SERVER_URL', 'SERVER_SECRET', 'BOT_SECRET', 'RUMBLE_JAVA_CACHE', 'JAVA_TOOL_OPTIONS', '_JAVA_OPTIONS', 'JDK_JAVA_OPTIONS')}


def run_black_box(command, env, log):
    # Preserve opaque opponent startup/error output without returning source-bearing
    # Python traceback lines to the implementation process or decision system.
    with log.open('w', encoding='utf-8') as stream:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=990)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=15)
            raise RuntimeError('Official runner deadline/interruption; opaque log retained') from None
        if code:
            raise RuntimeError(f'Official runner exit{code}; opaque log retained')


def validate_metrics(path, rounds, expected_schema=2):
    def finite_json(value):
        raise ValueError('Nonfinite telemetry: ' + value)
    rows = [json.loads(line, parse_constant=finite_json) for line in path.read_text().splitlines()]
    if len(rows) != rounds or [r['round'] for r in rows] != list(range(1, rounds + 1)):
        raise ValueError('Facet completion must contain every round exactly once')
    if len({r['gameId'] for r in rows}) != 1 or any(r['schema'] != expected_schema for r in rows):
        raise ValueError('Wrong game identity or telemetry schema')
    for row in rows:
        if expected_schema == 3 and any(row.get(k) != 'success' for k in ('snapshotStatus','terminalObservation','terminalHash')):
            raise ValueError('Explicit failed/quarantined terminal evidence; not a valid completion')
        if not 0 <= row['shotsConfirmed'] <= row['shotsAccepted'] <= row['shotsRequested']:
            raise ValueError('Invalid physical shot transaction counts')
        if row['neighborBytes'] > 134217728:
            raise ValueError('Memory payload cap violated')
    return rows


def metric_summary(row):
    counters = row['counters']
    decisions = counters.get('turn.decisions', 0)
    skipped = counters.get('turn.skipped', 0)
    turns = decisions + skipped
    observations = row['observationsCountSumMinMax']
    def observation(name):
        v = observations.get(name)
        return None if not v else dict(count=v[0], mean=v[1]/v[0] if v[0] else None, min=v[2], max=v[3], sum=v[1])
    def percentile_upper(bins, q):
        total = sum(bins)
        if not total: return None
        cumulative = 0
        for index, count in enumerate(bins):
            cumulative += count
            if cumulative >= q*total: return 2**(index+1)
    return dict(decisions=decisions, reportedSkippedTurns=skipped, botTurnDenominator=turns,
                skipRate=skipped/turns if turns else None,
                wallCollisions=counters.get('collision.wall',0), botCollisions=counters.get('collision.bot',0),
                combinedCollisionRate=(counters.get('collision.wall',0)+counters.get('collision.bot',0))/turns if turns else None,
                requested=row['shotsRequested'], accepted=row['shotsAccepted'], confirmed=row['shotsConfirmed'],
                physicalHits=counters.get('shot.hit',0), bulletContacts=counters.get('bullet.contact',0),
                damageGiven=observation('damageGiven'), damageTaken=observation('damageTaken'),
                range=observation('range'), scanAge=observation('scanAge'), energy=observation('energy'),
                latencyNanosBucketUpper={k:{'p50':percentile_upper(v,.5),'p99':percentile_upper(v,.99),'maxBucket':percentile_upper(v,1)} for k,v in row['timingLog2Nanos'].items()},
                failuresAndFallbacks=counters, gcCount=row['gcCount'], gcMillis=row['gcMillis'])


def summarize(games):
    if len(games) != 15 or any(sum(g['opponent'] == opponent for g in games) != 5 for opponent in COHORTS):
        raise ValueError('Complete fixed fifteen-game schedule required')
    summary = {}
    for opponent in COHORTS:
        selected = [g for g in games if g['opponent'] == opponent]
        if len(selected) != 5:
            raise ValueError('Complete five-game cohort required')
        summary[opponent] = {
            'rounds': 100, 'coldGames': 5,
            'equalGameMeanScoreShare': statistics.mean(g['scoreShare'] for g in selected),
            'officialFirstPlaces': sum(g['facet']['firstPlaces'] for g in selected),
            'gameScoreShares': [g['scoreShare'] for g in selected],
            'metricsByGame': [metric_summary(g['finalTelemetry']) for g in selected],
            'finalCumulativeTelemetryByGame': [g['finalTelemetry'] for g in selected],
        }
    for opponent, cohort in summary.items():
        rows = cohort['metricsByGame']
        turns = sum(r['botTurnDenominator'] for r in rows)
        skipped = sum(r['reportedSkippedTurns'] for r in rows)
        collisions = sum(r['wallCollisions']+r['botCollisions'] for r in rows)
        failures = sum(v for r in rows for k,v in r['failuresAndFallbacks'].items() if k.endswith('.failure'))
        share_pass = None if opponent == 'AISurvivor' else cohort['equalGameMeanScoreShare'] > .4
        rates_pass = turns > 0 and skipped/turns < .005 and collisions/turns < .01
        engineering_pass = False if not rates_pass else None if failures else True
        cohort['registeredGateAssessment'] = dict(turns=turns, skippedTurns=skipped, collisions=collisions,
            caughtSubsystemFailures=failures, scoreShareGatePass=share_pass, engineeringGatePass=engineering_pass,
            decision='reject-profile' if engineering_pass is False or share_pass is False else
                     'requires-independent-failure-classification' if engineering_pass is None else
                     'descriptive-AIS-cohort-no-new-strength-threshold' if share_pass is None else 'passes-requested-cohort-screen-only')
    return summary


def validate_freeze(frozen):
    for relative, expected in frozen['fileSHA256'].items():
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT) or benchmark.digest(path) != expected:
            raise ValueError('Frozen file differs: ' + relative)
    import facet_evidence
    actual = {name: benchmark.digest(ROOT/'experiments/Facet'/name)
              for name in frozen['facetSourceFiles']}
    if facet_evidence.tree_digest(actual) != frozen['facetSourceSHA256']:
        raise ValueError('Frozen source/input manifest differs')
    return actual


def runtime_provenance(java):
    runtime = subprocess.check_output([java, '-XshowSettings:properties', '-version'], stderr=subprocess.STDOUT, text=True)
    os_release = Path('/etc/os-release').read_text()
    if '25.0.1+8' not in runtime or 'Eclipse Adoptium' not in runtime:
        raise ValueError('Expected Temurin25.0.1+8 runtime')
    if platform.machine() != 'x86_64' or 'VERSION_ID="26.04"' not in os_release:
        raise ValueError('Expected Ubuntu26.04 x86_64; no fallback permitted')
    if platform.python_version() != '3.13.9':
        raise ValueError('Expected Python3.13.9 runtime')
    return dict(java=runtime, javaExecutableSHA256=benchmark.digest(Path(java)), python=platform.python_version(),
                pythonExecutableSHA256=benchmark.digest(Path(os.sys.executable)), osRelease=os_release,
                githubImageOS=os.environ.get('ImageOS'), githubImageVersion=os.environ.get('ImageVersion'))


def authorize_registration(frozen):
    if not frozen.get('independentReviewEvidence') or not frozen.get('premeasurementApproved'):
        raise ValueError('Independent premeasurement approval required')
    if frozen.get('freshAcquisitionAfterInvalid') and not frozen.get('userAuthorizedFreshBudget'):
        raise ValueError('Explicit user authorization required for fresh budget after invalid screen')


def main(args):
    output = args.output.resolve()
    if output.exists():
        raise ValueError('Never overwrite or resume a measured attempt')
    frozen = json.loads(args.freeze.read_text())
    if frozen['roundsPerGame'] != 20 or frozen['gamesPerOpponent'] != 5 or frozen['opponents'] != list(COHORTS) or frozen['seeds'] != list(SEEDS):
        raise ValueError('Frozen game structure differs')
    for name, path in [('facetJarSHA256', ROOT/'experiments/Facet/Facet.jar'), ('aisJarSHA256', ROOT/'AISurvivor/AISurvivor.jar'),
                       ('facetConfigSHA256', ROOT/'experiments/Facet/src/main/resources/facet.properties')]:
        if benchmark.digest(path) != frozen[name]:
            raise ValueError('Frozen artifact differs: ' + name)
    authorize_registration(frozen)
    validate_freeze(frozen)
    output.mkdir(parents=True)
    shutil.copy2(args.freeze, output/'registration.json')
    report = {'status': 'preparing', 'freezeSHA256': benchmark.digest(args.freeze), 'runtime': platform.platform(), 'games': [],
              'java': subprocess.check_output([args.java, '-version'], stderr=subprocess.STDOUT, text=True),
              'seatAndEngineSeedControl': 'unsupported; alternating launch order only; rounds within a game dependent'}
    target = output/'results.json'
    save(target, report)
    try:
        report['observedRuntimeBeforeGames'] = runtime_provenance(args.java)
        cache = args.cache.resolve()
        runner = benchmark.download(cache, 'robocode-tankroyale-runner-1.4.0.jar', benchmark.ASSETS['robocode-tankroyale-runner-1.4.0.jar'])
        report['runnerSHA256'] = benchmark.digest(runner)
        if report['runnerSHA256'] != frozen['runnerSHA256']:
            raise ValueError('Frozen official runner differs')
        with zipfile.ZipFile(runner) as official_runner:
            embedded = official_runner.read('robocode-tankroyale-server.jar')
            if hashlib.sha256(embedded).hexdigest() != frozen['serverSHA256']:
                raise ValueError('Frozen official server differs')
        # Packaging utilities verify pinned bytes without exposing source or private telemetry.
        source, python, provenance = native.prepare(cache, args.python)
        retcon_source, api_cache, retcon_provenance = retcon.prepare(cache)
        report.update(pythonOpponents=provenance, retcon=retcon_provenance, status='running')
        save(target, report)
        for opponent in COHORTS:
            for index, seed in enumerate(SEEDS):
                validate_freeze(frozen)
                run = output/f'{opponent}-g{index+1:02d}'
                run.mkdir()
                bot = run/'Facet';bot.mkdir()
                for name in ('Facet.jar', 'Facet.json', 'Facet.sh'):
                    shutil.copy2(ROOT/'experiments/Facet'/name, bot/name)
                (bot/'Facet.sh').chmod(0o755)
                env = clean_env()
                env.update(JAVA_HOME=str(Path(args.java).resolve().parents[1]), PATH=str(Path(args.java).resolve().parent)+os.pathsep+env.get('PATH',''),
                           FACET_SEED=str(seed), FACET_METRICS_PATH=str(run/'FACET_metrics.jsonl'),
                           FACET_TRACE_PATH=str(run/'FACET_trace.jsonl'), FACET_STATE_PATH=str(run/'FACET_model.dat'))
                if opponent == 'AISurvivor':
                    other = run/'opponents/AISurvivor';other.mkdir(parents=True)
                    for name in ('AISurvivor.jar', 'AISurvivor.json', 'AISurvivor.sh'):
                        shutil.copy2(ROOT/'AISurvivor'/name, other/name)
                    (other/'AISurvivor.sh').chmod(0o755)
                    env.update(AISURVIVOR_MODE='train', AISURVIVOR_SEED=str(seed), AISURVIVOR_MODEL_PATH=str(run/'AIS_model.dat'), AISURVIVOR_METRICS_PATH=str(run/'AIS_metrics.jsonl'))
                elif opponent == 'Retcon':
                    other = run/'opponents/Retcon'
                    shutil.copytree(retcon_source, other)
                    env['RUMBLE_JAVA_CACHE'] = str(api_cache)
                    if retcon.source_hash(other) != retcon_provenance['sourceSHA256']:
                        raise ValueError('Staged Retcon source pin differs')
                else:
                    env.update(native.stage(source, run, python))
                    other = run/'opponents/diamond-port'
                    shutil.copytree(source/'bots/ports/diamond-port', other)
                    for script in other.glob('*.sh'):
                        script.chmod(0o755)
                    report['executedOpponentHashes'] = native.verify_staged(run, ['diamond-port'])
                order = [bot, other] if index % 2 == 0 else [other, bot]
                record = {'opponent': opponent, 'game': index+1, 'seed': seed, 'rounds': 20,
                          'launchOrder': [p.name for p in order], 'initialFacetModel': 'absent', 'status': 'running'}
                report['games'].append(record);save(target, report)
                cmd = [args.java, '-cp', str(runner), str(ROOT/'tools/BenchmarkBattle.java'), '20', '900', str(run/'battle.json'), *map(str, order)]
                record['launchArgv'] = cmd
                record['launchStateEnvironment'] = {k:v for k,v in env.items() if k.startswith(('FACET_', 'AISURVIVOR_', 'ROBOCODE_')) or k in ('JAVA_HOME','RUMBLE_JAVA_CACHE','PYTHONNOUSERSITE')}
                save(target, report)
                run_black_box(cmd, env, run/'runner.log')
                # Completion callbacks flush asynchronously; this bounded readiness wait never starts another battle.
                deadline = time.monotonic()+5
                while time.monotonic() < deadline:
                    try:
                        rows = validate_metrics(run/'FACET_metrics.jsonl', 20, frozen.get('metricsSchema', 2))
                        if (run/'FACET_model.dat').is_file():
                            break
                    except (ValueError, FileNotFoundError, json.JSONDecodeError):
                        pass
                    time.sleep(.05)
                rows = validate_metrics(run/'FACET_metrics.jsonl', 20, frozen.get('metricsSchema', 2))
                result = json.loads((run/'battle.json').read_text())
                expected_name = next(x['name'] for x in provenance['opponents'] if Path(x['descriptor']).stem == opponent) if opponent == 'diamond-port' else opponent
                if result['numberOfRounds'] != 20 or len(result['results']) != 2 or {r['name'] for r in result['results']} != {'Facet', expected_name}:
                    raise ValueError('Official result roster/round count differs')
                facet = next(r for r in result['results'] if r['name'] == 'Facet')
                if facet['version'] != '0.1.0':
                    raise ValueError('Facet identity differs')
                expected_version = json.loads((other/(opponent+'.json')).read_text())['version']
                if next(r for r in result['results'] if r['name'] == expected_name)['version'] != expected_version:
                    raise ValueError('Opponent version differs')
                if opponent == 'AISurvivor':
                    benchmark.validate_metrics([json.loads(x) for x in (run/'AIS_metrics.jsonl').read_text().splitlines()], 20, 'train', seed, 20)
                record.update(status='complete', official=result, facet=facet,
                              scoreShare=facet['totalScore']/sum(r['totalScore'] for r in result['results']),
                              finalTelemetry=rows[-1], facetModelSHA256=benchmark.digest(run/'FACET_model.dat'))
                save(target, report)
        report.update(status='complete', summary=summarize(report['games']))
    except BaseException as error:
        report.update(status='invalid', error=str(error))
        raise
    finally:
        save(target, report)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--freeze', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--java', required=True)
    p.add_argument('--python', required=True)
    main(p.parse_args())
