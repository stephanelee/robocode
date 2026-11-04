#!/usr/bin/env python3
"""Frozen Facet duel acquisition; official runner, black-box pinned opponents."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import time
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


def validate_metrics(path, rounds):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if len(rows) != rounds or [r['round'] for r in rows] != list(range(1, rounds + 1)):
        raise ValueError('Facet completion must contain every round exactly once')
    if len({r['gameId'] for r in rows}) != 1 or any(r['schema'] != 2 for r in rows):
        raise ValueError('Wrong game identity or telemetry schema')
    for row in rows:
        if not 0 <= row['shotsConfirmed'] <= row['shotsAccepted'] <= row['shotsRequested']:
            raise ValueError('Invalid physical shot transaction counts')
        if row['neighborBytes'] > 134217728:
            raise ValueError('Memory payload cap violated')
    return rows


def summarize(games):
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
            'finalCumulativeTelemetryByGame': [g['finalTelemetry'] for g in selected],
        }
    return summary


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
    if not frozen.get('independentReviewEvidence') or not frozen.get('premeasurementApproved'):
        raise ValueError('Independent premeasurement approval required')
    output.mkdir(parents=True)
    shutil.copy2(args.freeze, output/'registration.json')
    report = {'status': 'preparing', 'freezeSHA256': benchmark.digest(args.freeze), 'runtime': platform.platform(), 'games': [],
              'java': subprocess.check_output([args.java, '-version'], stderr=subprocess.STDOUT, text=True),
              'seatAndEngineSeedControl': 'unsupported; alternating launch order only; rounds within a game dependent'}
    target = output/'results.json'
    save(target, report)
    try:
        cache = args.cache.resolve()
        runner = benchmark.download(cache, 'robocode-tankroyale-runner-1.4.0.jar', benchmark.ASSETS['robocode-tankroyale-runner-1.4.0.jar'])
        report['runnerSHA256'] = benchmark.digest(runner)
        # Packaging utilities verify pinned bytes without exposing source or private telemetry.
        source, python, provenance = native.prepare(cache, args.python)
        retcon_source, api_cache, retcon_provenance = retcon.prepare(cache)
        report.update(pythonOpponents=provenance, retcon=retcon_provenance, status='running')
        save(target, report)
        for opponent in COHORTS:
            for index, seed in enumerate(SEEDS):
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
                benchmark.run_process(cmd, ROOT, env, run/'runner.log', 990)
                # Completion callbacks flush asynchronously; this bounded readiness wait never starts another battle.
                deadline = time.monotonic()+5
                while time.monotonic() < deadline:
                    try:
                        rows = validate_metrics(run/'FACET_metrics.jsonl', 20)
                        if (run/'FACET_model.dat').is_file():
                            break
                    except (ValueError, FileNotFoundError, json.JSONDecodeError):
                        pass
                    time.sleep(.05)
                rows = validate_metrics(run/'FACET_metrics.jsonl', 20)
                result = json.loads((run/'battle.json').read_text())
                names = {'Facet': '0.1.0', 'AISurvivor': None, 'Retcon': '0.2.0', 'diamond-port': '1.8.28'}
                expected_name = next(x['name'] for x in provenance['opponents'] if Path(x['descriptor']).stem == opponent) if opponent == 'diamond-port' else opponent
                if result['numberOfRounds'] != 20 or len(result['results']) != 2 or {r['name'] for r in result['results']} != {'Facet', expected_name}:
                    raise ValueError('Official result roster/round count differs')
                facet = next(r for r in result['results'] if r['name'] == 'Facet')
                if facet['version'] != '0.1.0':
                    raise ValueError('Facet identity differs')
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
