"""Execute exactly one independently cleared Peak screen OR fresh confirmation stage."""
import argparse
import json
import os
from pathlib import Path
import resource
import shutil
import socket
import subprocess
import time

import benchmark
import retcon
import strong_melee_opponents as native
from meridian_peak_common import ROOT, REG, OUT, CACHE, save, metrics, passes_screen, confirmation_decision
from meridian_peak_profile import BASE, CONTROL_JAR, PACKAGE_SHA, PROVENANCE, preflight, validate_payload, check_host, verify_existing
import meridian_peak_verify as prepared


def games(stage, frozen):
    rows = json.loads((REG / 'games.json').read_text())
    for game in rows:
        if game['enemy'] == 'Walls':
            game['version'] = frozen['walls']['version']
    return [game for game in rows if game['stage'] == stage]


def check_confirmation():
    """Require independently audited archived PASS, never infer clearance from favorable rows."""
    audit = json.loads((REG / 'screen-audit.json').read_text())
    evidence = REG / 'screen-evidence'
    if audit.get('decision') != 'approved-screen-pass' or not audit.get('reviewer'):
        raise ValueError('Independent archived screen PASS audit missing')
    if audit.get('freezeSHA256') != benchmark.digest(REG / 'freeze.json'):
        raise ValueError('Confirmation candidate/tool/source freeze differs from audited screen')
    files = audit.get('evidenceFilesSHA256', {})
    if not {'decision.json', 'activation.json', 'profile.json', 'freeze.json', 'package-contents.json', 'host.json', 'prepared-environment.json'} <= files.keys():
        raise ValueError('Audit omits required archived stage provenance')
    if set(files) != {str(p.relative_to(evidence)) for p in evidence.rglob('*') if p.is_file()}:
        raise ValueError('Audited screen file manifest is incomplete or contains extra paths')
    for name, expected in files.items():
        path = evidence / name
        if not path.resolve().is_relative_to(evidence.resolve()) or benchmark.digest(path) != expected:
            raise ValueError('Audited screen archive differs: ' + name)
    decision = json.loads((evidence / 'decision.json').read_text())
    if decision.get('status') != 'complete' or decision.get('stage') != 'screen' or decision.get('freezeSHA256') != audit['freezeSHA256'] or not passes_screen(decision['games']):
        raise ValueError('Archived screen does not pass the unchanged registered rules')
    activation = json.loads((evidence / 'activation.json').read_text())
    if (activation.get('stage') != 'screen' or activation.get('receiptRef') != receipt_ref('screen')
            or audit.get('screenReceiptObject') != activation.get('receiptObject') or not activation.get('receiptObject')):
        raise ValueError('Archived screen durable receipt is missing or unaudited')
    host = json.loads((evidence / 'host.json').read_text())
    if (host != activation.get('hostIdentity') or benchmark.digest(evidence / 'host.json') != activation.get('hostDocumentSHA256')
            or benchmark.digest(evidence / 'prepared-environment.json') != host.get('preparationSHA256')):
        raise ValueError('Archived prepared-environment proof differs from the audited stage host/receipt binding')
    profile = json.loads((evidence / 'profile.json').read_text())
    if benchmark.digest(evidence / 'freeze.json') != audit['freezeSHA256'] or benchmark.digest(evidence / 'package-contents.json') != PACKAGE_SHA:
        raise ValueError('Archived screen source/package freeze differs')
    validate_payload(profile)  # Either preallowed image independently; all frozen assets/tools/packages identical.
    frozen = json.loads((REG / 'freeze.json').read_text())
    expected_games = games('screen', frozen)
    for game, archived in zip(expected_games, decision['games']):
        if any(archived.get(key) != value for key, value in game.items()):
            raise ValueError('Archived screen schedule changed')
        path = evidence / 'raw' / game['id']
        # Reconcile source recordings and model DTOs again, with exact identities and no game starts.
        actual = metrics(path, game)
        if actual != {key: archived[key] for key in actual}:
            raise ValueError('Archived screen reduction differs from audited official/private evidence')
    return audit


def receipt_ref(stage):
    if stage not in ('screen', 'confirmation'):
        raise ValueError('Unregistered acquisition stage')
    return 'refs/tags/experiment-receipts/meridian-peak-v1-' + stage


def validate_receipt(receipt, stage, head):
    if (receipt.get('experiment') != 'meridian-peak-v1' or receipt.get('stage') != stage
            or receipt.get('payloadSHA') != head or os.environ.get('PAYLOAD_SHA') != head
            or receipt.get('runID') != os.environ.get('GITHUB_RUN_ID')
            or receipt.get('runAttempt') != 1 or os.environ.get('GITHUB_RUN_ATTEMPT') != '1'
            or receipt.get('receiptRef') != receipt_ref(stage)
            or receipt.get('freezeSHA256') != benchmark.digest(REG / 'freeze.json')
            or len(receipt.get('receiptObject', '')) != 40):
        raise ValueError('Wrong stage/head/run/freeze durable activation receipt')
    check_host(stage, receipt)


def control_bundle(execution):
    folder = execution / 'control'
    (folder / 'bot').mkdir(parents=True)
    original = json.loads((ROOT / 'benchmarks/meridian-aim/profile.json').read_text())['assets']
    for name in ('Meridian.jar', 'bot/Meridian.json', 'bot/Meridian.sh', 'bot/Meridian.cmd'):
        key = 'experiments/Meridian/' + name
        raw = subprocess.check_output(['git', 'show', BASE + ':' + key], cwd=ROOT)
        target = folder / name
        target.write_bytes(raw)
        if benchmark.digest(target) != original[key]:
            raise ValueError('Frozen v3 control bundle differs')
    if benchmark.digest(folder / 'Meridian.jar') != CONTROL_JAR:
        raise ValueError('Wrong original-v3 control JAR')
    (folder / 'bot/Meridian.sh').chmod(0o755)
    return folder


def port_free(port):
    if port == 7654:
        raise ValueError('Interactive server is forbidden')
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', port))


def play(game, profile, frozen, execution, control):
    from meridian_peak_profile import stage_walls
    data, stage = execution / 'raw' / game['id'], execution / 'stages' / game['id']
    data.mkdir(parents=True)
    stage.mkdir(parents=True)
    (stage / 'home').mkdir()
    bundle = control if game['arm'] == 'control' else ROOT / 'experiments/MeridianPeak'
    own = stage / game['ownName']
    shutil.copytree(bundle / 'bot', own)
    shutil.copy2(bundle / (game['ownName'] + '.jar'), own / (game['ownName'] + '.jar'))
    (own / (game['ownName'] + '.sh')).chmod(0o755)
    env = {key: os.environ[key] for key in ('PATH', 'JAVA_HOME', 'LANG') if key in os.environ}
    env.update(HOME=str(stage / 'home'), PYTHONDONTWRITEBYTECODE='1')
    if game['arm'] == 'control':
        env.update(MERIDIAN_SEED=str(game['seed']), MERIDIAN_METRICS=str(data / 'markers.txt'))
    else:
        env.update(MERIDIAN_PEAK_SEED=str(game['seed']), MERIDIAN_PEAK_METRICS=str(data / 'markers.txt'),
                   MERIDIAN_PEAK_STATE_DIR=str(data / 'state'))
    if game['enemy'] == 'Retcon':
        rc = prepared.verify_retcon()
        source = CACHE / 'retcon' / rc['commit'] / 'Retcon'
        api = CACHE / 'retcon-api'
        other = stage / 'Retcon'
        benchmark.isolate_opponent(source, other)
        if retcon.source_hash(other) != rc['sourceSHA256']:
            raise ValueError('Retcon source tree differs')
        env['RUMBLE_JAVA_CACHE'] = str(api)
    elif game['enemy'] == 'Diamond Port':
        base = CACHE / 'strong-python' / profile['runtime']['python']['commit']
        env.update(native.stage(base / 'source', stage, base / 'venv/bin/python'))
        other = stage / 'opponents/diamond-port'
        benchmark.isolate_opponent(base / 'source/bots/ports/diamond-port', other)
        native.verify_staged(stage, ['diamond-port'])
    elif game['enemy'] == 'AISurvivor':
        other = stage / 'AISurvivor'
        other.mkdir()
        for name in ('AISurvivor.jar', 'AISurvivor.json', 'AISurvivor.sh', 'AISurvivor.cmd'):
            shutil.copy2(ROOT / 'AISurvivor' / name, other / name)
        (other / 'AISurvivor.sh').chmod(0o755)
        env.update(AISURVIVOR_MODEL_PATH=str(data / 'ais-model.dat'), AISURVIVOR_METRICS_PATH=str(data / 'ais.jsonl'),
                   AISURVIVOR_SEED=str(game['seed']), AISURVIVOR_MODE='train')
    elif game['enemy'] == 'Walls':
        other = stage_walls(stage, frozen['walls'])
    else:
        raise ValueError('Unregistered opponent')
    command = [str(Path(env['JAVA_HOME']) / 'bin/java'), '-cp', str(CACHE / 'robocode-tankroyale-runner-1.4.0.jar'),
               str(REG / 'MeridianPeakBattle.java'), str(game['port']), str(game['rounds']), str(data), str(own), str(other)]
    save(data / 'attempt.json', dict(game=game, command=command, environment=env,
        stagedSHA256={str(p.relative_to(stage)): benchmark.digest(p) for p in stage.rglob('*') if p.is_file()},
        profileSHA256=benchmark.digest(PROVENANCE / 'profile.json'), freezeSHA256=benchmark.digest(REG / 'freeze.json'),
        payloadSHA=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()))
    port_free(game['port'])
    before, start = resource.getrusage(resource.RUSAGE_CHILDREN), time.monotonic()
    try:
        benchmark.run_process(command, stage, env, data / 'runner.log', 1680)
    finally:
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        save(data / 'process-cost.json', dict(elapsedWallSeconds=time.monotonic()-start,
             waitedChildUserSecondsDelta=after.ru_utime-before.ru_utime,
             waitedChildSystemSecondsDelta=after.ru_stime-before.ru_stime,
             definition='Python RUSAGE_CHILDREN delta for the owned runner/process tree when reaped; includes server/observer/opponents'))
    result = metrics(data, game)
    result['processCost'] = json.loads((data / 'process-cost.json').read_text())
    return result


def run(stage):
    profile = preflight(stage, require_frozen=True)
    frozen = validate_payload(profile)
    if stage == 'confirmation':
        check_confirmation()
    receipt = json.loads((PROVENANCE / (stage + '-activation.json')).read_text())
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    validate_receipt(receipt, stage, head)
    execution = OUT / stage / 'acquisition'
    execution.mkdir(parents=True)  # Refuse complete or partial prior root; never resume.
    save(execution / 'activation.json', receipt)
    save(execution / 'profile.json', profile)
    shutil.copy2(REG / 'freeze.json', execution / 'freeze.json')
    shutil.copy2(PROVENANCE / 'package-contents.json', execution / 'package-contents.json')
    shutil.copy2(PROVENANCE / 'prepared-environment.json', execution / 'prepared-environment.json')
    shutil.copy2(PROVENANCE / (stage + '-host.json'), execution / 'host.json')
    report = dict(stage=stage, status='pending', freezeSHA256=benchmark.digest(REG / 'freeze.json'), games=[], decision='pending')
    save(execution / 'decision.json', report)
    try:
        control = control_bundle(execution) if stage == 'screen' else None
        for game in games(stage, frozen):
            report['inProgressGame'] = game
            save(execution / 'decision.json', report)
            try:
                check_host(stage, receipt)
                verify_existing(stage)
                validate_payload(profile)
                row = dict(**game, status='valid', **play(game, profile, frozen, execution, control))
            except Exception as error:
                report['games'].append(dict(**game, status='invalid', error=str(error)))
                raise
            report['games'].append(row)
            del report['inProgressGame']
            save(execution / 'decision.json', report)
        decision = passes_screen(report['games']) if stage == 'screen' else confirmation_decision(report['games'])
        report.update(status='complete', decision=decision)
    except BaseException as error:
        report.update(status='invalid', decision='incomplete; no comparison decision', error=str(error))
        raise
    finally:
        save(execution / 'decision.json', report)
        save(execution / 'artifact-files.json', {str(p.relative_to(execution)): benchmark.digest(p) for p in execution.rglob('*') if p.is_file()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('screen', 'confirmation'), required=True)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if args.run:
        run(args.stage)
    else:
        profile = preflight(args.stage, require_frozen=True)
        if args.stage == 'confirmation':
            check_confirmation()
