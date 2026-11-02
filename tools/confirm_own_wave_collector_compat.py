"""One explicit recorder-version compatibility exception; frozen V3 math/gates unchanged."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

import containment_forecast_v3 as v3
from legal_scan_dataset import load_events, object_keys, parse_json, validate_manifest

MODEL_SHA = 'a34109512f3d451ceee64b1e47342d91b07eae8be3f1670896c448925562ddcc'
SCREEN_SHA = '4070177e8917fc9a131f31e6c63f215a78996bfcf665ae512412cb32126b9005'
OLD = dict(botJarSha256='1b99fa0ac303e5e5f89ed91bb3803841ded54910628f335fe1aeaeba65e34c9a', collectorVersion='legal-scan-v1')
NEW = dict(botJarSha256='5ebfa64adf38ab227766b97f7f68e82aefb20251ff8d5f30575bf6b69e67c883', collectorVersion='legal-scan-v1.1-closed-round')
SOURCE_PINS = {
    'toolSha256': ('containment_forecast_v3.py', '0dae90437a67e7b3add3b5950f5bfc7971970bb6fee5973fc80fec269a85c38b'),
    'dependencySha256': ('motion_forecast.py', 'bf9e4a16a05bab1f0772403f7f8d0d51244616cf2dd74b8f9f52baebea0f5732'),
    'contextToolSha256': ('own_wave_context_diagnostic.py', 'd7749e737a9d2042f9270daed63bb2b60a054f26174533e9319c164b85bffea2'),
    'datasetToolSha256': ('legal_scan_dataset.py', '66efeb6ea4fdffdb16ce31f1d50da1dcd0fdcec03dd4dfe4a74985c5f4fd6885'),
}
FAILED_IDS = {'w3-retcon-01', 'w3-target-01', 'w3-retcon-02', 'w3-spinbot-01', 'w3-retcon-03', 'w3-walls-01'}
FAILED_ACQUISITIONS = {'own-wave-v3-' + name for name in FAILED_IDS}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def full_commit(value):
    if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{40}', value):
        raise ValueError('Full immutable commit SHA required')
    return value


def registered(commit, path, root):
    full_commit(commit)
    if Path(path).is_absolute() or '..' in Path(path).parts:
        raise ValueError('Registration must be a repository-relative path')
    raw = subprocess.check_output(['git', 'show', f'{commit}:{Path(path).as_posix()}'], cwd=root)
    return parse_json(raw.decode('utf-8')), hashlib.sha256(raw).hexdigest()


def validate_registration(registration, manifest, model, capture_source):
    object_keys(registration, ('schemaVersion', 'modelSha256', 'screenReportSha256', 'oldCollector',
        'newCollector', 'collectorSourceCommit', 'engineVersion', 'games', 'excludedGameIds', 'excludedAcquisitionIds'),
        ('reason', 'failedAttemptRun'))
    validate_manifest(manifest)
    if type(registration['schemaVersion']) is not int or registration['schemaVersion'] != 1:
        raise ValueError('Unsupported compatibility registration')
    if registration['modelSha256'] != MODEL_SHA or registration['screenReportSha256'] != SCREEN_SHA:
        raise ValueError('Only the frozen successful V3 model/screen pair is allowed')
    if registration['oldCollector'] != OLD or registration['newCollector'] != NEW:
        raise ValueError('Only the explicitly reviewed collector/JAR pair transition is allowed')
    full_commit(registration['collectorSourceCommit'])
    full_commit(capture_source)
    provenance = manifest['provenance']
    if provenance['sourceCommit'] != capture_source or registration['engineVersion'] != '1.4.0' or provenance['engineVersion'] != '1.4.0':
        raise ValueError('Actual capture source/engine differs from registration')
    if any(provenance[k] != value for k, value in NEW.items()) or any(model['captureProvenance'][k] != value for k, value in OLD.items()):
        raise ValueError('Actual or training collector provenance differs')
    if manifest['games'] != registration['games']:
        raise ValueError('Whole fresh manifest roster/order/pins/seeds differs from immutable registration')
    games = manifest['games']
    if len(games) != 6 or sum(g['split'] == 'withheld' for g in games) != 3 or sum(g['split'] == 'transfer' for g in games) != 3:
        raise ValueError('Exactly three fresh withheld and three transfer games are required')
    for key, failed in (('excludedGameIds', FAILED_IDS), ('excludedAcquisitionIds', FAILED_ACQUISITIONS)):
        if not isinstance(registration[key], list) or not failed <= set(registration[key]):
            raise ValueError('Registration must explicitly exclude all failed-attempt identities')
    excluded_games = set(model['priorGameIds']) | set(registration['excludedGameIds'])
    excluded_captures = set(model['priorAcquisitionIds']) | set(registration['excludedAcquisitionIds'])
    if any(g['gameId'] in excluded_games or g['acquisitionId'] in excluded_captures for g in games):
        raise ValueError('Old, spent or failed game identities cannot be reused')


def validate_frozen(model_path, screen_path, tools_dir):
    if sha(model_path) != MODEL_SHA or sha(screen_path) != SCREEN_SHA:
        raise ValueError('Frozen model or development report bytes changed')
    model = parse_json(model_path.read_text(encoding='utf-8'))
    screen = parse_json(screen_path.read_text(encoding='utf-8'))
    if screen['stage'] != 'screen' or screen['modelSha256'] != MODEL_SHA or screen['decision']['status'] != 'development-pass':
        raise ValueError('The frozen model must have passed its original development screen')
    for key, (filename, expected) in SOURCE_PINS.items():
        if sha(tools_dir / filename) != expected or model[key] != expected or screen[key] != expected:
            raise ValueError('Frozen scorer/helper/dependency source changed: ' + filename)
    for key in ('manifestSha256', 'eventsSha256'):
        if model[key] != screen[key]:
            raise ValueError('Development screen and training input hashes disagree')
    return model, screen


def validate_code(registration, manifest, root):
    source = manifest['provenance']['sourceCommit']
    subprocess.run(['git', 'diff', '--quiet', registration['collectorSourceCommit'], source, '--',
        'src/main/java', 'src/test/java', 'AISurvivor/AISurvivor.jar'], cwd=root, check=True)
    jar = subprocess.check_output(['git', 'show', source + ':AISurvivor/AISurvivor.jar'], cwd=root)
    if hashlib.sha256(jar).hexdigest() != NEW['botJarSha256']:
        raise ValueError('Actual capture source does not contain the registered new collector JAR')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'events', 'model', 'screen-report', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--registration-commit', required=True)
    parser.add_argument('--registration-path', required=True)
    parser.add_argument('--capture-source-commit', required=True)
    parser.add_argument('--confirmation-manifest-sha256', required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Output must be new; preserve every prior result')
    root = Path(__file__).resolve().parents[1]
    model, screen = validate_frozen(args.model, args.screen_report, Path(__file__).parent)
    registration, registration_sha = registered(args.registration_commit, args.registration_path, root)
    manifest = parse_json(args.manifest.read_text(encoding='utf-8'))
    if sha(args.manifest) != args.confirmation_manifest_sha256:
        raise ValueError('Actual confirmation manifest SHA256 differs from the capture artifact')
    validate_registration(registration, manifest, model, args.capture_source_commit)
    validate_code(registration, manifest, root)
    records = list(load_events(args.events))
    if any(row['gameId'] not in {g['gameId'] for g in manifest['games']} for row in records):
        raise ValueError('Undeclared game in fresh observations')
    # Call the original frozen scorer and original exact gates without monkey-patching.
    cells = v3.screen(records, manifest, model, ('withheld', 'transfer'))
    decision = v3.confirmation_decision(cells, manifest)
    report = dict(stage='confirm-collector-compat', cells=cells, decision=decision,
        modelSha256=MODEL_SHA, screenReportSha256=SCREEN_SHA,
        originalTrainingProvenance=model['captureProvenance'], actualCaptureProvenance=manifest['provenance'],
        trainingManifestSha256=model['manifestSha256'], trainingEventsSha256=model['eventsSha256'],
        actualManifestSha256=sha(args.manifest), actualEventsSha256=sha(args.events),
        registrationCommit=args.registration_commit, registrationPath=args.registration_path,
        registrationSha256=registration_sha, wrapperSha256=sha(__file__),
        frozenSourceHashes={key: pin[1] for key, pin in SOURCE_PINS.items()},
        compatibilityException=dict(old=OLD, new=NEW),
        scope='single explicit collector boundary repair; unchanged model/math/gates; offline mechanism only')
    args.output.mkdir(parents=True)
    (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'decision': decision, 'modelSha256': MODEL_SHA}, indent=2))


if __name__ == '__main__':
    main()
