"""Fixed four-candidate containment ranking; train/dev only, no live controller."""
import argparse
import json
import math
from pathlib import Path

import motion_forecast as v1
from legal_scan_dataset import build_samples, load_events, parse_json, validate_manifest

CANDIDATES = ('current', 'linear', 'circular', 'headon')


class BinaryMoments:
    def __init__(self):
        n = len(v1.FEATURES)
        self.count = 0
        self.sx, self.sy = [0.0] * n, [0.0] * 4
        self.xx = [[0.0] * n for _ in range(n)]
        self.xy = [[0.0] * 4 for _ in range(n)]

    def add(self, x, labels):
        self.count += 1
        for a in range(4):
            self.sy[a] += labels[a]
        for i in range(len(x)):
            self.sx[i] += x[i]
            for j in range(i + 1):
                self.xx[i][j] += x[i] * x[j]
            for a in range(4):
                self.xy[i][a] += x[i] * labels[a]

    def model(self):
        if self.count < 2:
            raise ValueError('Insufficient eligible training labels')
        n, count = len(v1.FEATURES), self.count
        mean = [x / count for x in self.sx]
        scale = [math.sqrt(max(0, self.xx[i][i] / count - mean[i] ** 2)) or 1.0 for i in range(n)]
        gram = [[(self.xx[max(i, j)][min(i, j)] - count * mean[i] * mean[j]) / (scale[i] * scale[j])
                 + (v1.ALPHA if i == j else 0) for j in range(n)] for i in range(n)]
        coefficients = [v1.solve(gram, [(self.xy[i][a] - mean[i] * self.sy[a]) / scale[i]
                                       for i in range(n)]) for a in range(4)]
        return dict(count=count, mean=mean, scale=scale,
                    intercept=[y / count for y in self.sy], coefficients=coefficients)


def fit(records, manifest):
    games = validate_manifest(manifest)
    moments = {h: BinaryMoments() for h in v1.HORIZONS}
    # Partition before constructing windows: no old withheld/transfer enters this tool.
    train_manifest = dict(manifest, games=[g for g in manifest['games'] if g['split'] == 'train'])
    train_ids = {g['gameId'] for g in train_manifest['games']}
    train_records = [r for r in records if r['gameId'] in train_ids]
    for sample in build_samples(train_records, train_manifest, v1.HORIZONS, 8):
        if not sample['label']['observed']:
            continue
        game = games[sample['gameId']]
        x, eligible, _ = v1.inputs(sample, game)
        if eligible:
            labels = []
            for name in CANDIDATES:
                point = v1.baseline(sample, game, name)
                labels.append(int(math.hypot(point[0] - sample['label']['x'], point[1] - sample['label']['y']) <= 18))
            moments[sample['horizon']].add(x, labels)
    return dict(schemaVersion=1, kind='binary-containment-ranking-v2', features=list(v1.FEATURES),
                candidates=list(CANDIDATES), alpha=v1.ALPHA, horizons=list(v1.HORIZONS),
                models={str(h): moments[h].model() for h in v1.HORIZONS})


def predict(model, sample, game):
    x, eligible, _ = v1.inputs(sample, game)
    name = 'current'
    scores = None
    if eligible:
        m = model['models'][str(sample['horizon'])]
        z = [(v - mean) / scale for v, mean, scale in zip(x, m['mean'], m['scale'])]
        scores = [m['intercept'][a] + sum(c * value for c, value in zip(m['coefficients'][a], z)) for a in range(4)]
        if not all(math.isfinite(s) for s in scores):
            raise ValueError('Invalid nonfinite ranking scores')
        name = CANDIDATES[max(range(4), key=lambda a: scores[a])]
    return v1.baseline(sample, game, name), name, eligible, scores


def screen(records, manifest, model, splits=('dev',)):
    games = {g['gameId']: g for g in manifest['games'] if g['split'] in splits}
    dev_manifest = dict(manifest, games=list(games.values()))
    dev_records = [r for r in records if r['gameId'] in games]
    cells = {g: {str(h): dict(anchors=0, observed=0, eligible=0,
        selections={name: 0 for name in CANDIDATES}, methods={name: dict(errorSum=0.0,
        contained=0, eligibleErrorSum=0.0, eligibleContained=0) for name in ('current', 'scorer')})
        for h in v1.HORIZONS} for g in games}
    for sample in build_samples(dev_records, dev_manifest, v1.HORIZONS, 8):
        cell, game = cells[sample['gameId']][str(sample['horizon'])], games[sample['gameId']]
        cell['anchors'] += 1
        if not sample['label']['observed']:
            continue
        cell['observed'] += 1
        point, selected, eligible, _ = predict(model, sample, game)
        cell['eligible'] += eligible
        cell['selections'][selected] += 1
        for name, p in (('current', v1.baseline(sample, game, 'current')), ('scorer', point)):
            error = math.hypot(p[0] - sample['label']['x'], p[1] - sample['label']['y'])
            metric = cell['methods'][name]
            metric['errorSum'] += error
            metric['contained'] += error <= 18
            if eligible:
                metric['eligibleErrorSum'] += error
                metric['eligibleContained'] += error <= 18
    for horizons in cells.values():
        for cell in horizons.values():
            cell['coverage'] = cell['observed'] / cell['anchors'] if cell['anchors'] else 0
            for metric in cell['methods'].values():
                for prefix, count in (('', cell['observed']), ('eligible', cell['eligible'])):
                    metric[prefix + 'MeanError' if prefix else 'meanError'] = metric[prefix + 'ErrorSum' if prefix else 'errorSum'] / count if count else None
                    metric[prefix + 'Containment' if prefix else 'containment'] = metric[prefix + 'Contained' if prefix else 'contained'] / count if count else None
    return cells


def decision(cells):
    coverage = len(cells) == 3 and all(cells[g][str(h)]['observed'] >= 50 for g in cells for h in v1.PRIMARY)
    coverage = coverage and all(cells[g][str(h)]['observed'] > 0 for g in cells for h in (8, 16))
    if not coverage:
        return dict(status='inconclusive', checks={'coverage': False})
    deltas = [v1.average(cells, g, 'scorer', 'containment') - v1.average(cells, g, 'current', 'containment') for g in cells]
    checks = dict(coverage=True, primaryGain=sum(deltas) / len(deltas) >= .02,
                  improvedGames=sum(d > 0 for d in deltas) >= 2,
                  worstGame=all(d >= -.02 for d in deltas),
                  shortHorizons=all(v1.average(cells, g, 'scorer', 'containment', (8, 16)) >=
                      v1.average(cells, g, 'current', 'containment', (8, 16)) - .02 for g in cells))
    return dict(status='development-pass' if all(checks.values()) else 'reject', checks=checks,
                primaryContainmentDelta=sum(deltas) / len(deltas), gameDeltas=dict(zip(cells, deltas)))


def confirmation_decision(cells, manifest):
    retained = {g['gameId']: cells[g['gameId']] for g in manifest['games'] if g['split'] == 'withheld'}
    result = decision(retained)
    transfer = [g['gameId'] for g in manifest['games'] if g['split'] == 'transfer']
    if result['status'] == 'inconclusive' or len(transfer) != 3 or any(
            cells[g][str(h)]['observed'] == 0 for g in transfer for h in v1.HORIZONS):
        return dict(status='inconclusive', checks={'coverage': False})
    result['checks']['transfer'] = all(
        v1.average(cells, g, 'scorer', 'containment') >= v1.average(cells, g, 'current', 'containment') - .02 and
        v1.average(cells, g, 'scorer', 'meanError') <= v1.average(cells, g, 'current', 'meanError') + 5 for g in transfer)
    result['status'] = 'offline-confirmation-pass' if all(result['checks'].values()) else 'reject'
    return result


def validate_fresh(manifest, model):
    games = manifest['games']
    if len(games) != 6 or sum(g['split'] == 'withheld' for g in games) != 3 or sum(g['split'] == 'transfer' for g in games) != 3:
        raise ValueError('Fresh confirmation requires three withheld and three transfer games')
    if any(g['gameId'] in model['priorGameIds'] or g['acquisitionId'] in model['priorAcquisitionIds'] for g in games):
        raise ValueError('Previously inspected games cannot become fresh confirmation')
    for key in ('botJarSha256', 'engineVersion', 'collectorVersion'):
        if manifest['provenance'][key] != model['captureProvenance'][key]:
            raise ValueError('Confirmation capture runtime/collector/JAR differs')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'events', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--stage', choices=('fit', 'screen', 'confirm'), default='fit')
    parser.add_argument('--model', type=Path)
    parser.add_argument('--model-sha256')
    parser.add_argument('--confirmation-manifest-sha256')
    parser.add_argument('--screen-report', type=Path)
    parser.add_argument('--screen-report-sha256')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve prior outputs; output must be new')
    manifest = parse_json(args.manifest.read_text(encoding='utf-8'))
    validate_manifest(manifest)
    permitted = {g['gameId'] for g in manifest['games'] if args.stage == 'confirm' or g['split'] in ('train', 'dev')}
    records = [r for r in load_events(args.events) if r['gameId'] in permitted]
    provenance = dict(manifestSha256=v1.sha(args.manifest), eventsSha256=v1.sha(args.events),
        toolSha256=v1.sha(__file__), dependencySha256=v1.sha(v1.__file__),
        datasetToolSha256=v1.sha(Path(__file__).with_name('legal_scan_dataset.py')))
    if args.stage == 'fit':
        model = fit(records, manifest)
        model.update(provenance)
        model.update(priorGameIds=[g['gameId'] for g in manifest['games']],
                     priorAcquisitionIds=[g['acquisitionId'] for g in manifest['games']],
                     captureProvenance=manifest['provenance'])
        report = dict(stage='fit', scope='train only; ranking scores are not calibrated probabilities')
    else:
        if not args.model or not args.model_sha256 or v1.sha(args.model) != args.model_sha256:
            raise ValueError('Frozen model SHA256 required before development screen')
        model = parse_json(args.model.read_text(encoding='utf-8'))
        compared = provenance if args.stage == 'screen' else {k: v for k, v in provenance.items() if k not in ('manifestSha256', 'eventsSha256')}
        if any(model[k] != value for k, value in compared.items()):
            raise ValueError('Frozen input/source hashes differ')
        if model['features'] != list(v1.FEATURES) or model['candidates'] != list(CANDIDATES) or model['alpha'] != v1.ALPHA:
            raise ValueError('Frozen model specification differs')
        if args.stage == 'confirm':
            if v1.sha(args.manifest) != args.confirmation_manifest_sha256:
                raise ValueError('Reviewed immutable fresh manifest SHA256 required')
            if not args.screen_report or v1.sha(args.screen_report) != args.screen_report_sha256:
                raise ValueError('Frozen successful development report SHA256 required')
            prior = parse_json(args.screen_report.read_text(encoding='utf-8'))
            if prior['stage'] != 'screen' or prior['modelSha256'] != args.model_sha256 or prior['decision']['status'] != 'development-pass':
                raise ValueError('Fresh confirmation is conditional on the same model passing development')
            if any(prior[k] != model[k] for k in provenance):
                raise ValueError('Development report does not match frozen training inputs/source')
            validate_fresh(manifest, model)
        cells = screen(records, manifest, model, ('withheld', 'transfer') if args.stage == 'confirm' else ('dev',))
        report = dict(stage=args.stage, cells=cells,
                      decision=confirmation_decision(cells, manifest) if args.stage == 'confirm' else decision(cells),
                      modelSha256=args.model_sha256, trainingManifestSha256=model['manifestSha256'],
                      trainingEventsSha256=model['eventsSha256'],
                      scope='offline legal-label containment only; no live strength claim')
        if args.stage == 'confirm':
            report['screenReportSha256'] = args.screen_report_sha256
    args.output.mkdir(parents=True)
    if args.stage == 'fit':
        (args.output / 'model.json').write_text(json.dumps(model, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        report['modelSha256'] = v1.sha(args.output / 'model.json')
    report.update(provenance)
    (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(report if args.stage == 'fit' else {'modelSha256': args.model_sha256, 'decision': report['decision']}, indent=2))


if __name__ == '__main__':
    main()
