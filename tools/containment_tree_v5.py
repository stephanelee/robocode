"""Fixed shallow nonlinear gain selector. No live controller or hyperparameter search."""
import argparse
from bisect import bisect_right
import json
import math
from pathlib import Path

import containment_forecast_v3 as v3
from legal_scan_dataset import build_samples, load_events, parse_json, validate_manifest

GAIN_CANDIDATES = ('linear', 'circular', 'headon')
PARAMETERS = dict(maxLeafDepth=3, minLeaf=200, quantileCount=32,
    quantileRule='sorted[floor(j*(n-1)/33)],j=1..32,deduplicate;left<=threshold',
    minReduction=1e-12, minimumGain=.02, objective='sum-three-output-squared-error',
    target='candidate-containment-minus-current-containment',
    tieOrder='feature-index,ascending-threshold;candidate=linear,circular,headon')


def thresholds(values):
    ordered = sorted(values)
    return sorted({ordered[j * (len(ordered) - 1) // 33] for j in range(1, 33)})


def build_tree(xs, ys, depth=0):
    n = len(xs)
    if n < PARAMETERS['minLeaf'] or len(ys) != n:
        raise ValueError('At least 200 eligible train labels required per horizon/leaf')
    totals = [sum(y[a] for y in ys) for a in range(3)]
    node = dict(count=n, means=[value / n for value in totals])
    if depth == PARAMETERS['maxLeafDepth'] or n < 2 * PARAMETERS['minLeaf']:
        return node
    best, best_reduction = None, PARAMETERS['minReduction']
    parent_term = sum(value * value / n for value in totals)
    for feature in range(len(v3.FEATURES)):
        order = sorted(range(n), key=lambda i: xs[i][feature])
        values = [xs[i][feature] for i in order]
        cuts = thresholds(values)
        positions = {bisect_right(values, cut): cut for cut in cuts
                     if PARAMETERS['minLeaf'] <= bisect_right(values, cut) <= n - PARAMETERS['minLeaf']}
        sums = [0, 0, 0]
        for left_count, row in enumerate(order, 1):
            for a in range(3):
                sums[a] += ys[row][a]
            if left_count not in positions:
                continue
            reduction = sum(sums[a] ** 2 / left_count + (totals[a] - sums[a]) ** 2 / (n - left_count)
                            for a in range(3)) - parent_term
            if reduction > best_reduction:
                best_reduction = reduction
                best = feature, positions[left_count]
    if best is None:
        return node
    feature, threshold = best
    left = [i for i in range(n) if xs[i][feature] <= threshold]
    right = [i for i in range(n) if xs[i][feature] > threshold]
    node.update(feature=feature, threshold=threshold, reduction=best_reduction,
        left=build_tree([xs[i] for i in left], [ys[i] for i in left], depth + 1),
        right=build_tree([xs[i] for i in right], [ys[i] for i in right], depth + 1))
    return node


def tree_gains(node, values):
    while 'feature' in node:
        node = node['left'] if values[node['feature']] <= node['threshold'] else node['right']
    if node['count'] < PARAMETERS['minLeaf'] or not all(math.isfinite(g) for g in node['means']):
        raise ValueError('Invalid or unsupported frozen leaf')
    return node['means']


def gain_targets(sample, game):
    def contained(name):
        point = v3.v1.baseline(sample, game, name)
        return int(math.hypot(point[0] - sample['label']['x'], point[1] - sample['label']['y']) <= 18)
    current = contained('current')
    return [contained(name) - current for name in GAIN_CANDIDATES]


def fit(records, manifest):
    validate_manifest(manifest)
    games = {g['gameId']: g for g in manifest['games'] if g['split'] == 'train'}
    train_manifest = dict(manifest, games=list(games.values()))
    train_records = [r for r in records if r['gameId'] in games]
    contexts = dict(v3.own_context.contexts(train_records, train_manifest, allowed_splits=('train',)))
    data = {h: ([], []) for h in v3.v1.HORIZONS}
    for sample in build_samples(train_records, train_manifest, v3.v1.HORIZONS, 8):
        if not sample['label']['observed']:
            continue
        game = games[sample['gameId']]
        values, eligible, _ = v3.inputs(sample, game, contexts[v3.sample_key(sample)])
        if eligible:
            data[sample['horizon']][0].append(values)
            data[sample['horizon']][1].append(gain_targets(sample, game))
    return dict(schemaVersion=1, kind='nonlinear-gain-tree-v5', features=list(v3.FEATURES),
        contextSemantics=v3.CONTEXT_SEMANTICS, gainCandidates=list(GAIN_CANDIDATES), parameters=PARAMETERS.copy(),
        horizons=list(v3.v1.HORIZONS), trees={str(h): build_tree(*data[h]) for h in v3.v1.HORIZONS})


def predict(model, sample, game, context):
    values, eligible, _ = v3.inputs(sample, game, context)
    name, gains = 'current', None
    if eligible:
        gains = tree_gains(model['trees'][str(sample['horizon'])], values)
        best = max(range(3), key=lambda a: gains[a])
        if gains[best] >= PARAMETERS['minimumGain']:
            name = GAIN_CANDIDATES[best]
    return v3.v1.baseline(sample, game, name), name, eligible, gains


def screen(records, manifest, model, splits=('dev',)):
    games = {g['gameId']: g for g in manifest['games'] if g['split'] in splits}
    selected_manifest = dict(manifest, games=list(games.values()))
    selected_records = [r for r in records if r['gameId'] in games]
    contexts = dict(v3.own_context.contexts(selected_records, selected_manifest, allowed_splits=splits))
    cells = {g: {str(h): dict(anchors=0, observed=0, eligible=0,
        selections={name: 0 for name in v3.CANDIDATES}, methods={name: dict(errorSum=0.0,
        contained=0, eligibleErrorSum=0.0, eligibleContained=0) for name in ('current', 'scorer')})
        for h in v3.v1.HORIZONS} for g in games}
    for sample in build_samples(selected_records, selected_manifest, v3.v1.HORIZONS, 8):
        cell, game = cells[sample['gameId']][str(sample['horizon'])], games[sample['gameId']]
        cell['anchors'] += 1
        if not sample['label']['observed']:
            continue
        cell['observed'] += 1
        point, choice, eligible, _ = predict(model, sample, game, contexts[v3.sample_key(sample)])
        cell['eligible'] += eligible
        cell['selections'][choice] += 1
        for name, p in (('current', v3.v1.baseline(sample, game, 'current')), ('scorer', point)):
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


def source_hashes():
    return dict(toolSha256=v3.v1.sha(__file__), v3ToolSha256=v3.v1.sha(v3.__file__),
        dependencySha256=v3.v1.sha(v3.v1.__file__), contextToolSha256=v3.v1.sha(v3.own_context.__file__),
        datasetToolSha256=v3.v1.sha(Path(__file__).with_name('legal_scan_dataset.py')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'events', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--stage', choices=('fit', 'screen'), default='fit')
    parser.add_argument('--model', type=Path)
    parser.add_argument('--model-sha256')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('New output required; preserve all prior evidence')
    manifest = parse_json(args.manifest.read_text(encoding='utf-8'))
    validate_manifest(manifest)
    permitted = {g['gameId'] for g in manifest['games'] if g['split'] in ('train', 'dev')}
    records = [r for r in load_events(args.events) if r['gameId'] in permitted]
    provenance = dict(manifestSha256=v3.v1.sha(args.manifest), eventsSha256=v3.v1.sha(args.events), **source_hashes())
    if args.stage == 'fit':
        model = fit(records, manifest)
        model.update(provenance, priorGameIds=[g['gameId'] for g in manifest['games']],
            priorAcquisitionIds=[g['acquisitionId'] for g in manifest['games']], captureProvenance=manifest['provenance'])
        report = dict(stage='fit', scope='train-only fixed tree; empirical gains are not calibrated confidence')
    else:
        if not args.model or not args.model_sha256 or v3.v1.sha(args.model) != args.model_sha256:
            raise ValueError('Frozen model SHA256 required')
        model = parse_json(args.model.read_text(encoding='utf-8'))
        if any(model[key] != value for key, value in provenance.items()):
            raise ValueError('Frozen model/source/input hash mismatch')
        if (model['kind'] != 'nonlinear-gain-tree-v5' or model['features'] != list(v3.FEATURES)
                or model['contextSemantics'] != v3.CONTEXT_SEMANTICS or model['parameters'] != PARAMETERS
                or model['gainCandidates'] != list(GAIN_CANDIDATES) or model['horizons'] != list(v3.v1.HORIZONS)):
            raise ValueError('Frozen model definition differs from the prospective family')
        cells = screen(records, manifest, model)
        report = dict(stage='screen', cells=cells, decision=v3.decision(cells), modelSha256=args.model_sha256,
            scope='development screen only; no fresh confirmation or live strength claim')
    args.output.mkdir(parents=True)
    if args.stage == 'fit':
        (args.output / 'model.json').write_text(json.dumps(model, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        report['modelSha256'] = v3.v1.sha(args.output / 'model.json')
    report.update(provenance)
    (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(report if args.stage == 'fit' else {'modelSha256': args.model_sha256, 'decision': report['decision']}, indent=2))


if __name__ == '__main__':
    main()
