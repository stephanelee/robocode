"""Fixed, train-only residual ridge pilot. Offline radius-18 containment is not a hit rate."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from legal_scan_dataset import build_samples, load_events, parse_json, validate_manifest

HORIZONS = (8, 16, 24, 32, 40)
PRIMARY = (24, 32, 40)
ALPHA = 10.0
FEATURES = ('relative_x', 'relative_y', 'target_vx', 'target_vy', 'turn_rate',
            'acceleration', 'scan_age', 'range', 'wall_left', 'wall_right',
            'wall_bottom', 'wall_top', 'own_vx', 'own_vy', 'intent_speed',
            'intent_turn', 'intent_speed_present', 'intent_turn_present',
            'shot_age', 'shot_power', 'shot_count', 'shot_present',
            'history_present_fraction', 'distinct_scan_count')
BASELINES = ('headon', 'linear', 'circular', 'current')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def velocity(state):
    angle = math.radians(state['direction'])
    return [state['speed'] * math.cos(angle), state['speed'] * math.sin(angle)]


def inputs(sample, game):
    """Deliberately accepts only history, not labels or identity as predictors."""
    frames = sample['features']['history']
    last = frames[-1]
    s, own = last['scan'], last['own']
    scans = {f['scan']['observedTurn']: f['scan'] for f in frames if f['scan'] is not None}
    ordered = sorted(scans.values(), key=lambda x: x['observedTurn'])
    rate = acceleration = 0.0
    if len(ordered) >= 2:
        old, now = ordered[-2:]
        dt = now['observedTurn'] - old['observedTurn']
        rate = ((now['direction'] - old['direction'] + 180) % 360 - 180) / dt
        acceleration = (now['speed'] - old['speed']) / dt
    age = sample['anchorTurn'] - s['observedTurn']
    dx, dy = s['x'] - own['x'], s['y'] - own['y']
    intent = last['previousIntent'] or {}
    shots = {shot['bulletId']: shot for f in frames for shot in f['confirmedShots']}
    latest = max(shots.values(), key=lambda shot: (shot['turn'], shot['bulletId'])) if shots else None
    values = [dx, dy, *velocity(s), rate, acceleration, age, math.hypot(dx, dy),
              s['x'] - 18, game['arenaWidth'] - 18 - s['x'], s['y'] - 18,
              game['arenaHeight'] - 18 - s['y'], *velocity(own),
              intent.get('targetSpeed') or 0, intent.get('turnRate') or 0,
              int(intent.get('targetSpeed') is not None), int(intent.get('turnRate') is not None),
              min(40, sample['anchorTurn'] - latest['turn']) if latest else 40,
              latest['power'] if latest else 0, len(shots), int(bool(shots)),
              sum(f['recordPresent'] for f in frames) / len(frames), len(scans)]
    eligible = age <= 2 and len(scans) >= 2 and all(math.isfinite(x) for x in values)
    return values, eligible, rate


def baseline(sample, game, name):
    s = sample['features']['history'][-1]['scan']
    if name == 'headon':
        return [s['x'], s['y']]
    _, _, rate = inputs(sample, game)
    time = sample['horizon'] + sample['anchorTurn'] - s['observedTurn']
    angle, omega = math.radians(s['direction']), math.radians(rate if name != 'linear' else 0)
    if abs(omega) < 1e-8:
        x, y = s['x'] + math.cos(angle) * s['speed'] * time, s['y'] + math.sin(angle) * s['speed'] * time
    else:
        radius = s['speed'] / omega
        x = s['x'] + radius * (math.sin(angle + omega * time) - math.sin(angle))
        y = s['y'] + radius * (math.cos(angle) - math.cos(angle + omega * time))
    if name == 'current':
        x, y = max(18, min(game['arenaWidth'] - 18, x)), max(18, min(game['arenaHeight'] - 18, y))
    return [x, y]


def solve(matrix, rhs):
    """Positive definite ridge system, Cholesky without numerical dependencies."""
    n = len(rhs)
    lower = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            value = matrix[i][j] - sum(lower[i][k] * lower[j][k] for k in range(j))
            if i == j:
                if value <= 0 or not math.isfinite(value):
                    raise ValueError('Non-positive or non-finite ridge system')
                lower[i][j] = math.sqrt(value)
            else:
                lower[i][j] = value / lower[j][j]
    forward = []
    for i in range(n):
        forward.append((rhs[i] - sum(lower[i][j] * forward[j] for j in range(i))) / lower[i][i])
    result = [0.0] * n
    for i in reversed(range(n)):
        result[i] = (forward[i] - sum(lower[j][i] * result[j] for j in range(i + 1, n))) / lower[i][i]
    return result


class Moments:
    def __init__(self):
        n = len(FEATURES)
        self.count = 0
        self.sx, self.sy = [0.0] * n, [0.0] * 2
        self.xx = [[0.0] * n for _ in range(n)]
        self.xy = [[0.0] * 2 for _ in range(n)]

    def add(self, x, y):
        self.count += 1
        for a in range(2):
            self.sy[a] += y[a]
        for i in range(len(x)):
            self.sx[i] += x[i]
            for j in range(i + 1):
                self.xx[i][j] += x[i] * x[j]
            for a in range(2):
                self.xy[i][a] += x[i] * y[a]

    def model(self):
        if self.count < 2:
            raise ValueError('Insufficient eligible train labels at a horizon')
        n, count = len(FEATURES), self.count
        mean = [x / count for x in self.sx]
        scale = [math.sqrt(max(0, self.xx[i][i] / count - mean[i] ** 2)) or 1.0 for i in range(n)]
        center = [y / count for y in self.sy]
        gram = [[(self.xx[max(i, j)][min(i, j)] - count * mean[i] * mean[j]) / (scale[i] * scale[j])
                 + (ALPHA if i == j else 0) for j in range(n)] for i in range(n)]
        coefficients = [solve(gram, [(self.xy[i][a] - mean[i] * self.sy[a]) / scale[i] for i in range(n)]) for a in range(2)]
        return dict(count=count, mean=mean, scale=scale, intercept=center, coefficients=coefficients)


def fit(records, manifest):
    games = validate_manifest(manifest)
    moments = {h: Moments() for h in HORIZONS}
    for sample in build_samples(records, manifest, HORIZONS, 8):
        if sample['split'] != 'train' or not sample['label']['observed']:
            continue
        x, eligible, _ = inputs(sample, games[sample['gameId']])
        if eligible:
            base = baseline(sample, games[sample['gameId']], 'current')
            moments[sample['horizon']].add(x, [sample['label'][axis] - base[i] for i, axis in enumerate(('x', 'y'))])
    return dict(schemaVersion=1, features=list(FEATURES), alpha=ALPHA, horizons=list(HORIZONS),
                models={str(h): moments[h].model() for h in HORIZONS})


def predict(model, sample, game):
    x, eligible, _ = inputs(sample, game)
    if not eligible:
        return baseline(sample, game, model['selectedBaseline']), False
    m = model['models'][str(sample['horizon'])]
    z = [(v - mean) / scale for v, mean, scale in zip(x, m['mean'], m['scale'])]
    base = baseline(sample, game, 'current')
    return [base[a] + m['intercept'][a] + sum(c * v for c, v in zip(m['coefficients'][a], z)) for a in range(2)], True


def evaluate(records, manifest, model, splits):
    games = validate_manifest(manifest)
    names = BASELINES + (('ridge',) if model else ())
    cells = {g: {str(h): dict(anchors=0, observed=0, eligible=0,
             methods={n: dict(errorSum=0.0, contained=0, eligibleErrorSum=0.0, eligibleContained=0) for n in names}) for h in HORIZONS}
             for g, game in games.items() if game['split'] in splits}
    for sample in build_samples(records, manifest, HORIZONS, 8):
        if sample['split'] not in splits:
            continue
        game, cell = games[sample['gameId']], cells[sample['gameId']][str(sample['horizon'])]
        cell['anchors'] += 1
        if not sample['label']['observed']:
            continue
        cell['observed'] += 1
        _, eligible, _ = inputs(sample, game)
        cell['eligible'] += eligible
        predictions = {n: baseline(sample, game, n) for n in BASELINES}
        if model:
            predictions['ridge'], eligible = predict(model, sample, game)
        for name, point in predictions.items():
            error = math.hypot(point[0] - sample['label']['x'], point[1] - sample['label']['y'])
            cell['methods'][name]['errorSum'] += error
            cell['methods'][name]['contained'] += error <= 18
            if eligible:
                cell['methods'][name]['eligibleErrorSum'] += error
                cell['methods'][name]['eligibleContained'] += error <= 18
    for horizons in cells.values():
        for cell in horizons.values():
            n = cell['observed']
            cell['coverage'] = n / cell['anchors'] if cell['anchors'] else 0
            for metric in cell['methods'].values():
                metric['meanError'] = metric['errorSum'] / n if n else None
                metric['containment'] = metric['contained'] / n if n else None
                metric['eligibleMeanError'] = metric['eligibleErrorSum'] / cell['eligible'] if cell['eligible'] else None
                metric['eligibleContainment'] = metric['eligibleContained'] / cell['eligible'] if cell['eligible'] else None
    return cells


def average(cells, game, method, metric, horizons=PRIMARY):
    values = [cells[game][str(h)]['methods'][method][metric] for h in horizons]
    return sum(values) / len(values) if all(v is not None for v in values) else None


def select_baseline(dev):
    if not dev or any(average(dev, g, 'current', 'meanError') is None for g in dev):
        raise ValueError('Missing dev horizon labels')
    return min(BASELINES, key=lambda name: sum(average(dev, g, name, 'meanError') for g in dev) / len(dev))


def decision(cells, manifest, selected):
    withheld = [g['gameId'] for g in manifest['games'] if g['split'] == 'withheld']
    transfer = [g['gameId'] for g in manifest['games'] if g['split'] == 'transfer']
    coverage = len(withheld) == 3 and len(transfer) == 3 and all(
        cells[g][str(h)]['observed'] >= 50 for g in withheld for h in PRIMARY)
    coverage = coverage and all(cells[g][str(h)]['observed'] > 0 for g in withheld + transfer for h in HORIZONS)
    if not coverage:
        return dict(status='inconclusive', reason='Insufficient whole-game/horizon label coverage', checks={'coverage': False})
    base = [average(cells, g, selected, 'meanError') for g in withheld]
    learned = [average(cells, g, 'ridge', 'meanError') for g in withheld]
    checks = dict(coverage=True, primaryError=sum(learned) <= .90 * sum(base),
        improvedGames=sum(x < b for x, b in zip(learned, base)) >= 2,
        worstGame=all(x <= 1.10 * b for x, b in zip(learned, base)),
        shortHorizons=all(average(cells, g, 'ridge', 'meanError', (8, 16)) <= 1.10 * average(cells, g, selected, 'meanError', (8, 16)) for g in withheld),
        containment=sum(average(cells, g, 'ridge', 'containment') - average(cells, g, selected, 'containment') for g in withheld) / len(withheld) >= .02,
        transfer=all(average(cells, g, 'ridge', 'containment') >= average(cells, g, selected, 'containment') - .02 and
            average(cells, g, 'ridge', 'meanError') <= average(cells, g, selected, 'meanError') + 5 for g in transfer))
    return dict(status='offline-pass' if all(checks.values()) else 'reject', checks=checks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'events', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--stage', choices=('fit', 'evaluate'), default='fit')
    parser.add_argument('--model', type=Path)
    parser.add_argument('--model-sha256')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Output must be new; never overwrite prior evidence')
    manifest = parse_json(args.manifest.read_text(encoding='utf-8'))
    records = list(load_events(args.events))
    provenance = dict(manifestSha256=sha(args.manifest), eventsSha256=sha(args.events), toolSha256=sha(__file__))
    if args.stage == 'fit':
        model = fit(records, manifest)
        dev = evaluate(records, manifest, None, ('dev',))
        model.update(provenance)
        model['selectedBaseline'] = select_baseline(dev)
        report = dict(stage='fit', dev=dev, selectedBaseline=model['selectedBaseline'], scope='train fit and development baseline selection only')
    else:
        if not args.model or not args.model_sha256 or sha(args.model) != args.model_sha256:
            raise ValueError('Evaluation requires the previously frozen model SHA256')
        model = parse_json(args.model.read_text(encoding='utf-8'))
        if any(model[k] != v for k, v in provenance.items()) or model['features'] != list(FEATURES) or model['alpha'] != ALPHA:
            raise ValueError('Frozen model/source/input mismatch')
        cells = evaluate(records, manifest, model, ('withheld', 'transfer'))
        report = dict(stage='evaluate', cells=cells, modelSha256=args.model_sha256,
                      decision=decision(cells, manifest, model['selectedBaseline']), scope='offline observable-label mechanism only; no live strength claim')
    args.output.mkdir(parents=True)
    if args.stage == 'fit':
        (args.output / 'model.json').write_text(json.dumps(model, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        report['modelSha256'] = sha(args.output / 'model.json')
    report.update(provenance)
    (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('stage', 'modelSha256')}, indent=2))


if __name__ == '__main__':
    main()
