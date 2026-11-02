"""TRAIN/DEV-only estimated own-wave coverage; no labels, fitting or match collection."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

from legal_scan_dataset import load_events, parse_json, validated_groups, validate_manifest


def contexts(records, manifest, allowed_splits=('train', 'dev')):
    """Yield (game, round, turn, target), past-only context. No future label access."""
    if not allowed_splits or set(allowed_splits) - {'train', 'dev', 'withheld', 'transfer'}:
        raise ValueError('Explicit known context splits required')
    permitted = {g['gameId'] for g in manifest['games'] if g['split'] in allowed_splits}
    restricted = dict(manifest, games=[g for g in manifest['games'] if g['gameId'] in permitted])
    _, groups = validated_groups([r for r in records if r['gameId'] in permitted], restricted)
    rounds = {}
    for (game, number, target), turns in groups.items():
        for turn, row in turns.items():
            rounds.setdefault((game, number), {}).setdefault(turn, {})[target] = row
    for (game, number), turns in sorted(rounds.items()):
        known, delivered = {}, {}
        for turn, targets in sorted(turns.items()):
            # Own snapshots/event lists agree across targets under schema validation.
            shots = next(iter(targets.values()))['confirmedShots']
            delivered[turn] = {s['bulletId'] for s in shots}
            new = []
            for shot in shots:
                if shot['bulletId'] not in known:
                    known[shot['bulletId']] = dict(shot, firstSeenTurn=turn)
                    new.append(turn - shot['turn'])
            window_ids = set().union(*(delivered.get(t, set()) for t in range(max(1, turn - 7), turn + 1)))
            for target, row in sorted(targets.items()):
                scan = row['scan']
                context = dict(scanAge=None if scan is None else turn - scan['observedTurn'],
                    firstSeenDelays=new, estimatedWaves=[], earliestEligible=None,
                    oldWithin40=False, nearestOldETA=0.0)
                if scan is not None:
                    for shot in known.values():
                        speed = 20 - 3 * shot['power']
                        age = turn - shot['turn']
                        radius = speed * (age + 1)
                        distance = math.hypot(scan['x'] - shot['x'], scan['y'] - shot['y'])
                        gap = distance - radius
                        if gap < -18:
                            continue
                        wave = dict(bulletId=shot['bulletId'], eventAge=age,
                            firstSeenTurn=shot['firstSeenTurn'], deliveryAge=turn - shot['firstSeenTurn'],
                            missingFromFirstDelivery8=shot['firstSeenTurn'] < turn - 7,
                            missingFromRecorded8=shot['bulletId'] not in window_ids,
                            radius=radius, gap=gap, eta=gap / speed, eligible=gap > speed)
                        context['estimatedWaves'].append(wave)
                    eligible = [w for w in context['estimatedWaves'] if w['eligible']]
                    if eligible:
                        context['earliestEligible'] = min(eligible, key=lambda w: (w['eta'], w['bulletId']))
                    old_near = [w for w in context['estimatedWaves'] if w['missingFromFirstDelivery8'] and 0 <= w['eta'] <= 40]
                    if old_near:
                        context['oldWithin40'] = True
                        context['nearestOldETA'] = min(w['eta'] for w in old_near)
                yield (game, number, turn, target), context


def counts(records, manifest):
    games = {g['gameId']: dict(split=g['split'], anchors=0, missingScans=0, scanAgeBands={'0': 0, '1-2': 0, '>2': 0},
        uniqueConfirmedShots=0, firstSeenDelay={'0': 0, '1-2': 0, '>2': 0},
        anyPotentialWave=0, anyEligibleWave=0, anyEventAgeOver8=0,
        anyMissingFromFirstDelivery8=0, anyMissingFromRecorded8=0,
        earliestEligibleMissingFrom8=0, oldWithin40=0,
        oldWaveEtaBands={'overlap': 0, '0-8': 0, '8-16': 0, '16-40': 0, '>40': 0},
        visibilityDisagreements=0) for g in manifest['games'] if g['split'] in ('train', 'dev')}
    seen_turns = set()
    for key, context in contexts(records, manifest):
        game, number, turn, _ = key
        out = games[game]
        if key[:3] not in seen_turns:
            seen_turns.add(key[:3])
            for delay in context['firstSeenDelays']:
                out['uniqueConfirmedShots'] += 1
                out['firstSeenDelay']['0' if delay == 0 else '1-2' if delay <= 2 else '>2'] += 1
        if context['scanAge'] is None:
            out['missingScans'] += 1
            continue
        out['anchors'] += 1
        age = context['scanAge']
        out['scanAgeBands']['0' if age == 0 else '1-2' if age <= 2 else '>2'] += 1
        waves = context['estimatedWaves']
        out['anyPotentialWave'] += bool(waves)
        out['anyEligibleWave'] += any(w['eligible'] for w in waves)
        out['anyEventAgeOver8'] += any(w['eventAge'] > 8 for w in waves)
        out['anyMissingFromFirstDelivery8'] += any(w['missingFromFirstDelivery8'] for w in waves)
        out['anyMissingFromRecorded8'] += any(w['missingFromRecorded8'] for w in waves)
        earliest = context['earliestEligible']
        out['earliestEligibleMissingFrom8'] += bool(earliest and earliest['missingFromFirstDelivery8'])
        out['oldWithin40'] += context['oldWithin40']
        for w in waves:
            out['visibilityDisagreements'] += w['missingFromFirstDelivery8'] != w['missingFromRecorded8']
            if w['missingFromFirstDelivery8']:
                eta = w['eta']
                band = 'overlap' if eta < 0 else '0-8' if eta <= 8 else '8-16' if eta <= 16 else '16-40' if eta <= 40 else '>40'
                out['oldWaveEtaBands'][band] += 1
    return games


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'events', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Output must be new; preserve prior diagnostic')
    manifest = parse_json(args.manifest.read_text(encoding='utf-8'))
    validate_manifest(manifest)
    if args.events.suffix == '.gz':
        with gzip.open(args.events, 'rt', encoding='utf-8') as stream:
            records = [parse_json(line) for line in stream if line.strip()]
    else:
        records = list(load_events(args.events))
    result = dict(schemaVersion=1, scope='TRAIN/DEV ONLY; estimated radial context, not active bullets or Retcon internal waves',
        toolSha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        manifestSha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        eventsFileSha256=hashlib.sha256(args.events.read_bytes()).hexdigest(),
        games=counts(records, manifest))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
