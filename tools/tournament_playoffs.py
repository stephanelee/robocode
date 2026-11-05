"""Fixed top-eight double-elimination bracket; no battle execution in this module."""
import hashlib
import json
from pathlib import Path

# Sources are seed numbers, winners (W), or losers (L) of prior match IDs.
WAVES = [
    [('U1', 1, 8), ('U2', 4, 5), ('U3', 2, 7), ('U4', 3, 6)],
    [('U5', 'WU1', 'WU2'), ('U6', 'WU3', 'WU4'),
     ('L1', 'LU1', 'LU2'), ('L2', 'LU3', 'LU4')],
    [('L3', 'WL1', 'LU6'), ('L4', 'WL2', 'LU5')],
    [('U7', 'WU5', 'WU6'), ('L5', 'WL3', 'WL4')],
    [('L6', 'WL5', 'LU7')],
    [('G1', 'WU7', 'WL6')],
    [('G2', 'WU7', 'WL6')],
]
RULES = 'Two 20-round games; total score, then first places; one extra 20-round game; then registered SHA256 draw of lots. Two playoff losses eliminate; G1 reset if lower finalist wins.'


def qualification(output):
    import tournament
    output = Path(output)
    reports = []
    hashes = {}
    for number in range(1, 5):
        path = output / f'swiss-{number}.json'
        report = json.loads(path.read_text())
        if report['status'] != 'complete' or report['completedGames'] != 14:
            raise ValueError('All four Swiss rounds must be complete')
        if len(report['draw']) != 7 or any(p['status'] != 'complete' for p in report['draw']):
            raise ValueError('Swiss pairing coverage incomplete')
        reports.append(report)
        hashes[path.name] = tournament.digest(path)
    rows = tournament.swiss_standings(reports)
    # An exact numeric tie requires a separately documented qualification ruling.
    for a, b in zip(rows, rows[1:]):
        if (a['points'], a['buchholz'], a['scoreShare']) == (b['points'], b['buchholz'], b['scoreShare']):
            raise ValueError('Exact Swiss ranking tie requires a recorded ruling')
    return dict(swissSHA256=hashes, seeds=[dict(seed=i, **r) for i, r in enumerate(rows[:8], 1)],
                waves=[[list(match) for match in wave] for wave in WAVES], rules=RULES, lottery='sha256(minimind-playoff-20261105|MATCH_ID|a|b) modulo 2',
                maxGames=45, maxRounds=900)


def decision(pair):
    """Resolve a valid pairing or report that its registered extra game is needed."""
    games = pair.get('games', [])
    if len(games) < 2 or any(g['status'] != 'complete' for g in games):
        return dict(winner=None, needsExtra=False, resolution='pending')
    if [g['game'] for g in games] not in ([1, 2], [1, 2, 3]):
        raise ValueError('Unexpected playoff game sequence')
    names = [pair['a'], pair['b']]
    scores, places = {}, {}
    # Match tournament display names to official descriptors (Adaptive Prime has a space).
    key = lambda n: ''.join(n.split())
    def value(game, name, field):
        found = [r for r in game['official']['results'] if key(r['name']) == key(name)]
        if len(found) != 1:
            raise ValueError('Playoff result identity mismatch')
        return found[0][field]
    for name in names:
        scores[name] = sum(value(g, name, 'totalScore') for g in games[:2])
        places[name] = sum(value(g, name, 'firstPlaces') for g in games[:2])
    for totals, rule in [(scores, 'combined-score'), (places, 'round-first-places')]:
        if totals[names[0]] != totals[names[1]]:
            if len(games) != 2:
                raise ValueError('Unnecessary extra game')
            return dict(winner=max(names, key=totals.get), needsExtra=False, resolution=rule)
    if len(games) == 2:
        return dict(winner=None, needsExtra=True, resolution='extra-game-required')
    extra = {name: value(games[2], name, 'totalScore') for name in names}
    if extra[names[0]] != extra[names[1]]:
        return dict(winner=max(names, key=extra.get), needsExtra=False, resolution='extra-game-score')
    material = f'minimind-playoff-20261105|{pair["matchId"]}|{pair["a"]}|{pair["b"]}'
    sha = hashlib.sha256(material.encode()).hexdigest()
    return dict(winner=names[int(sha, 16) % 2], needsExtra=False,
                resolution='registered-draw-of-lots', lotterySHA256=sha)


def context(output, number):
    """Validate canonical completed history and derive the next wave, without play."""
    import tournament
    if number not in range(1, 8):
        raise ValueError('Unknown playoff wave')
    output = Path(output)
    registered = qualification(output)
    tournament.freeze_manifest(output / 'playoff-registration.json', registered)
    seeds = {r['seed']: r['name'] for r in registered['seeds']}
    matches = {}
    losses = {name: 0 for name in seeds.values()}
    completed = []
    def source(ref):
        if isinstance(ref, int):
            return seeds[ref]
        pair = matches[ref[1:]]
        return pair['winner'] if ref[0] == 'W' else next(n for n in [pair['a'], pair['b']] if n != pair['winner'])
    for wave in range(1, number):
        path = output / f'playoff-{wave}.json'
        report = json.loads(path.read_text())
        expected = [(mid, source(a), source(b)) for mid, a, b in WAVES[wave-1]]
        actual = [(p['matchId'], p['a'], p['b']) for p in report['draw']]
        if actual != expected or report['status'] != 'complete':
            raise ValueError('Prior playoff wave incomplete or draw differs')
        if report['completedGames'] != sum(len(p['games']) for p in report['draw']):
            raise ValueError('Prior playoff game count differs')
        for pair in report['draw']:
            if pair['status'] != 'complete' or decision(pair)['winner'] != pair['winner']:
                raise ValueError('Prior playoff winner differs from official games')
            loser = next(n for n in [pair['a'], pair['b']] if n != pair['winner'])
            if any(losses[n] >= 2 for n in [pair['a'], pair['b']]):
                raise ValueError('Eliminated bot reentered playoff')
            losses[loser] += 1
            matches[pair['matchId']] = pair
        completed.append(report)
    if number == 7 and matches['G1']['winner'] == matches['U7']['winner']:
        raise ValueError('Grand-final reset not required; champion already decided')
    pairs = []
    for mid, a, b in WAVES[number-1]:
        a, b = source(a), source(b)
        if a == b or any(losses[n] >= 2 for n in [a, b]):
            raise ValueError('Invalid playoff participants')
        pairs.append(dict(matchId=mid, a=a, b=b, status='scheduled', games=[]))
    return pairs, dict(algorithm='Fixed seeded double-elimination bracket', seeds=registered['seeds'],
                       rules=RULES, priorLosses=losses,
                       priorStageSHA256={f'playoff-{i}.json': tournament.digest(output / f'playoff-{i}.json') for i in range(1, number)})


def standings(seeds, reports):
    rows = {s['name']: dict(name=s['name'], seed=s['seed'], played=0, wins=0, losses=0, eliminated=False) for s in seeds}
    matches = {}
    for report in reports:
        for p in report['draw']:
            if p['status'] != 'complete':
                continue
            if p.get('winner') not in (p['a'], p['b']):
                raise ValueError('Completed playoff match lacks winner')
            for name in (p['a'], p['b']):
                if rows[name]['losses'] >= 2:
                    raise ValueError('Eliminated bot reentered playoff')
                rows[name]['played'] += 1
                if name == p['winner']:
                    rows[name]['wins'] += 1
                else:
                    rows[name]['losses'] += 1
                    rows[name]['eliminated'] = rows[name]['losses'] == 2
            matches[p['matchId']] = p
    champion = None
    if 'G2' in matches:
        champion = matches['G2']['winner']
    elif 'G1' in matches and matches['G1']['winner'] == matches['U7']['winner']:
        champion = matches['G1']['winner']
    if champion and sum(not r['eliminated'] for r in rows.values()) != 1:
        raise ValueError('Champion does not leave exactly one survivor')
    return list(rows.values()), champion
