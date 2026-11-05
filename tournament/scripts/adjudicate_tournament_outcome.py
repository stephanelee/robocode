"""Apply a hash-bound, independently reviewed telemetry ruling without rerunning games.

Only handles the pinned engine's legitimate simultaneous score-win/death flags.
Original metrics, battle output, replay and failed report snapshot remain intact.
"""
import argparse
import copy
import gzip
import json
from pathlib import Path
import re
import sys

import tournament


def apply(output, source, ruling_path):
    ruling = json.loads(ruling_path.read_text())
    if not re.fullmatch(r'playoff-[1-7]', ruling['stage']) or not re.fullmatch(re.escape(ruling['stage'])+r'-validation-stop-\d+\.json', ruling['snapshotName']):
        raise ValueError('Unsafe stage or snapshot name')
    if ruling['decision'] != 'accept-original-official-score' or ruling['exception'] != 'score-winner-can-also-die':
        raise ValueError('Unknown ruling')
    path = output/(ruling['stage']+'.json')
    if tournament.digest(path) != ruling['originalReportSHA256']:
        raise ValueError('Stage changed since independent review')
    report = json.loads(path.read_text())
    if report['completedGames'] != sum(g['status']=='complete' for p in report['draw'] for g in p['games']):
        raise ValueError('Stage counters differ from validated history')
    pair = report['draw'][ruling['pair']-1]
    game = next(g for g in pair['games'] if g['game'] == ruling['game'])
    if pair['matchId'] != ruling['matchId'] or game['status'] != 'failed' or game.get('error') != 'Invalid round outcome flags':
        raise ValueError('Ruling does not match this failed game')
    if report['status'] != 'incomplete' or game['directory'] != ruling['directory']:
        raise ValueError('Unexpected stage/game state')
    run = output/game['directory']
    official_path = run/'battle.json'
    metrics_path = run/'AISurvivor-metrics.jsonl'
    if tournament.digest(official_path) != ruling['officialSHA256'] or tournament.digest(metrics_path) != ruling['metricsSHA256']:
        raise ValueError('Original game evidence changed')
    if tournament.digest(source/'tools/benchmark.py') != ruling['benchmarkSHA256']:
        raise ValueError('Frozen telemetry validator changed')
    official = json.loads(official_path.read_text())
    tournament.validate_result(official, [tuple(x) for x in ruling['identities']])
    if game.get('unvalidatedOfficial') != official:
        raise ValueError('Original retained official result differs')
    recording = tournament.collect_recording(run/'recordings', output)
    if recording['sha256'] != ruling['recordingSHA256']:
        raise ValueError('Original recording changed')
    metrics = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
    bad = [r['round'] for r in metrics if r.get('won') is True and r.get('died') is True]
    if bad != ruling['jointOutcomeRounds'] or not bad:
        raise ValueError('Joint-outcome rows differ from the reviewed ruling')
    ais = next(r['id'] for r in official['results'] if r['name']=='AISurvivor')
    starts, ends, rounds, last, deaths, previous, proof = 0, [], [], {}, set(), {}, []
    with gzip.open(output/recording['path'], 'rt') as stream:
        for line in stream:
            event = json.loads(line)
            if event['type']=='GameStartedEventForObserver':
                starts += 1
                if event['gameSetup']['numberOfRounds'] != 20 or len(event['participants']) != 2:
                    raise ValueError('Recording setup differs')
            elif event['type']=='TickEventForObserver':
                n = event['roundNumber']
                last[n] = event
                if any(e['type']=='BotDeathEvent' and e['victimId']==ais for e in event['events']):
                    deaths.add(n)
            elif event['type']=='RoundEndedEventForObserver':
                n = event['roundNumber']
                rounds.append(n)
                current = {r['id']:r for r in event['results']}
                if n in bad:
                    own = current[ais]
                    old = previous.get(ais, {'totalScore':0,'firstPlaces':0})
                    own_delta = own['totalScore']-old['totalScore']
                    other = next(r for identity,r in current.items() if identity!=ais)
                    other_delta = other['totalScore']-previous.get(other['id'],{'totalScore':0})['totalScore']
                    energy = next(b['energy'] for b in last[n]['botStates'] if b['id']==ais)
                    if n not in deaths or energy>0 or own['firstPlaces']-old['firstPlaces']!=1 or own_delta<other_delta:
                        raise ValueError('Observer does not corroborate score win plus death')
                    proof.append(dict(round=n, score=own_delta, opponentScore=other_delta, died=True, firstPlace=True))
                previous = current
            elif event['type']=='GameEndedEventForObserver':
                ends.append(event)
    if starts!=1 or len(ends)!=1 or rounds!=list(range(1,21)) or set(last)!=set(range(1,21)):
        raise ValueError('Recording game/round coverage differs')
    for row in official['results']:
        recorded = next(r for r in ends[0]['results'] if r['id']==row['id'])
        if any(recorded[k]!=v for k,v in row.items()):
            raise ValueError('Recording final result differs')
    if any(m['turns']!=last[m['round']]['turnNumber'] for m in metrics):
        raise ValueError('Telemetry terminal turn differs from observer')
    # A validation-only projection disables precisely the disputed exclusion.
    # It never changes actual metrics, learning, official scores or recordings.
    projected = copy.deepcopy(metrics)
    for row in projected:
        if row.get('won') is True and row.get('died') is True:
            row['won'] = False
    sys.path.insert(0, str(source/'tools'))
    import benchmark
    benchmark.validate_metrics(projected, 20, 'train', 20261104, 0)
    original = path.read_bytes()
    snapshot = output/(ruling['snapshotName'])
    if snapshot.exists():
        raise ValueError('Original failed-stage snapshot already exists; no repeated ruling')
    failure = [f for f in report['failures'] if f['pair']==ruling['pair'] and f['game']==ruling['game']]
    if len(failure)!=1:
        raise ValueError('Unexpected failure history')
    snapshot.write_bytes(original)
    note = dict(rulingSHA256=tournament.digest(ruling_path), decision=ruling['decision'],
                originalReportSHA256=ruling['originalReportSHA256'], originalStatus='failed',
                originalError=game['error'], observerProof=proof, reviewedAt=tournament.stamp())
    game.update(status='complete', official=official, recording=recording, adjudication=note)
    report.setdefault('resolvedFailures', []).extend([dict(**f, adjudication=note) for f in failure])
    report['failures'] = [f for f in report['failures'] if f not in failure]
    report.setdefault('adjudications', []).append(note)
    report['completedGames'] += 1
    report['completedRounds'] += 20
    # Leave the stage incomplete until explicit unplayed-only continuation.
    tournament.save(path, report)
    print(json.dumps(dict(accepted=game['directory'], originalScore=official,
                         gamesValidated=report['completedGames'], newGames=0, proof=proof),indent=2))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--ruling', type=Path, required=True)
    args = parser.parse_args()
    apply(args.output.resolve(), args.source.resolve(), args.ruling.resolve())
