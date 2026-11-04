#!/usr/bin/env python3
"""Analyze frozen Facet results only; never launches bots or trains a model."""
import argparse
import json
from pathlib import Path
import statistics
import math
from facet_benchmark import COHORTS, metric_summary


def validate_metric_shape(row):
    """Validate every field consumed by this renderer before using numeric evidence."""
    scalar_keys = ('shotsConfirmed', 'shotsAccepted', 'shotsRequested', 'neighborBytes',
                   'gcCount', 'gcMillis', 'surfTruncations', 'geometryFallbacks', 'unmatchedConfirmations')
    for key in scalar_keys:
        if type(row[key]) is not int or row[key] < 0:
            raise ValueError('invalid own numeric scalar: ' + key)
    counters = row['counters']
    if not isinstance(counters, dict) or any(type(v) is not int or v < 0 for v in counters.values()):
        raise ValueError('invalid own counters')
    observations = row['observationsCountSumMinMax']
    if not isinstance(observations, dict):
        raise ValueError('invalid own observations')
    for value in observations.values():
        if not isinstance(value, list) or len(value) != 4 or any(type(v) not in (int, float) for v in value) or value[0] < 0 or int(value[0]) != value[0] or value[2] > value[3]:
            raise ValueError('invalid own count/sum/min/max observation')
    timings = row['timingLog2Nanos']
    if not isinstance(timings, dict) or any(not isinstance(v, list) or not 1 <= len(v) <= 64 or any(type(n) is not int or n < 0 for n in v) for v in timings.values()):
        raise ValueError('invalid own latency histogram')


def read_own_metrics(path):
    rows, usable, errors = [], [], []
    if not path.is_file():
        return rows, usable, [dict(error='missing own metrics file')]
    previous = 0
    last_counts = (0, 0, 0)
    def finite(value):
        if isinstance(value, dict):
            for v in value.values(): finite(v)
        elif isinstance(value, list):
            for v in value: finite(v)
        elif isinstance(value, (int, float)) and not isinstance(value, bool) and not math.isfinite(value):
            raise ValueError('nonfinite own numeric value')
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        try:
            row = json.loads(line)
            finite(row)
            if not isinstance(row, dict):
                raise ValueError('own metric row must be an object')
            rows.append(row)
            round_id = row.get('round')
            if type(round_id) is not int or not previous < round_id <= 20:
                raise ValueError('duplicate/out-of-order/out-of-range own round')
            previous = round_id
            if row.get('schema') == 3 and any(row.get(k) != 'success' for k in ('snapshotStatus', 'terminalObservation', 'terminalHash')):
                raise ValueError('explicit failed/quarantined terminal evidence')
            if row.get('schema') not in (2, 3):
                raise ValueError('unsupported own telemetry schema')
            validate_metric_shape(row)
            counts = (row['shotsConfirmed'], row['counters'].get('shot.hit', 0), row['counters'].get('turn.decisions', 0))
            if any(type(v) is not int or v < old for v, old in zip(counts, last_counts)):
                raise ValueError('invalid/decreasing cumulative own counts')
            if any(type(row[k]) is not int for k in ('shotsConfirmed', 'shotsAccepted', 'shotsRequested')) or not 0 <= row['shotsConfirmed'] <= row['shotsAccepted'] <= row['shotsRequested']:
                raise ValueError('invalid own physical-shot transaction')
            metric_summary(row)  # Missing numeric fields are unusable, never zero evidence.
            usable.append(row)
            last_counts = counts
        except (ValueError, KeyError, TypeError, AttributeError, ZeroDivisionError, OverflowError, RecursionError) as error:
            errors.append(dict(line=line_number, error=str(error)))
    return rows, usable, errors


def load_evidence(path):
    """Read public official results and own numeric metrics; never repair originals."""
    data = json.loads(path.read_text())
    recorded = []
    for original in data.get('games', []):
        g = dict(original)
        if g['opponent'] not in COHORTS or type(g['game']) is not int or not 1 <= g['game'] <= 5:
            raise ValueError('Unregistered game path')
        directory = path.parent / f"{g['opponent']}-g{g['game']:02d}"
        official_path = directory / 'battle.json'
        official = g.get('official')
        if official_path.is_file():
            raw = json.loads(official_path.read_text())
            if official is not None and official != raw:
                raise ValueError('Stored and raw official results differ')
            official = raw
        if official is None:
            continue
        if official['numberOfRounds'] != 20 or len(official['results']) != 2 or sum(r['name'] == 'Facet' for r in official['results']) != 1:
            raise ValueError('Official result shape differs')
        facet = next(r for r in official['results'] if r['name'] == 'Facet')
        total = sum(r['totalScore'] for r in official['results'])
        if total <= 0:
            raise ValueError('Invalid official score denominator')
        metrics = directory / 'FACET_metrics.jsonl'
        rows, usable, errors = read_own_metrics(metrics)
        identities = {r.get('round') for r in rows if type(r.get('round')) is int and 1 <= r['round'] <= 20}
        g.update(official=official, facet=facet, scoreShare=facet['totalScore']/total,
                 metricRows=rows, usableMetricRows=usable, metricErrors=errors,
                 missingMetricRounds=sorted(set(range(1, 21))-identities),
                 unusableMetricRounds=sorted(identities-{r['round'] for r in usable}))
        recorded.append(g)
    # A descriptive partial cohort never inherits a passing registered disposition.
    summaries = {}
    for opponent in COHORTS:
        games = [g for g in recorded if g['opponent'] == opponent]
        if not games:
            continue
        raw = [g['usableMetricRows'][-1] for g in games if g['usableMetricRows']]
        original = data.get('summary', {}).get(opponent)
        accepted = all(g.get('status') == 'complete' and not g['metricErrors'] and not g['missingMetricRounds'] and not g['unusableMetricRounds'] for g in games)
        disposition = original['registeredGateAssessment']['decision'] if data['status'] == 'complete' and original and accepted else 'invalid-or-incomplete-descriptive-only'
        summaries[opponent] = dict(coldGames=len(games), rounds=sum(g['official']['numberOfRounds'] for g in games),
            officialFirstPlaces=sum(g['facet']['firstPlaces'] for g in games),
            equalGameMeanScoreShare=statistics.mean(g['scoreShare'] for g in games),
            metricsByGame=[metric_summary(r) for r in raw], finalCumulativeTelemetryByGame=raw,
            registeredGateAssessment=dict(decision=disposition))
    data['summary'] = summaries
    return data, recorded


def learning_intervals(games):
    curves = []
    for g in games:
        previous_round = previous_shots = previous_hits = previous_turns = 0
        for row in g.get('usableMetricRows', g['metricRows']):
            shots = row['shotsConfirmed']; hits = row['counters'].get('shot.hit', 0); turns = row['counters'].get('turn.decisions', 0)
            delta = (shots-previous_shots, hits-previous_hits, turns-previous_turns)
            if min(delta) < 0:
                raise ValueError('Cumulative learning counters decreased')
            interval = dict(opponent=g['opponent'], game=g['game'], fromRound=previous_round+1,
                            throughRound=row['round'], shots=delta[0], hits=delta[1], decisions=delta[2])
            if row['round'] == previous_round+1:
                interval['round'] = row['round']
            curves.append(interval)
            previous_round, previous_shots, previous_hits, previous_turns = row['round'], shots, hits, turns
    return curves


def main(args):
    data, completed=load_evidence(args.results)
    games=data.get('games',[])
    scheduled_games = data.get('scheduledGames', 15)
    scheduled_rounds = data.get('scheduledRounds', 300)
    lines=['# Facet registered duel benchmark results','',
           f'Five cold20-round games per registered opponent ({scheduled_rounds}rounds total). Rounds within each game share online learning and are dependent. This is descriptive measurement, not confirmation or a Diamond-equivalence claim.','',
           f"Acquisition status: **{data['status']}**; accepted games **{sum(g.get('status') == 'complete' for g in games)}/{scheduled_games}**, officially finished games **{len(completed)}/{scheduled_games}**, official completed rounds **{sum(g['official']['numberOfRounds'] for g in completed)}/{scheduled_rounds}**. Invalid games and favorable outcomes remain included. No replacement, retuning or automatic confirmation.",'',
           '| Opponent | Cold games / rounds | Facet official first places | Equal-game mean score share | Registered disposition |',
           '| --- | ---: | ---: | ---: | --- |']
    for name,row in data.get('summary',{}).items():
        decision=row['registeredGateAssessment']['decision']
        lines.append(f"| {name} | {row['coldGames']} / {row['rounds']} | {row['officialFirstPlaces']} | {row['equalGameMeanScoreShare']:.2%} | {decision} |")
    if data.get('error'):lines += ['', 'Acquisition error: `'+data['error'].replace('`',"'")+'`. Incomplete scheduled games are not replacement candidates.']
    lines += ['', '## Individual games','',
              '| Opponent / game | Seed | Facet / opponent score | Facet first places | Share | Acquisition status | Missing / unusable metric rounds |',
              '| --- | ---: | ---: | ---: | ---: | --- | --- |']
    for g in completed:
        other=next(r for r in g['official']['results'] if r['name']!='Facet')
        lines.append(f"| {g['opponent']} / {g['game']} | {g['seed']} | {g['facet']['totalScore']} / {other['totalScore']} | {g['facet']['firstPlaces']} | {g['scoreShare']:.2%} | {g.get('status', 'unknown')} (screen {data['status']}) | {g['missingMetricRounds']} / {g['unusableMetricRounds']} |")
    lines += ['', 'Own telemetry errors and last usable cumulative round (partial values are not full-game totals):']
    for g in completed:
        last = g['usableMetricRows'][-1]['round'] if g['usableMetricRows'] else None
        lines.append(f"- {g['opponent']}/g{g['game']}: last usable round={last}; errors={json.dumps(g['metricErrors'])}")
    lines += ['', '## Facet telemetry','',
              'Physical confirmed counts and hit callbacks belong to Facet. Raw event damage can exceed official capped bullet-damage scoring due overkill. Turn denominator is decisions plus reported skipped-turn events. Incoming opponent private telemetry is not collected.','',
              '| Opponent | Confirmed shots / hits | Raw bullet damage given / taken | Wall / bot collisions | Skips / bot turns | Mean range | Mean scan age / max | Decision p99 bucket upper bounds by game |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
    for name,row in data.get('summary',{}).items():
        metrics=row['metricsByGame']
        if not metrics:
            lines.append(f'| {name} | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |')
            continue
        def obs(key):
            records=[r[key] for r in metrics if r.get(key)]
            count=sum(x['count'] for x in records)
            return sum(x['sum'] for x in records),count,max((x['max'] for x in records),default=0)
        given,_,_=obs('damageGiven');taken,_,_=obs('damageTaken');distance,n,_=obs('range');age,n_age,max_age=obs('scanAge')
        p99=[r['latencyNanosBucketUpper'].get('decision',{}).get('p99') for r in metrics]
        range_text=f'{distance/n:.2f}' if n else 'unavailable'
        age_text=f'{age/n_age:.3f} / {max_age:.0f}' if n_age else 'unavailable'
        lines.append(f"| {name} | {sum(r['confirmed'] for r in metrics)} / {sum(r['physicalHits'] for r in metrics)} | {given:.2f} / {taken:.2f} | {sum(r['wallCollisions'] for r in metrics)} / {sum(r['botCollisions'] for r in metrics)} | {sum(r['reportedSkippedTurns'] for r in metrics)} / {sum(r['botTurnDenominator'] for r in metrics)} | {range_text} | {age_text} | {p99} ns |")
    lines += ['', '## Work, energy and fallback evidence', '',
              '| Opponent | Requested / accepted / confirmed | Fire energy / hit bonus | GC count / milliseconds | Maximum recorded predictor steps | Surf truncations / geometry fallbacks | Unmatched confirmations |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name,row in data.get('summary',{}).items():
        raw=row['finalCumulativeTelemetryByGame']
        if not raw:
            lines.append(f'| {name} | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |')
            continue
        def observed_sum(key):return sum(r['observationsCountSumMinMax'].get(key,[0,0])[1] for r in raw)
        max_steps=max((r['observationsCountSumMinMax'].get('predictorSteps',[0,0,0,0])[3] for r in raw),default=0)
        lines.append(f"| {name} | {sum(r['shotsRequested'] for r in raw)} / {sum(r['shotsAccepted'] for r in raw)} / {sum(r['shotsConfirmed'] for r in raw)} | {observed_sum('energyFireCost'):.2f} / {observed_sum('energyHitBonus'):.2f} | {sum(r['gcCount'] for r in raw)} / {sum(r['gcMillis'] for r in raw)} | {max_steps:.0f} | {sum(r['surfTruncations'] for r in raw)} / {sum(r['geometryFallbacks'] for r in raw)} | {sum(r['unmatchedConfirmations'] for r in raw)} |")
    lines += ['', 'All raw counters, inference ambiguity, emergency/budget/shadow fallbacks, payload gauges and model hashes remain in results.json and original metrics. A missing required raw field is not a zero passing measurement. Above predictor maxima are observed turn maxima; GC totals are per fresh process.']
    curves=learning_intervals(completed)
    lines += ['', '## Limits and promotion','',
              'Latency values are powers-of-two histogram bucket upper bounds, not exact nanosecond quantiles. Raw metrics retain every subsystem histogram, GC/memory/work/fallback counter and whole-game extrema. Observed-interval shot/hit deltas are written to learning-by-round.json; gaps use fromRound/throughRound and have no invented individual round label. Missing raw metrics remain missing evidence, not zero counts. These deltas describe adaptation/co-adaptation and do not establish causal learning improvement. Round extrema cannot be obtained by differencing cumulative extrema.','',
              'Five game clusters per opponent limit precision; engine seats/seeds/initial positions were uncontrolled, with only launch order alternated. AIS results have no invented strength threshold. Adaptive/sample transfer, competitive melee and the declared unacquired skip/rejected/multicontact contracts still prevent broad promotion claims. Passing a requested cohort establishes only that cohort screen; failures remain evidence. No automatic fresh confirmation or tuning.','',
              'Exact registration, source/JAR/config/opponent/runtime identities, full per-game official results, private Facet metrics/traces/model hashes and failed preflight evidence accompany the report. Independent postmeasurement review and required CI are separate from implementation.']
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'results.md').write_text('\n'.join(lines)+'\n')
    (args.output/'learning-by-round.json').write_text(json.dumps(curves,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
