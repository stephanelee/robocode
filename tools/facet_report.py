#!/usr/bin/env python3
"""Analyze frozen Facet results only; never launches bots or trains a model."""
import argparse
import json
from pathlib import Path


def main(args):
    data=json.loads(args.results.read_text())
    games=data.get('games',[])
    completed=[g for g in games if g.get('status')=='complete']
    lines=['# Facet registered duel benchmark results','',
           'Five cold20-round games per opponent were registered (300rounds total). Rounds within each game share online learning and are dependent. This is descriptive measurement, not confirmation or a Diamond-equivalence claim.','',
           f"Acquisition status: **{data['status']}**; completed games **{len(completed)}/15**, official completed rounds **{sum(g['official']['numberOfRounds'] for g in completed)}/300**. No replacement, retuning or automatic confirmation.",'',
           '| Opponent | Cold games / rounds | Facet official first places | Equal-game mean score share | Registered disposition |',
           '| --- | ---: | ---: | ---: | --- |']
    for name,row in data.get('summary',{}).items():
        decision=row['registeredGateAssessment']['decision']
        lines.append(f"| {name} | {row['coldGames']} / {row['rounds']} | {row['officialFirstPlaces']} | {row['equalGameMeanScoreShare']:.2%} | {decision} |")
    if data.get('error'):lines += ['', 'Acquisition error: `'+data['error'].replace('`',"'")+'`. Incomplete scheduled games are not replacement candidates.']
    lines += ['', '## Individual games','',
              '| Opponent / game | Seed | Facet / opponent score | Facet first places | Share |',
              '| --- | ---: | ---: | ---: | ---: |']
    for g in completed:
        other=next(r for r in g['official']['results'] if r['name']!='Facet')
        lines.append(f"| {g['opponent']} / {g['game']} | {g['seed']} | {g['facet']['totalScore']} / {other['totalScore']} | {g['facet']['firstPlaces']} | {g['scoreShare']:.2%} |")
    lines += ['', '## Facet telemetry','',
              'Physical confirmed counts and hit callbacks belong to Facet. Raw event damage can exceed official capped bullet-damage scoring due overkill. Turn denominator is decisions plus reported skipped-turn events. Incoming opponent private telemetry is not collected.','',
              '| Opponent | Confirmed shots / hits | Raw bullet damage given / taken | Wall / bot collisions | Skips / bot turns | Mean range | Mean scan age / max | Decision p99 bucket upper bounds by game |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
    for name,row in data.get('summary',{}).items():
        metrics=row['metricsByGame']
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
        def observed_sum(key):return sum(r['observationsCountSumMinMax'].get(key,[0,0])[1] for r in raw)
        max_steps=max((r['observationsCountSumMinMax'].get('predictorSteps',[0,0,0,0])[3] for r in raw),default=0)
        lines.append(f"| {name} | {sum(r['shotsRequested'] for r in raw)} / {sum(r['shotsAccepted'] for r in raw)} / {sum(r['shotsConfirmed'] for r in raw)} | {observed_sum('energyFireCost'):.2f} / {observed_sum('energyHitBonus'):.2f} | {sum(r['gcCount'] for r in raw)} / {sum(r['gcMillis'] for r in raw)} | {max_steps:.0f} | {sum(r['surfTruncations'] for r in raw)} / {sum(r['geometryFallbacks'] for r in raw)} | {sum(r['unmatchedConfirmations'] for r in raw)} |")
    lines += ['', 'All raw counters, inference ambiguity, emergency/budget/shadow fallbacks, payload gauges and model hashes remain in results.json and original metrics. A missing required raw field is not a zero passing measurement. Above predictor maxima are observed turn maxima; GC totals are per fresh process.']
    curves=[]
    for g in completed:
        metrics=args.results.parent/f"{g['opponent']}-g{g['game']:02d}"/'FACET_metrics.jsonl'
        if not metrics.is_file():raise ValueError('Missing raw Facet metrics: '+str(metrics))
        rows=[json.loads(x) for x in metrics.read_text().splitlines()]
        previous_shots=previous_hits=previous_turns=0
        for row in rows:
            shots=row['shotsConfirmed'];hits=row['counters'].get('shot.hit',0);turns=row['counters'].get('turn.decisions',0)
            curves.append(dict(opponent=g['opponent'],game=g['game'],round=row['round'],shots=shots-previous_shots,hits=hits-previous_hits,decisions=turns-previous_turns))
            previous_shots,previous_hits,previous_turns=shots,hits,turns
    lines += ['', '## Limits and promotion','',
              'Latency values are powers-of-two histogram bucket upper bounds, not exact nanosecond quantiles. Raw metrics retain every subsystem histogram, GC/memory/work/fallback counter and whole-game extrema. Round-index shot/hit deltas are written to learning-by-round.json; they describe adaptation/co-adaptation and do not establish causal learning improvement. Round extrema cannot be obtained by differencing cumulative extrema.','',
              'Five game clusters per opponent limit precision; engine seats/seeds/initial positions were uncontrolled, with only launch order alternated. AIS results have no invented strength threshold. Adaptive/sample transfer, competitive melee and the declared unacquired skip/rejected/multicontact contracts still prevent broad promotion claims. Passing a requested cohort establishes only that cohort screen; failures remain evidence. No automatic fresh confirmation or tuning.','',
              'Exact registration, source/JAR/config/opponent/runtime identities, full per-game official results, private Facet metrics/traces/model hashes and failed preflight evidence accompany the report. Independent postmeasurement review and required CI are separate from implementation.']
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'results.md').write_text('\n'.join(lines)+'\n')
    (args.output/'learning-by-round.json').write_text(json.dumps(curves,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
