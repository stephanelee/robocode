"""Single preregistered48-round actual default-TRAIN screen; never a retention decision."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess

import benchmark
import compare
import retcon
import retcon_stats

BASE = '7104b65d03298caceb010a5bfd085bda65092d17'
BASE_JAR = 'd047bf99325d2ad73aedd87eac6a4baa1677170afff0c86b948f03ea93390e1d'
PLAN = '04145348b3a41a3db9d26b823b5fd27b8c11a447'


def mechanism(rows):
    total = 0
    counts = [0] * 6
    for row in rows:
        actions = row.get('actionCounts')
        eligible, forced = row.get('duelEligibleTurns'), row.get('forcedDuelTurns')
        if (row.get('duelPolicy') != 'controller' or not isinstance(actions, list) or len(actions) != 6
                or any(type(x) is not int or x < 0 for x in actions)
                or type(eligible) is not int or type(forced) is not int or eligible < 0
                or forced != eligible or actions[0] < forced):
            raise ValueError('Default controller selection/credit mechanism telemetry is invalid')
        total += forced
        counts = [a + b for a, b in zip(counts, actions)]
    if total == 0:
        raise ValueError('No eligible forced duel decisions were observed')
    return dict(forcedDuelTurns=total, actionCounts=counts)


def decision(blocks, samples):
    previous, candidate = (compare.aggregate(blocks, arm) for arm in ('previous', 'candidate'))
    deltas = [compare.score_share(b['candidate']) - compare.score_share(b['previous']) for b in blocks]
    checks = dict(complete=len(blocks) == 3 and all(b[a]['rounds'] == 5 for b in blocks for a in ('previous', 'candidate')),
        meanGain=statistics.mean(deltas) >= .05, positiveBlocks=sum(d > 0 for d in deltas) >= 2,
        worstBlock=min(deltas) >= -.10, wins=candidate['wins'] >= previous['wins'],
        noSkippedTurns=candidate['skippedTurns'] == previous['skippedTurns'] == 0,
        wallRate=candidate['wallsPer1000Turns'] <= previous['wallsPer1000Turns'] + 10,
        samples=len(samples) == 3 and {r['opponent'] for r in samples} == {'Target', 'SpinBot', 'Walls'} and all(
            r['candidate']['rounds'] == r['previous']['rounds'] == 3 and
            compare.score_share(r['candidate']) - compare.score_share(r['previous']) >= -.05 and
            r['candidate']['metrics']['skippedTurns'] == r['previous']['metrics']['skippedTurns'] == 0 for r in samples))
    return dict(status='advance-to-confirmation' if all(checks.values()) else 'reject', checks=checks,
        previous=previous, candidate=candidate, deltas=deltas, meanGain=statistics.mean(deltas),
        limitation='48-round default TRAIN policy screen only; no retention or learning-benefit claim')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous-bot-dir', type=Path, required=True)
    parser.add_argument('--candidate-source', required=True)
    parser.add_argument('--java', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.output.exists():
        raise ValueError('New output required; preserve all prior attempts')
    if benchmark.digest(args.previous_bot_dir / 'AISurvivor.jar') != BASE_JAR:
        raise ValueError('Expected the frozen accepted710 baseline')
    args.output.mkdir(parents=True)
    plan = compare.registered_plan(benchmark.ROOT / 'docs/improvements/default-duel-policy.md', PLAN)
    (args.output / 'registered-plan.md').write_bytes(plan)
    snapshots, hashes = compare.snapshot_bots(args.previous_bot_dir.resolve(), benchmark.ROOT / 'AISurvivor', args.output)
    os.environ.pop('AISURVIVOR_DUEL_POLICY', None)  # Test real product defaults in both versions, not an override.
    report = dict(status='running', previousSource=BASE, candidateSource=args.candidate_source,
        planCommit=PLAN, policy='train', policyEnvironment='AISURVIVOR_DUEL_POLICY unset',
        runtimeOS=retcon_stats.os_provenance(), bundleSHA256=hashes, blocks=[], samples=[], validatedRounds=0)
    def save():
        (args.output / 'screen.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    save()
    try:
        cache = benchmark.ROOT / '.benchmark-cache'
        source, api_cache, provenance = retcon.prepare(cache)
        runner = benchmark.download(cache, 'robocode-tankroyale-runner-1.4.0.jar', benchmark.ASSETS['robocode-tankroyale-runner-1.4.0.jar'])
        archive = benchmark.download(cache, 'sample-bots-Java-1.4.0.zip', benchmark.ASSETS['sample-bots-Java-1.4.0.zip'])
        sample_source = args.output / 'official-samples'
        benchmark.extract(archive, sample_source)
        report.update(opponentProvenance=provenance, runnerSHA256=benchmark.digest(runner),
            runtimeJava=subprocess.run([args.java, '-version'], capture_output=True, text=True, check=True).stderr)
        def play(opponent, arm, seed, rounds, output):
            compare.verify_snapshots(snapshots, hashes)
            output.mkdir(parents=True)
            model = output / 'model.dat'
            if model.exists():
                raise ValueError('Every source/game must start without a model')
            config = argparse.Namespace(java=args.java, bot_dir=snapshots[arm], seed=seed, timeout=300,
                training_rounds=rounds, require_metrics=True, require_retcon_stats=opponent == 'Retcon',
                opponent_env={'RUMBLE_JAVA_CACHE': str(api_cache)}, retcon_source=source)
            result = benchmark.battle(config, output, runner, sample_source, opponent, [opponent], 'train', rounds, model)
            if compare.checkpoint_rounds(model) != rounds or benchmark.digest(model) != result['modelSHA256']:
                raise ValueError('Fresh own model endpoint counter/hash differs from exact scored rounds')
            if arm == 'candidate':
                rows = [json.loads(line) for line in (output / (opponent + '-train') / 'metrics.jsonl').read_text().splitlines()]
                result['policyMechanism'] = mechanism(rows)
            report['validatedRounds'] += rounds
            return result
        for index, seed in enumerate((20261701, 20261702, 20261703)):
            block = dict(seed=seed, order=['candidate', 'previous'] if index == 1 else ['previous', 'candidate'])
            report['blocks'].append(block)
            for arm in block['order']:
                block[arm] = play('Retcon', arm, seed, 5, args.output / 'retcon' / str(seed) / arm)
                save()
        for index, opponent in enumerate(('Target', 'SpinBot', 'Walls')):
            row = dict(opponent=opponent, order=['candidate', 'previous'] if index == 1 else ['previous', 'candidate'])
            report['samples'].append(row)
            for arm in row['order']:
                row[arm] = play(opponent, arm, 20261704, 3, args.output / 'samples' / opponent / arm)
                save()
        compare.verify_snapshots(snapshots, hashes)
        report.update(status='complete', decision=decision(report['blocks'], report['samples']))
        save()
        print(json.dumps(report['decision'], indent=2))
    except BaseException as error:
        report.update(status='invalid', failure={'type': type(error).__name__, 'message': str(error)})
        save()
        raise


if __name__ == '__main__':
    main()
