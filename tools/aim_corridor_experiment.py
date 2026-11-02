"""Execute a preregistered aim-corridor experiment (default plan: docs/improvements/aim-corridor.md).

Stage A: 120-round tactical fixed minimum-risk Retcon primary (compare.py).
Stage B: official Target/SpinBot/Walls regression, 10 fixed rounds per arm each (always run).
Stage C: 120-round default online-training veto (compare.py), only when Stage A retains.
Retain only if A retains, B passes and C retains. Retcon native statistics are offline evidence only.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import benchmark
import compare
import retcon_stats

PLAN_PATH = 'docs/improvements/aim-corridor.md'
SEEDS = '20261201,20261202,20261203'
SAMPLE_SEED = 20261204
SAMPLE_ROUNDS = 10
SAMPLE_MAX_SHARE_LOSS = 0.05


def retcon_mechanism(comparison):
    """Aggregate Retcon's own actual/virtual gun counters per arm (offline diagnostic, never a gate)."""
    totals = {}
    for arm in ('previous', 'candidate'):
        row = dict(actualShots=0, actualHits=0, virtual={}, damageTakenByAIS=0.0, aisTurns=0)
        for block in comparison['blocks']:
            result = block[arm]
            stats = result.get('retconStats') or {}
            gun = stats.get('actualGun') or {}
            row['actualShots'] += gun.get('shots', 0)
            row['actualHits'] += gun.get('hits', 0)
            for model in stats.get('virtualGuns', []):
                entry = row['virtual'].setdefault(model['model'], dict(hits=0, shots=0))
                entry['hits'] += model['hits']
                entry['shots'] += model['shots']
            row['damageTakenByAIS'] += result['metrics'].get('damageTaken', 0)
            row['aisTurns'] += result['metrics'].get('turns', 0)
        row['actualHitRate'] = row['actualHits'] / row['actualShots'] if row['actualShots'] else None
        for entry in row['virtual'].values():
            entry['hitRate'] = entry['hits'] / entry['shots'] if entry['shots'] else None
        totals[arm] = row
    return totals


def samples_pass(samples):
    return len(samples) == 3 and all(
        row['candidate']['rounds'] == row['previous']['rounds'] == SAMPLE_ROUNDS and
        compare.score_share(row['candidate']) - compare.score_share(row['previous']) >= -SAMPLE_MAX_SHARE_LOSS - 1e-12 and
        row['candidate']['metrics']['skippedTurns'] == 0 for row in samples)


def final_decision(primary, samples, veto):
    checks = dict(primaryRetain=primary['decision']['status'] == 'retain', samples=samples_pass(samples),
                  onlineVetoRetain=veto is not None and veto['decision']['status'] == 'retain')
    return dict(status='retain' if all(checks.values()) else 'reject', checks=checks,
                onlineVetoRun=veto is not None,
                limitation='Opponent-specific Retcon result plus three sample regressions; not general strength')


def run_compare(args, policy, category, output):
    plan_path = getattr(args, 'plan_path', PLAN_PATH)
    seeds = getattr(args, 'seeds', SEEDS)
    subprocess.run([sys.executable, str(Path(__file__).with_name('compare.py')),
        '--previous-bot-dir', str(args.previous_bot_dir), '--candidate-bot-dir', str(benchmark.ROOT / 'AISurvivor'),
        '--previous-source', args.previous_source, '--candidate-source', args.candidate_source,
        '--plan', str(benchmark.ROOT / plan_path), '--plan-commit', args.plan_commit,
        '--purpose', 'primary', '--policy', policy, '--category', category, '--rounds', '20',
        '--seeds', seeds, '--java', args.java, '--timeout', '900', '--require-retcon-stats',
        '--output', str(output)], check=True)
    return json.loads((output / 'comparison.json').read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous-bot-dir', type=Path, required=True)
    parser.add_argument('--previous-source', required=True)
    parser.add_argument('--expected-previous-jar', required=True)
    parser.add_argument('--candidate-source', required=True)
    parser.add_argument('--plan-commit', required=True)
    parser.add_argument('--java', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plan-path', default=PLAN_PATH, help='Registered plan path inside the repository')
    parser.add_argument('--seeds', default=SEEDS, help='Three comma-separated Retcon block seeds')
    parser.add_argument('--sample-seed', type=int, default=SAMPLE_SEED)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.previous_bot_dir = args.previous_bot_dir.resolve()
    if args.output.exists():
        raise ValueError('Preserve prior evidence; output must be new')
    if benchmark.digest(args.previous_bot_dir / 'AISurvivor.jar') != args.expected_previous_jar:
        raise ValueError('Baseline JAR does not match the accepted source')
    args.output.mkdir(parents=True)
    report = dict(status='running', runtimeOS=retcon_stats.os_provenance(), samples=[], plan=args.plan_path, seeds=args.seeds, sampleSeed=args.sample_seed,
                  planCommit=args.plan_commit, previousSource=args.previous_source,
                  candidateSource=args.candidate_source, candidateJarSHA256=benchmark.digest(benchmark.ROOT / 'AISurvivor/AISurvivor.jar'),
                  previousJarSHA256=args.expected_previous_jar)
    destination = args.output / 'experiment.json'

    def save():
        destination.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    save()
    try:
        primary = run_compare(args, 'fixed-minrisk', 'tactical', args.output / 'primary')
        report['primary'] = dict(decision=primary['decision'], mechanism=retcon_mechanism(primary))
        save()
        cache = benchmark.ROOT / '.benchmark-cache'
        runner = benchmark.download(cache, 'robocode-tankroyale-runner-1.4.0.jar', benchmark.ASSETS['robocode-tankroyale-runner-1.4.0.jar'])
        archive = benchmark.download(cache, 'sample-bots-Java-1.4.0.zip', benchmark.ASSETS['sample-bots-Java-1.4.0.zip'])
        samples = args.output / 'official-samples'
        benchmark.extract(archive, samples)
        snapshots = {arm: args.output / 'primary/artifacts' / arm for arm in ('previous', 'candidate')}
        for index, opponent in enumerate(('Target', 'SpinBot', 'Walls')):
            compare.verify_snapshots(snapshots, primary['bundleSHA256'])
            row = dict(opponent=opponent, order=['candidate', 'previous'] if index == 1 else ['previous', 'candidate'])
            report['samples'].append(row)
            for arm in row['order']:
                output = args.output / 'samples' / opponent / arm
                output.mkdir(parents=True)
                config = argparse.Namespace(java=args.java, bot_dir=snapshots[arm], seed=args.sample_seed,
                    opponent_env={}, retcon_source=None, timeout=300, training_rounds=20, require_metrics=True)
                row[arm] = benchmark.battle(config, output, runner, samples, opponent, [opponent],
                    'fixed-minrisk', SAMPLE_ROUNDS, output / 'absent-model.dat')
                row[arm]['scoreShare'] = compare.score_share(row[arm])
                save()
        compare.verify_snapshots(snapshots, primary['bundleSHA256'])
        veto = None
        if primary['decision']['status'] == 'retain':
            veto = run_compare(args, 'online-train', 'correctness', args.output / 'online-veto')
            report['onlineVeto'] = dict(decision=veto['decision'], mechanism=retcon_mechanism(veto))
        report.update(status='complete', decision=final_decision(primary, report['samples'], veto))
        save()
        print(json.dumps(report['decision'], indent=2))
    except BaseException as error:
        report.update(status='invalid', failure=dict(type=type(error).__name__, message=str(error)))
        save()
        raise


if __name__ == '__main__':
    main()
