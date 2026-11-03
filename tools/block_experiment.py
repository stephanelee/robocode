"""Calibrated multi-block candidate experiment (first used by docs/improvements/duel-surf.md).

Stage A: N blocks x R fixed minimum-risk Retcon rounds per arm.
Stage B: official Target/SpinBot/Walls regression (always run).
Stage C: N blocks x R default online-training Retcon rounds per arm with fresh models (only if A passes).
The budget comes from the A/A calibration (docs/aa-calibration.md, PR #37). Retcon native statistics are
offline evidence only.
"""
import argparse
import json
import math
import re
import statistics
import subprocess
from pathlib import Path

import benchmark
import compare
import retcon
import retcon_stats

Z = 1.6448536269514722  # one-sided 95%
SAMPLE_ROUNDS = 10


def block_stats(blocks):
    deltas = [b['candidate']['scoreShare'] - b['previous']['scoreShare'] for b in blocks]
    mean = statistics.mean(deltas)
    se = statistics.stdev(deltas) / math.sqrt(len(deltas))
    totals = {}
    for arm in ('previous', 'candidate'):
        turns = sum(b[arm]['metrics']['turns'] for b in blocks)
        walls = sum(b[arm]['metrics']['wallCollisions'] for b in blocks)
        totals[arm] = dict(meanShare=statistics.mean(b[arm]['scoreShare'] for b in blocks),
                           wins=sum(b[arm]['firstPlaces'] for b in blocks), turns=turns, walls=walls,
                           wallsPer1000Turns=1000 * walls / turns,
                           skippedTurns=sum(b[arm]['metrics']['skippedTurns'] for b in blocks),
                           damageTaken=sum(b[arm]['metrics']['damageTaken'] for b in blocks),
                           damageDealt=sum(b[arm]['metrics']['damageDealt'] for b in blocks),
                           shots=sum(b[arm]['metrics']['shotsFired'] for b in blocks),
                           hits=sum(b[arm]['metrics']['bulletHits'] for b in blocks))
    return dict(deltas=deltas, meanDelta=mean, deltaSE=se, lowerBound95=mean - Z * se, arms=totals)


def primary_decision(blocks, blocks_required):
    s = block_stats(blocks)
    prev, cand = s['arms']['previous'], s['arms']['candidate']
    checks = dict(complete=len(blocks) == blocks_required,
                  meanGain=s['meanDelta'] >= 0.05 - 1e-12,
                  lowerBoundPositive=s['lowerBound95'] > 0,
                  winsNotLower=cand['wins'] >= prev['wins'],
                  noSkippedTurns=cand['skippedTurns'] == 0,
                  wallRate=cand['wallsPer1000Turns'] <= prev['wallsPer1000Turns'] + 10 + 1e-12)
    return dict(status='pass' if all(checks.values()) else 'fail', checks=checks, **s)


def online_decision(blocks, blocks_required):
    s = block_stats(blocks)
    checks = dict(complete=len(blocks) == blocks_required,
                  nonInferior=s['lowerBound95'] >= -0.05 - 1e-12,
                  noSkippedTurns=s['arms']['candidate']['skippedTurns'] == 0)
    return dict(status='pass' if all(checks.values()) else 'fail', checks=checks, **s)


def samples_pass(samples):
    return len(samples) == 3 and all(
        row['candidate']['rounds'] == row['previous']['rounds'] == SAMPLE_ROUNDS and
        row['candidate']['scoreShare'] - row['previous']['scoreShare'] >= -0.05 - 1e-12 and
        row['candidate']['metrics']['skippedTurns'] == 0 for row in samples)


def final_decision(primary, samples, online):
    checks = dict(primary=primary['status'] == 'pass', samples=samples_pass(samples),
                  online=online is not None and online['status'] == 'pass')
    return dict(status='retain' if all(checks.values()) else 'reject', checks=checks, onlineRun=online is not None)


def retcon_mechanism(blocks):
    result = {}
    for arm in ('previous', 'candidate'):
        row = dict(actualHits=0, actualShots=0, virtual={})
        for b in blocks:
            stats = b[arm].get('retconStats') or {}
            gun = stats.get('actualGun') or {}
            row['actualHits'] += gun.get('hits', 0)
            row['actualShots'] += gun.get('shots', 0)
            for model in stats.get('virtualGuns', []):
                entry = row['virtual'].setdefault(model['model'], dict(hits=0, shots=0))
                entry['hits'] += model['hits']
                entry['shots'] += model['shots']
        row['actualHitRate'] = row['actualHits'] / row['actualShots'] if row['actualShots'] else None
        result[arm] = row
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous-bot-dir', type=Path, required=True)
    parser.add_argument('--expected-previous-jar', required=True)
    parser.add_argument('--previous-source', required=True)
    parser.add_argument('--candidate-source', required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-commit', required=True)
    parser.add_argument('--blocks', type=int, required=True)
    parser.add_argument('--rounds', type=int, default=20)
    parser.add_argument('--seed-base', type=int, required=True)
    parser.add_argument('--sample-seed', type=int, required=True)
    parser.add_argument('--java', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise ValueError('Preserve prior evidence; output must be new')
    previous_dir = args.previous_bot_dir.resolve()
    if benchmark.digest(previous_dir / 'AISurvivor.jar') != args.expected_previous_jar:
        raise ValueError('Baseline JAR does not match the accepted source')
    plan = compare.registered_plan(args.plan, args.plan_commit)
    output.mkdir(parents=True)
    (output / 'preregistered-plan.md').write_bytes(plan)
    snapshots, hashes = compare.snapshot_bots(previous_dir, (benchmark.ROOT / 'AISurvivor').resolve(), output)
    report = dict(schemaVersion=1, status='running', plan=str(args.plan), planCommit=args.plan_commit,
                  previousSource=args.previous_source, candidateSource=args.candidate_source, bundleSHA256=hashes,
                  blocksPerArm=args.blocks, roundsPerBlock=args.rounds, seedBase=args.seed_base,
                  sampleSeed=args.sample_seed, runtimeOS=retcon_stats.os_provenance(),
                  primaryBlocks=[], samples=[], onlineBlocks=[])
    destination = output / 'experiment.json'

    def save():
        destination.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    save()
    try:
        runtime = subprocess.run([args.java, '-XshowSettings:properties', '-version'], capture_output=True,
                                 text=True, timeout=15, check=True)
        if not re.search(r'java\.specification\.version\s*=\s*(2[2-9]|[3-9]\d)', runtime.stderr):
            raise ValueError('Retcon source launcher requires JDK22+')
        report['runtimeJava'] = (runtime.stderr + runtime.stdout).strip()
        cache = (benchmark.ROOT / '.benchmark-cache').resolve()
        source, api_cache, provenance = retcon.prepare(cache)
        runner = benchmark.download(cache, 'robocode-tankroyale-runner-1.4.0.jar',
                                    benchmark.ASSETS['robocode-tankroyale-runner-1.4.0.jar'])
        report.update(opponentProvenance=[provenance], runnerSHA256=benchmark.digest(runner))

        def retcon_stage(name, policy, key):
            for index in range(args.blocks):
                seed = args.seed_base + index
                order = ['previous', 'candidate'] if index % 2 == 0 else ['candidate', 'previous']
                block = dict(seed=seed, order=order)
                for arm in order:
                    compare.verify_snapshots(snapshots, hashes)
                    arm_output = output / name / f'block-{index + 1:02d}' / arm
                    arm_output.mkdir(parents=True)
                    config = argparse.Namespace(java=args.java, bot_dir=snapshots[arm], seed=seed, timeout=900,
                        training_rounds=args.rounds, require_metrics=True, require_retcon_stats=True,
                        opponent_env={'RUMBLE_JAVA_CACHE': str(api_cache)}, retcon_source=source)
                    result = benchmark.battle(config, arm_output, runner, output, 'Retcon', ['Retcon'], policy,
                                              args.rounds, arm_output / 'model.dat')
                    if result['rounds'] != args.rounds or result['metrics']['roundsRecorded'] != args.rounds:
                        raise ValueError('Validated battle/telemetry budget differs from requested block')
                    result['scoreShare'] = compare.score_share(result)
                    block[arm] = result
                report[key].append(block)
                save()
                print(f"{name} block {index + 1}: previous={block['previous']['scoreShare']:.2%} "
                      f"candidate={block['candidate']['scoreShare']:.2%}", flush=True)

        retcon_stage('primary', 'fixed-minrisk', 'primaryBlocks')
        report['primary'] = primary_decision(report['primaryBlocks'], args.blocks)
        report['primaryMechanism'] = retcon_mechanism(report['primaryBlocks'])
        save()
        archive = benchmark.download(cache, 'sample-bots-Java-1.4.0.zip', benchmark.ASSETS['sample-bots-Java-1.4.0.zip'])
        samples = output / 'official-samples'
        benchmark.extract(archive, samples)
        for index, opponent in enumerate(('Target', 'SpinBot', 'Walls')):
            row = dict(opponent=opponent, order=['candidate', 'previous'] if index == 1 else ['previous', 'candidate'])
            report['samples'].append(row)
            for arm in row['order']:
                compare.verify_snapshots(snapshots, hashes)
                arm_output = output / 'samples' / opponent / arm
                arm_output.mkdir(parents=True)
                config = argparse.Namespace(java=args.java, bot_dir=snapshots[arm], seed=args.sample_seed,
                    opponent_env={}, retcon_source=None, timeout=300, training_rounds=20, require_metrics=True)
                row[arm] = benchmark.battle(config, arm_output, runner, samples, opponent, [opponent],
                                            'fixed-minrisk', SAMPLE_ROUNDS, arm_output / 'absent-model.dat')
                row[arm]['scoreShare'] = compare.score_share(row[arm])
                save()
        online = None
        if report['primary']['status'] == 'pass':
            retcon_stage('online', 'train', 'onlineBlocks')
            online = report['online'] = online_decision(report['onlineBlocks'], args.blocks)
            report['onlineMechanism'] = retcon_mechanism(report['onlineBlocks'])
        compare.verify_snapshots(snapshots, hashes)
        report.update(status='complete', decision=final_decision(report['primary'], report['samples'], online))
        save()
        print(json.dumps(report['decision'], indent=2))
    except BaseException as error:
        report.update(status='invalid', failure=dict(type=type(error).__name__, message=str(error)))
        save()
        raise


if __name__ == '__main__':
    main()
