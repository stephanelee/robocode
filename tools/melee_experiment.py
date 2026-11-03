"""Registered melee candidate comparison (docs/improvements/coherent-melee.md).

Stage A (primary, superiority): default-product 8-bot melee against seven official samples, N blocks per arm.
Stage B (non-inferiority): default-product Retcon 1v1 blocks.
Stage C: official Target/SpinBot/Walls 1v1 regression.
All stages run in default `train` mode with fresh models. Retcon native statistics are offline evidence only.
"""
import argparse
import json
import re
import subprocess
from pathlib import Path

import benchmark
import compare
import retcon
import retcon_stats
from block_experiment import SAMPLE_ROUNDS, block_stats, noninferiority_decision, retcon_mechanism, samples_pass

MELEE = ['Corners', 'Crazy', 'RamFire', 'TrackFire', 'VelociBot', 'SpinBot', 'Walls']


def melee_decision(blocks, blocks_required, min_gain=0.02):
    """Superiority: mean block share gain >= min_gain, Student-t 95% lower bound > 0, at least as many first places,
    no more skipped turns and no material wall-rate increase."""
    s = block_stats(blocks, 't')
    prev, cand = s['arms']['previous'], s['arms']['candidate']
    checks = dict(complete=len(blocks) == blocks_required,
                  meanGain=s['meanDelta'] >= min_gain - 1e-12,
                  lowerBoundPositive=s['lowerBound95'] > 0,
                  firstPlacesNotLower=cand['wins'] >= prev['wins'],
                  skippedNotMore=cand['skippedTurns'] <= prev['skippedTurns'],
                  wallRate=cand['wallsPer1000Turns'] <= prev['wallsPer1000Turns'] + 10 + 1e-12)
    return dict(status='pass' if all(checks.values()) else 'fail', checks=checks, **s)


def final_decision(melee, retcon_check, samples):
    checks = dict(melee=melee['status'] == 'pass', retcon=retcon_check['status'] == 'pass',
                  samples=samples_pass(samples))
    return dict(status='retain' if all(checks.values()) else 'reject', checks=checks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous-bot-dir', type=Path, required=True)
    parser.add_argument('--expected-previous-jar', required=True)
    parser.add_argument('--previous-source', required=True)
    parser.add_argument('--candidate-source', required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-commit', required=True)
    parser.add_argument('--melee-blocks', type=int, required=True)
    parser.add_argument('--melee-rounds', type=int, required=True)
    parser.add_argument('--retcon-blocks', type=int, required=True)
    parser.add_argument('--retcon-rounds', type=int, required=True)
    parser.add_argument('--seed-base', type=int, required=True)
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
    report = dict(schemaVersion=1, status='running', planCommit=args.plan_commit, previousSource=args.previous_source,
                  candidateSource=args.candidate_source, bundleSHA256=hashes, seedBase=args.seed_base,
                  meleeBlocks=args.melee_blocks, meleeRounds=args.melee_rounds, retconBlocks=args.retcon_blocks,
                  retconRounds=args.retcon_rounds, runtimeOS=retcon_stats.os_provenance(),
                  meleeBlocksData=[], retconBlocksData=[], samples=[])
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
        archive = benchmark.download(cache, 'sample-bots-Java-1.4.0.zip', benchmark.ASSETS['sample-bots-Java-1.4.0.zip'])
        samples = output / 'official-samples'
        benchmark.extract(archive, samples)
        report.update(opponentProvenance=[provenance], runnerSHA256=benchmark.digest(runner))

        def stage(key, name, opponents, opponent_dir, env, retcon_source, blocks, rounds, seed_offset):
            for index in range(blocks):
                seed = args.seed_base + seed_offset + index
                order = ['previous', 'candidate'] if index % 2 == 0 else ['candidate', 'previous']
                block = dict(seed=seed, order=order)
                for arm in order:
                    compare.verify_snapshots(snapshots, hashes)
                    arm_output = output / key / f'block-{index + 1:02d}' / arm
                    arm_output.mkdir(parents=True)
                    config = argparse.Namespace(java=args.java, bot_dir=snapshots[arm], seed=seed, timeout=1800,
                        training_rounds=rounds, require_metrics=True, require_retcon_stats=retcon_source is not None,
                        opponent_env=env, retcon_source=retcon_source)
                    result = benchmark.battle(config, arm_output, runner, opponent_dir, name, opponents, 'train',
                                              rounds, arm_output / 'model.dat')
                    if result['rounds'] != rounds or result['metrics']['roundsRecorded'] != rounds:
                        raise ValueError('Validated battle/telemetry budget differs from requested block')
                    result['scoreShare'] = compare.score_share(result)
                    block[arm] = result
                report[key].append(block)
                save()
                print(f"{key} block {index + 1}: previous={block['previous']['scoreShare']:.2%} "
                      f"candidate={block['candidate']['scoreShare']:.2%}", flush=True)

        stage('meleeBlocksData', 'melee', MELEE, samples, {}, None, args.melee_blocks, args.melee_rounds, 0)
        report['melee'] = melee_decision(report['meleeBlocksData'], args.melee_blocks)
        save()
        stage('retconBlocksData', 'Retcon', ['Retcon'], output, {'RUMBLE_JAVA_CACHE': str(api_cache)}, source,
              args.retcon_blocks, args.retcon_rounds, 100)
        report['retcon'] = noninferiority_decision(report['retconBlocksData'], args.retcon_blocks, 't')
        report['retconMechanism'] = retcon_mechanism(report['retconBlocksData'])
        save()
        for index, opponent in enumerate(('Target', 'SpinBot', 'Walls')):
            row = dict(opponent=opponent, order=['candidate', 'previous'] if index == 1 else ['previous', 'candidate'])
            report['samples'].append(row)
            for arm in row['order']:
                compare.verify_snapshots(snapshots, hashes)
                arm_output = output / 'samples' / opponent / arm
                arm_output.mkdir(parents=True)
                config = argparse.Namespace(java=args.java, bot_dir=snapshots[arm], seed=args.seed_base + 200,
                    opponent_env={}, retcon_source=None, timeout=600, training_rounds=SAMPLE_ROUNDS, require_metrics=True)
                row[arm] = benchmark.battle(config, arm_output, runner, samples, opponent, [opponent], 'train',
                                            SAMPLE_ROUNDS, arm_output / 'model.dat')
                row[arm]['scoreShare'] = compare.score_share(row[arm])
                save()
        compare.verify_snapshots(snapshots, hashes)
        report.update(status='complete', decision=final_decision(report['melee'], report['retcon'], report['samples']))
        save()
        print(json.dumps(report['decision'], indent=2))
    except BaseException as error:
        report.update(status='invalid', failure=dict(type=type(error).__name__, message=str(error)))
        save()
        raise


if __name__ == '__main__':
    main()
