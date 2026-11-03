"""A/A noise calibration against Retcon (docs/aa-calibration.md).

Runs the same accepted AISurvivor bundle as two arms, `a` and `b`, in fixed minimum-risk 1v1 games against the
pinned Retcon, alternating arm order by block. It measures between-block score-share variance and estimates the
block budget a future +5 pp comparison would need. It changes no gameplay and makes no promotion decision.
"""
import argparse
import json
import math
import random
import re
import statistics
import subprocess
from pathlib import Path

import benchmark
import compare
import retcon
import retcon_stats

Z_ALPHA = 1.6448536269514722  # one-sided 5%
Z_POWER = 0.8416212335729143  # 80% power
EFFECT = 0.05


def required_blocks(sd_delta, effect=EFFECT):
    """Blocks per arm so a one-sided 5% test on the mean paired-block delta has 80% power at `effect`."""
    if not (sd_delta > 0 and math.isfinite(sd_delta)):
        raise ValueError('delta SD must be positive and finite')
    return math.ceil(((Z_ALPHA + Z_POWER) * sd_delta / effect) ** 2)


def current_gate_pass_rate(sd_delta, effect, draws=200_000, seed=20261301):
    """Simulated pass rate of the existing tactical score checks (3 blocks; mean >= +5 pp, >= 2 blocks > 0,
    no block < -5 pp) when block deltas are independent normal(effect, sd_delta). Score checks only."""
    rng = random.Random(seed)
    passed = 0
    for _ in range(draws):
        deltas = [rng.gauss(effect, sd_delta) for _ in range(3)]
        if sum(deltas) / 3 >= 0.05 and sum(d > 0 for d in deltas) >= 2 and min(deltas) >= -0.05:
            passed += 1
    return passed / draws


def summarize(blocks):
    shares = {arm: [b[arm]['scoreShare'] for b in blocks] for arm in ('a', 'b')}
    deltas = [b['a']['scoreShare'] - b['b']['scoreShare'] for b in blocks]
    pooled = shares['a'] + shares['b']
    sd_delta = statistics.stdev(deltas)
    result = dict(blocks=len(blocks), roundsPerBlock=blocks[0]['a']['rounds'],
                  meanShare={arm: statistics.mean(values) for arm, values in shares.items()},
                  pooledMeanShare=statistics.mean(pooled), pooledBlockShareSD=statistics.stdev(pooled),
                  meanDelta=statistics.mean(deltas), deltaSD=sd_delta,
                  deltaSE=sd_delta / math.sqrt(len(deltas)), deltaRange=[min(deltas), max(deltas)],
                  blocksPerArmFor80PctPowerAt5pp=required_blocks(sd_delta),
                  roundsPerArmFor80PctPowerAt5pp=required_blocks(sd_delta) * blocks[0]['a']['rounds'],
                  currentGatePassRate={str(e): current_gate_pass_rate(sd_delta, e) for e in (0.0, 0.05, 0.10)})
    mechanism = {}
    for arm in ('a', 'b'):
        hits = shots = 0
        for b in blocks:
            gun = (b[arm].get('retconStats') or {}).get('actualGun') or {}
            hits += gun.get('hits', 0)
            shots += gun.get('shots', 0)
        mechanism[arm] = dict(retconHits=hits, retconShots=shots, retconHitRate=hits / shots if shots else None,
                              wins=sum(b[arm]['firstPlaces'] for b in blocks))
    result['mechanism'] = mechanism
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bot-dir', type=Path, required=True)
    parser.add_argument('--expected-jar', required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-commit', required=True)
    parser.add_argument('--blocks', type=int, default=10)
    parser.add_argument('--rounds', type=int, default=20)
    parser.add_argument('--seed-base', type=int, default=20261301)
    parser.add_argument('--java', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise ValueError('Preserve prior evidence; output must be new')
    if benchmark.digest(args.bot_dir / 'AISurvivor.jar') != args.expected_jar:
        raise ValueError('Bot JAR does not match the accepted source')
    plan = compare.registered_plan(args.plan, args.plan_commit)
    output.mkdir(parents=True)
    (output / 'preregistered-plan.md').write_bytes(plan)
    snapshots, hashes = compare.snapshot_bots(args.bot_dir.resolve(), args.bot_dir.resolve(), output)
    snapshots = dict(a=snapshots['previous'], b=snapshots['candidate'])
    hashes = dict(a=hashes['previous'], b=hashes['candidate'])
    report = dict(schemaVersion=1, status='running', purpose='A/A noise calibration; no gameplay decision',
                  source=args.source, planCommit=args.plan_commit, bundleSHA256=hashes, blocks=[],
                  runtimeOS=retcon_stats.os_provenance(), plannedRounds=2 * args.blocks * args.rounds)
    destination = output / 'calibration.json'

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
        for index in range(args.blocks):
            seed = args.seed_base + index
            order = ['a', 'b'] if index % 2 == 0 else ['b', 'a']
            block = dict(seed=seed, order=order)
            for arm in order:
                compare.verify_snapshots({arm: snapshots[arm]}, {arm: hashes[arm]})
                arm_output = output / f'block-{index + 1:02d}' / arm
                arm_output.mkdir(parents=True)
                config = argparse.Namespace(java=args.java, bot_dir=snapshots[arm], seed=seed, timeout=900,
                    training_rounds=20, require_metrics=True, require_retcon_stats=True,
                    opponent_env={'RUMBLE_JAVA_CACHE': str(api_cache)}, retcon_source=source)
                result = benchmark.battle(config, arm_output, runner, output, 'Retcon', ['Retcon'],
                                          'fixed-minrisk', args.rounds, arm_output / 'absent-model.dat')
                if result['rounds'] != args.rounds or result['metrics']['roundsRecorded'] != args.rounds:
                    raise ValueError('Validated battle/telemetry budget differs from requested block')
                result['scoreShare'] = compare.score_share(result)
                block[arm] = result
            report['blocks'].append(block)
            save()
            print(f"block {index + 1}: a={block['a']['scoreShare']:.2%} b={block['b']['scoreShare']:.2%}", flush=True)
        compare.verify_snapshots(snapshots, hashes)
        report.update(status='complete', summary=summarize(report['blocks']))
        save()
        print(json.dumps(report['summary'], indent=2))
    except BaseException as error:
        report.update(status='invalid', failure=dict(type=type(error).__name__, message=str(error)))
        save()
        raise


if __name__ == '__main__':
    main()
