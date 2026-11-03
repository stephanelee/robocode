"""Registered start-of-game reliability comparison (docs/improvements/warmup-reliability.md).

Plays many short games per arm, alternating arm order, against pinned Retcon (1v1) and a seven-bot official sample
melee. Each game start is a trial for cold-JVM skipped turns. Gameplay is not evaluated beyond descriptive scores.
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

MELEE = ['Corners', 'Crazy', 'RamFire', 'TrackFire', 'VelociBot', 'SpinBot', 'Walls']


def summarize(games):
    """Per arm: games, games with any skipped turn, total skipped turns and the largest recorded turn latency."""
    result = {}
    for arm in ('previous', 'candidate'):
        rows = [g[arm] for g in games]
        latencies = [r for g in rows for r in g['maxTurnMicros'] if r is not None]
        result[arm] = dict(games=len(rows), gamesWithSkips=sum(any(s > 0 for s in g['skipped']) for g in rows),
                           skippedTurns=sum(sum(g['skipped']) for g in rows),
                           maxTurnMicros=max(latencies) if latencies else None,
                           slowTurns=sum(g['slowTurns'] for g in rows if g['slowTurns'] is not None),
                           meanScoreShare=sum(g['scoreShare'] for g in rows) / len(rows) if rows else None)
    return result


def decision(summary):
    """Registered rule: retain if the candidate never skips more; an effect is claimed only for strictly fewer."""
    prev, cand = summary['previous'], summary['candidate']
    not_worse = cand['skippedTurns'] <= prev['skippedTurns'] and cand['gamesWithSkips'] <= prev['gamesWithSkips']
    effect = not_worse and cand['gamesWithSkips'] < prev['gamesWithSkips']
    status = 'retain-effect' if effect else 'retain-neutral' if not_worse else 'reject'
    return dict(status=status, notWorse=not_worse, fewerSkippingGames=effect)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous-bot-dir', type=Path, required=True)
    parser.add_argument('--expected-previous-jar', required=True)
    parser.add_argument('--previous-source', required=True)
    parser.add_argument('--candidate-source', required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-commit', required=True)
    parser.add_argument('--games', type=int, required=True, help='Games per arm per scenario')
    parser.add_argument('--rounds', type=int, default=2)
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
                  candidateSource=args.candidate_source, bundleSHA256=hashes, gamesPerArmPerScenario=args.games,
                  roundsPerGame=args.rounds, seedBase=args.seed_base, runtimeOS=retcon_stats.os_provenance(),
                  scenarios={})
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
        scenarios = [('retcon', 'Retcon', ['Retcon'], output, {'RUMBLE_JAVA_CACHE': str(api_cache)}, source),
                     ('melee', 'melee', MELEE, samples, {}, None)]
        for key, name, opponents, opponent_dir, env, retcon_source in scenarios:
            games = report['scenarios'].setdefault(key, dict(games=[]))['games']
            for index in range(args.games):
                seed = args.seed_base + index
                order = ['previous', 'candidate'] if index % 2 == 0 else ['candidate', 'previous']
                game = dict(seed=seed, order=order)
                for arm in order:
                    compare.verify_snapshots(snapshots, hashes)
                    arm_output = output / key / f'game-{index + 1:02d}' / arm
                    arm_output.mkdir(parents=True)
                    config = argparse.Namespace(java=args.java, bot_dir=snapshots[arm], seed=seed, timeout=900,
                        training_rounds=args.rounds, require_metrics=True, require_retcon_stats=key == 'retcon',
                        opponent_env=env, retcon_source=retcon_source)
                    result = benchmark.battle(config, arm_output, runner, opponent_dir, name, opponents, 'train',
                                              args.rounds, arm_output / 'model.dat')
                    if result['rounds'] != args.rounds or result['metrics']['roundsRecorded'] != args.rounds:
                        raise ValueError('Validated battle/telemetry budget differs from requested game')
                    rows = [json.loads(line) for line in
                            next(arm_output.rglob('metrics.jsonl')).read_text().splitlines()]
                    game[arm] = dict(scoreShare=compare.score_share(result), rank=result['rank'],
                                     firstPlaces=result['firstPlaces'], skipped=[r['skippedTurns'] for r in rows],
                                     maxTurnMicros=[r.get('maxTurnMicros') for r in rows],
                                     slowTurns=sum(r.get('slowTurns', 0) for r in rows) if arm == 'candidate' else None)
                games.append(game)
                save()
                print(f"{key} game {index + 1}: previous skips={sum(game['previous']['skipped'])} "
                      f"candidate skips={sum(game['candidate']['skipped'])}", flush=True)
            report['scenarios'][key]['summary'] = summarize(games)
        all_games = [g for s in report['scenarios'].values() for g in s['games']]
        report['summary'] = summarize(all_games)
        compare.verify_snapshots(snapshots, hashes)
        report.update(status='complete', decision=decision(report['summary']))
        save()
        print(json.dumps(report['decision'], indent=2))
    except BaseException as error:
        report.update(status='invalid', failure=dict(type=type(error).__name__, message=str(error)))
        save()
        raise


if __name__ == '__main__':
    main()
