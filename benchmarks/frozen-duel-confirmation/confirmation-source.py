"""Ten fresh games of one frozen controller; no training or gameplay changes."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import benchmark
import compare
import retcon
import retcon_stats

SOURCE = '7104b65d03298caceb010a5bfd085bda65092d17'
JAR = 'd047bf99325d2ad73aedd87eac6a4baa1677170afff0c86b948f03ea93390e1d'
LAYOUT = 'tools/fixtures/frozen-duel-confirmation/games.json'
PLAN = 'docs/improvements/frozen-duel-confirmation.md'


def decision(rows):
    if len(rows) != 10 or len({r['gameId'] for r in rows}) != 10:
        raise ValueError('Exactly ten unique games required')
    if any(r['battle']['rounds'] != 20 for r in rows):
        raise ValueError('Exactly twenty rounds per game required')
    shares = [compare.score_share(r['battle']) for r in rows]
    wins = sum(r['battle']['firstPlaces'] for r in rows)
    checks = dict(meanShare=statistics.mean(shares) > .5,
                  majorityRoundWins=wins > 100, majorityGames=sum(s > .5 for s in shares) > 5)
    return dict(status='confirmed' if all(checks.values()) else 'not-confirmed', checks=checks,
        meanShare=statistics.mean(shares), wins=wins, rounds=200, gamesAboveHalf=sum(s > .5 for s in shares),
        gameShares=shares, gameShareSD=statistics.stdev(shares),
        descriptiveTLower95=statistics.mean(shares) - 1.8331129326536335 * statistics.stdev(shares) / math.sqrt(10),
        descriptiveWinWilson95=compare.wilson(wins, 200),
        limitation='Fixed-policy pinned-opponent confirmation only. Within-game adaptation makes rounds non-IID; intervals are descriptive. No default-policy or general-strength claim.')


def registration(commit):
    if len(commit) != 40 or any(c not in '0123456789abcdef' for c in commit):
        raise ValueError('Full immutable registration commit required')
    layout = subprocess.check_output(['git', 'show', commit + ':' + LAYOUT], cwd=benchmark.ROOT)
    plan = subprocess.check_output(['git', 'show', commit + ':' + PLAN], cwd=benchmark.ROOT)
    games = json.loads(layout)['games']
    expected = [(b, i, 20261600 + (b - 1) * 10 + i) for b in (1, 2) for i in range(1, 6)]
    if [(g['batch'], g['index'], g['seed']) for g in games] != expected:
        raise ValueError('Registration differs from fixed ten-game budget/order/seeds')
    if [g['gameId'] for g in games] != [f'fc710-b{b}-g{i:02d}' for b, i, _ in expected]:
        raise ValueError('Game identities differ from registration')
    return games, layout, plan


def run(args):
    args.output, args.bot_dir, args.cache = args.output.resolve(), args.bot_dir.resolve(), args.cache.resolve()
    if args.output.exists():
        raise ValueError('Output must be new; never replace an attempt')
    args.output.mkdir(parents=True)
    games, layout, plan = registration(args.plan_commit)
    (args.output / 'registered-plan.md').write_bytes(plan)
    (args.output / 'registered-games.json').write_bytes(layout)
    report = dict(status='running', botSource=SOURCE, botJarSHA256=JAR,
        experimentSource=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        registrationCommit=args.plan_commit, planSHA256=hashlib.sha256(plan).hexdigest(),
        layoutSHA256=hashlib.sha256(layout).hexdigest(), batch=args.batch, games=[], runtimeOS=retcon_stats.os_provenance())
    target = args.output / 'batch.json'
    def save():
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    save()
    try:
        if benchmark.digest(args.bot_dir / 'AISurvivor.jar') != JAR:
            raise ValueError('Frozen bot JAR differs')
        runtime = subprocess.run([args.java, '-XshowSettings:properties', '-version'], capture_output=True, text=True, timeout=15, check=True)
        report['runtimeJava'] = runtime.stdout + runtime.stderr
        source, api_cache, provenance = retcon.prepare(args.cache)
        runner = benchmark.download(args.cache, 'robocode-tankroyale-runner-1.4.0.jar', benchmark.ASSETS['robocode-tankroyale-runner-1.4.0.jar'])
        report.update(opponentProvenance=provenance, runnerSHA256=benchmark.digest(runner))
        for game in (g for g in games if g['batch'] == args.batch):
            if benchmark.digest(args.bot_dir / 'AISurvivor.jar') != JAR:
                raise ValueError('Frozen bot JAR changed during batch')
            report['inProgressGame'] = game['gameId']; save()
            options = argparse.Namespace(java=args.java, bot_dir=args.bot_dir, seed=game['seed'], timeout=900,
                training_rounds=20, require_metrics=True, require_retcon_stats=True,
                opponent_env={'RUMBLE_JAVA_CACHE': str(api_cache)}, retcon_source=source)
            output = args.output / game['gameId']; output.mkdir()
            battle = benchmark.battle(options, output, runner, output, 'Retcon', ['Retcon'], 'fixed-minrisk', 20, output / 'absent-model.dat')
            if battle['metrics']['skippedTurns'] or battle['metrics']['checkpointLoaded'] or battle['modelSHA256'] is not None:
                raise ValueError('Confirmation requires zero AIS skips and no model loads/writes')
            report['games'].append(dict(**game, battle=battle)); del report['inProgressGame']; save()
        report['status'] = 'complete'
    except BaseException as error:
        report.update(status='invalid', error=str(error)); raise
    finally:
        save()


def aggregate(args):
    args.input, args.output = args.input.resolve(), args.output.resolve()
    if args.output.exists():
        raise ValueError('Output must be new')
    games, layout, plan = registration(args.plan_commit)
    reports = [json.loads(p.read_bytes()) for p in sorted(args.input.rglob('batch.json'))]
    if len(reports) != 2 or {r['batch'] for r in reports} != {1, 2}:
        raise ValueError('Both distinct batches required')
    if len({r['experimentSource'] for r in reports}) != 1:
        raise ValueError('Batches must come from the same frozen execution source')
    for r in reports:
        if (r['status'] != 'complete' or r['botSource'] != SOURCE or r['botJarSHA256'] != JAR
                or r['registrationCommit'] != args.plan_commit or r['planSHA256'] != hashlib.sha256(plan).hexdigest()
                or r['layoutSHA256'] != hashlib.sha256(layout).hexdigest()):
            raise ValueError('Incomplete or mismatched batch provenance')
    rows = sorted([g for r in reports for g in r['games']], key=lambda g: (g['batch'], g['index']))
    if [{k: g[k] for k in ('gameId', 'batch', 'index', 'seed')} for g in rows] != games:
        raise ValueError('Actual games differ from registered layout')
    result = dict(status='complete', botSource=SOURCE, botJarSHA256=JAR, registrationCommit=args.plan_commit,
        games=rows, decision=decision(rows), batches=[dict(batch=r['batch'], meanShare=statistics.mean(compare.score_share(g['battle']) for g in r['games']), wins=sum(g['battle']['firstPlaces'] for g in r['games'])) for r in reports])
    args.output.mkdir(parents=True)
    (args.output / 'confirmation.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(result['decision'], indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan-commit', required=True)
    parser.add_argument('--batch', type=int, choices=(1, 2))
    parser.add_argument('--input', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bot-dir', type=Path)
    parser.add_argument('--java', default='java')
    parser.add_argument('--cache', type=Path, default=benchmark.ROOT / '.benchmark-cache')
    args = parser.parse_args()
    if args.input:
        aggregate(args)
    elif args.batch and args.bot_dir:
        run(args)
    else:
        parser.error('Use --input for aggregation or --batch and --bot-dir for a batch')
