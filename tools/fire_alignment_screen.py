"""Execute the prospectively locked 48-round diagnostic; never a retention gate."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import benchmark
import compare
import retcon_stats

BASE = '204ad4c9'
BASE_JAR = '1b99fa0ac303e5e5f89ed91bb3803841ded54910628f335fe1aeaeba65e34c9a'
PLAN = '35e95cc92d728584161a992fdc8b84fb04979d69'


def screen_decision(comparison, samples):
    previous, candidate = (compare.aggregate(comparison['blocks'], arm) for arm in ('previous', 'candidate'))
    deltas = [b['candidate']['scoreShare'] - b['previous']['scoreShare'] for b in comparison['blocks']]
    mechanism_defined = previous['shotsFired'] > 0 and candidate['shotsFired'] > 0
    checks = dict(complete=len(deltas) == 3 and all(b[a]['rounds'] == 5 for b in comparison['blocks'] for a in ('previous', 'candidate')),
        meanShare=sum(deltas) / 3 >= 0, nonnegativeBlocks=sum(d >= 0 for d in deltas) >= 2,
        worstBlock=min(deltas) >= -.10, noSkippedTurns=candidate['skippedTurns'] == 0,
        shotRate=candidate['shotsFired'] / candidate['turns'] >= .5 * previous['shotsFired'] / previous['turns'])
    if mechanism_defined:
        checks['shotEfficiency'] = (candidate['hitRate'] > previous['hitRate'] or
            candidate['damageDealt'] / candidate['shotsFired'] > previous['damageDealt'] / previous['shotsFired'])
    checks['samples'] = len(samples) == 3 and all(
        row['candidate']['rounds'] == row['previous']['rounds'] == 3 and
        compare.score_share(row['candidate']) - compare.score_share(row['previous']) >= -.05 and
        row['candidate']['metrics']['skippedTurns'] == 0 for row in samples)
    return dict(status='inconclusive' if not mechanism_defined else 'advance-to-confirmation' if all(checks.values()) else 'reject',
        checks=checks, previous=previous, candidate=candidate, scoreShareDeltas=deltas,
        damagePerShot={arm: value['damageDealt'] / value['shotsFired'] if value['shotsFired'] else None
                       for arm, value in (('previous', previous), ('candidate', candidate))},
        limitation='48-round diagnostic only; no gameplay retention or strength claim')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous-bot-dir', type=Path, required=True)
    parser.add_argument('--candidate-source', required=True)
    parser.add_argument('--java', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve prior evidence; output must be new')
    if benchmark.digest(args.previous_bot_dir / 'AISurvivor.jar') != BASE_JAR:
        raise ValueError('Baseline JAR does not match accepted source')
    args.output.mkdir(parents=True)
    report = dict(status='running', runtimeOS=retcon_stats.os_provenance(), samples=[])
    destination = args.output / 'screen.json'
    def save():
        destination.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    save()
    try:
        subprocess.run([sys.executable, str(Path(__file__).with_name('compare.py')),
            '--previous-bot-dir', str(args.previous_bot_dir), '--candidate-bot-dir', str(benchmark.ROOT / 'AISurvivor'),
            '--previous-source', subprocess.check_output(['git', 'rev-parse', BASE], text=True).strip(),
            '--candidate-source', args.candidate_source, '--plan', str(benchmark.ROOT / 'docs/improvements/fire-alignment-screen.md'),
            '--plan-commit', PLAN, '--purpose', 'diagnostic', '--policy', 'fixed-minrisk', '--rounds', '5',
            '--seeds', '2026110401,2026110402,2026110403', '--java', args.java, '--timeout', '300',
            '--require-retcon-stats', '--output', str(args.output / 'retcon')], check=True)
        comparison = json.loads((args.output / 'retcon/comparison.json').read_text())
        cache = benchmark.ROOT / '.benchmark-cache'
        runner = benchmark.download(cache, 'robocode-tankroyale-runner-1.4.0.jar', benchmark.ASSETS['robocode-tankroyale-runner-1.4.0.jar'])
        archive = benchmark.download(cache, 'sample-bots-Java-1.4.0.zip', benchmark.ASSETS['sample-bots-Java-1.4.0.zip'])
        samples = args.output / 'official-samples'
        benchmark.extract(archive, samples)
        snapshots = {arm: args.output / 'retcon/artifacts' / arm for arm in ('previous', 'candidate')}
        for index, opponent in enumerate(('Target', 'SpinBot', 'Walls')):
            compare.verify_snapshots(snapshots, comparison['bundleSHA256'])
            row = dict(opponent=opponent, order=['candidate', 'previous'] if index == 1 else ['previous', 'candidate'])
            report['samples'].append(row)
            for arm in row['order']:
                output = args.output / 'samples' / opponent / arm
                output.mkdir(parents=True)
                config = argparse.Namespace(java=args.java, bot_dir=snapshots[arm], seed=2026110404,
                    opponent_env={}, retcon_source=None, timeout=180, training_rounds=20, require_metrics=True)
                row[arm] = benchmark.battle(config, output, runner, samples, opponent, [opponent],
                    'fixed-minrisk', 3, output / 'absent-model.dat')
                save()
        compare.verify_snapshots(snapshots, comparison['bundleSHA256'])
        report.update(status='complete', decision=screen_decision(comparison, report['samples']))
        save()
        print(json.dumps(report['decision'], indent=2))
    except BaseException as error:
        report.update(status='invalid', failure=dict(type=type(error).__name__, message=str(error)))
        save()
        raise


if __name__ == '__main__':
    main()
