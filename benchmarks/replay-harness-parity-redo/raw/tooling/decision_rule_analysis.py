"""Offline operating characteristics of proposed block-comparison decision rules (review correction, PR #37).

Separates significance-only power from full-rule acceptance probability under the normal block-delta model, using
the A/A calibration's measured SD. Compares the registered z-bound with a Student-t bound, accounts for pilot-SD
uncertainty, and bootstraps the wins and wall guards from the real A/A blocks under the null. Deterministic: every
simulation uses a fixed seed. No matches are played.
"""
import argparse
import json
import math
import random
import statistics
from pathlib import Path

Z95 = 1.6448536269514722
# One-sided 95% Student-t quantiles t_{0.95, df} for df = 1..29 (blocks up to the 30-block cap).
T95 = {1: 6.313752, 2: 2.919986, 3: 2.353363, 4: 2.131847, 5: 2.015048, 6: 1.943180, 7: 1.894579,
       8: 1.859548, 9: 1.833113, 10: 1.812461, 11: 1.795885, 12: 1.782288, 13: 1.770933, 14: 1.761310,
       15: 1.753050, 16: 1.745884, 17: 1.739607, 18: 1.734064, 19: 1.729133, 20: 1.724718, 21: 1.720743,
       22: 1.717144, 23: 1.713872, 24: 1.710882, 25: 1.708141, 26: 1.705618, 27: 1.703288, 28: 1.701131,
       29: 1.699127}


def lower_bound(deltas, method):
    n = len(deltas)
    se = statistics.stdev(deltas) / math.sqrt(n)
    quantile = Z95 if method == 'z' else T95[n - 1]
    return statistics.mean(deltas) - quantile * se


def score_rule(deltas, method, floor):
    """Score part of a rule: one-sided 95% lower bound > 0 and, if floor is set, mean >= floor."""
    passes = lower_bound(deltas, method) > 0
    return passes and (floor is None or statistics.mean(deltas) >= floor - 1e-12)


def acceptance(blocks, effect, sd, method, floor, draws, seed, pilot_df=None):
    """P(score rule passes) for independent normal block deltas. With pilot_df, sigma is redrawn per trial from
    sd^2 * df / chi2_df to reflect uncertainty in a pilot SD estimated with df degrees of freedom."""
    rng = random.Random(seed)
    passed = 0
    for _ in range(draws):
        sigma = sd
        if pilot_df:
            chi2 = sum(rng.gauss(0, 1) ** 2 for _ in range(pilot_df))
            sigma = sd * math.sqrt(pilot_df / chi2)
        deltas = [rng.gauss(effect, sigma) for _ in range(blocks)]
        passed += score_rule(deltas, method, floor)
    return passed / draws


def blocks_for(target, effect, sd, method, floor, draws, seed, cap=30):
    for blocks in range(3, cap + 1):
        if acceptance(blocks, effect, sd, method, floor, draws, seed) >= target:
            return blocks
    return None


def null_guard_rates(aa_blocks, blocks, draws, seed):
    """Bootstrap the wins (candidate >= baseline) and wall (rate <= baseline + 10/1000 turns) guards from real A/A
    blocks under no effect: resample `blocks` A/A blocks with replacement and use arm a or b as candidate."""
    rng = random.Random(seed)
    wins_fail = wall_fail = either_fail = 0
    for _ in range(draws):
        sample = [rng.choice(aa_blocks) for _ in range(blocks)]
        cand, prev = ('a', 'b') if rng.random() < .5 else ('b', 'a')
        wins_ok = sum(b[cand]['firstPlaces'] for b in sample) >= sum(b[prev]['firstPlaces'] for b in sample)
        rate = {arm: 1000 * sum(b[arm]['metrics']['wallCollisions'] for b in sample) /
                sum(b[arm]['metrics']['turns'] for b in sample) for arm in (cand, prev)}
        wall_ok = rate[cand] <= rate[prev] + 10 + 1e-12
        wins_fail += not wins_ok
        wall_fail += not wall_ok
        either_fail += not (wins_ok and wall_ok)
    return dict(winsGuardFail=wins_fail / draws, wallGuardFail=wall_fail / draws, eitherFail=either_fail / draws)


def analyse(calibration, draws=40000, seed=20261103):
    sd = calibration['summary']['deltaSD']
    pilot_df = calibration['summary']['blocks'] - 1
    effects = [0.0, 0.05, 0.075, 0.10, 0.15]
    table = []
    for blocks in (8, 12, 16):
        for effect in effects:
            row = dict(blocks=blocks, effect=effect)
            for name, method, floor in (('significanceZ', 'z', None), ('significanceT', 't', None),
                                        ('registeredZWithFloor', 'z', 0.05), ('tWithFloor', 't', 0.05)):
                row[name] = acceptance(blocks, effect, sd, method, floor, draws, seed)
            row['tWithFloorPilotUncertain'] = acceptance(blocks, effect, sd, 't', 0.05, draws // 4, seed, pilot_df)
            table.append(row)
    budgets = {f'{target}@{effect}': blocks_for(target, effect, sd, 't', 0.05, draws // 2, seed)
               for target in (0.8,) for effect in (0.075, 0.10)}
    guards = {str(blocks): null_guard_rates(calibration['blocks'], blocks, draws, seed) for blocks in (8, 12, 16)}
    return dict(seed=seed, draws=draws, deltaSD=sd, pilotDegreesOfFreedom=pilot_df, table=table,
                blocksFor80PctAcceptanceTWithFloor=budgets, nullGuardFailureFromAABootstrap=guards,
                notes=['Normal independent block-delta model; rounds within a block share an adaptive Retcon.',
                       'Wins and wall guards are bootstrapped only under the null (no effect); their behaviour under '
                       'a real effect is not calibrated.', 'Skipped-turn guard: zero skips in all 400 A/A rounds; '
                       'not otherwise calibrated.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--draws', type=int, default=40000)
    args = parser.parse_args()
    result = analyse(json.loads(args.calibration.read_text()), args.draws)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
