#!/usr/bin/env python3
"""Reproduce every Aegis 0.2 development statistic from the retained raw logs.

Usage (from the repository root):  python3 -I benchmarks/aegis-v2-dev/analyses.py

Inputs are raw/<run>/{trace,diamond}.jsonl.gz and <run>/metrics.jsonl under this directory. A missing input is an
error, never silently skipped. `trace` rows are Aegis's own default-off diagnostic trace (legal observations only).
`diamond` rows come from a private dev-only instrumented copy of the pinned opponent (see
dev-only-instrumented-opponent.patch); they were never available to the bot at runtime or used in registered games.

Timing map used throughout. The opponent's `aim` row at its tick T aims for a bullet created in turn N = T + 2.
Its logged source (sx, sy) is its classic-order prediction of its own position at T + 1 = N − 1.
The runtime rule (Aegis.classicSpeed) uses legal scan speeds at N − 3 and N − 2 (= T − 1, T) and the scanned heading
at N − 1 (= T + 1). All three scans are received before the energy drop is detected at scan N, so the offline
comparison uses exactly the inputs available at runtime and involves no look-ahead.
"""
import gzip
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent


def rows(run, name):
    path = HERE / 'raw' / run / f'{name}.jsonl.gz'
    if not path.is_file():
        raise SystemExit(f'missing required input: {path}')
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        return [json.loads(re.sub(r'"mask":(\d+)', r'"mask":"\1"', line)) for line in stream]


def metrics(run):
    path = HERE / run / 'metrics.jsonl'
    if not path.is_file():
        raise SystemExit(f'missing required input: {path}')
    return [json.loads(line) for line in path.read_text().splitlines()]


def scans_by_round(trace):
    out, cur = [], None
    for r in trace:
        if r['ev'] == 'round':
            cur = {}; out.append(cur)
        elif r['ev'] == 'scan' and cur is not None:
            cur[r['turn']] = r
    return out


def diamond_rounds(diamond, kinds):
    """Opponent rows split into rounds at a turn-counter reset (its clock restarts each round)."""
    out, cur, last = [], [], -1
    for r in diamond:
        if r['ev'] not in kinds:
            continue
        t = r.get('t', 0)
        if t < last - 50:
            out.append(cur); cur = []
        last = t; cur.append(r)
    out.append(cur)
    return out


def sgn(x): return (x > 0) - (x < 0)


def rule_new(previous, current):  # mirrors Aegis.classicSpeed in the WIP patch
    if abs(current - previous) < 1e-9: return current
    if current != 0 and previous != 0 and sgn(current) != sgn(previous): return max(-8, min(8, current + sgn(current)))
    if abs(current) > abs(previous): return max(-8, min(8, current + sgn(current)))
    if current == 0: return -sgn(previous) if abs(previous) <= 2 else 0
    if abs(current) > 2: return current - 2 * sgn(current)
    return -(1 - abs(current) / 2) * sgn(current)


def rule_frozen(v0, v1):  # registered 0.1.0 rule (uses the scan at N-1 as its second input)
    s = 1 if v0 > 0 else -1
    if abs(v0) > 2.0001 and (abs(v1) < 1e-9 or (abs(abs(v1) - (abs(v0) / 2 - 1)) < 1e-6 and abs(v0) - abs(v1) > 2.0001)):
        return v0 - 2 * s
    return v1


def self_prediction():
    runs = ['instr1', 'instr2', 'instr3', 'instr4', 'aegis8', 'aegis9', 'aegis10', 'aegis18', 'aegis20', 'aegis22']
    errs = {'frozen 0.1.0 rule': [], 'new rule': []}
    for run in runs:
        scans = scans_by_round(rows(run, 'trace'))
        aims = diamond_rounds(rows(run, 'diamond'), {'aim'})
        if len(scans) != len(aims):
            raise SystemExit(f'{run}: round count mismatch {len(scans)} vs {len(aims)}')
        for sc, rr in zip(scans, aims):
            for a in rr:
                T = a['t']; sm, s0, s1 = sc.get(T - 1), sc.get(T), sc.get(T + 1)
                if not (sm and s0 and s1):
                    continue  # radar gap: no legal three-scan input
                h = math.radians(s1['dir'])
                for name, v in (('frozen 0.1.0 rule', rule_frozen(s0['v'], s1['v'])), ('new rule', rule_new(sm['v'], s0['v']))):
                    errs[name].append(math.hypot(s0['x'] + math.cos(h) * v - a['sx'], s0['y'] + math.sin(h) * v - a['sy']))
    print('## Self-prediction origin error vs logged opponent source (px)')
    for name, e in errs.items():
        print(f'{name}: n={len(e)} mean={statistics.mean(e):.3f} >1px={sum(x > 1 for x in e) / len(e):.4f} '
              f'>2px={sum(x > 2 for x in e) / len(e):.4f} max={max(e):.2f}')


def bin_of(a):
    rel = (a['fa'] - a['absb'] + math.pi) % (2 * math.pi) - math.pi
    return round(rel * a['od'] / 0.0207)  # 0.0207 rad = nominal candidate spacing at 14.15 px/turn


def label_bias():
    """P(off-centre aim) by the mean opponent label GF over the previous K wave breaks (history kept across rounds)."""
    res = defaultdict(Counter)
    for run in ['aegis20', 'aegis18', 'aegis22', 'instr4', 'aegis10']:
        hist = []
        for r in rows(run, 'diamond'):
            if r['ev'] == 'break':
                hist.append(r['gf'])
            elif r['ev'] == 'aim':
                b = bin_of(r)
                if abs(b) > 3 or len(hist) < 20:
                    continue
                m = sum(hist[-20:]) / 20
                res['neg (<-0.015)' if m < -0.015 else 'pos (>0.015)' if m > 0.015 else 'mid'][max(-1, min(1, b))] += 1
    print('## Aim candidate (orbit-normalised, -1 = behind) by mean GF of the previous 20 wave breaks')
    for k in sorted(res):
        c = res[k]; n = sum(c.values())
        print(f'{k}: n={n} centre={c[0] / n:.3f} behind={c[-1] / n:.3f} ahead={c[1] / n:.3f}')


def per_round_guns(run):
    print(f'## {run}: per-round centre-candidate share and gun used')
    for i, rr in enumerate(diamond_rounds(rows(run, 'diamond'), {'aim'}), 1):
        if not rr:
            continue
        c = Counter(max(-2, min(2, bin_of(a))) for a in rr)
        print(i, f'aims={len(rr)} centre={c[0] / len(rr):.3f} behind={c[-1]} ahead={c[1]}', dict(Counter(a['gun'][:8] for a in rr)))


def ratings(run):
    v = [r for r in rows(run, 'diamond') if r['ev'] == 'vg']
    if not v:
        raise SystemExit(f'{run}: no virtual-gun rating rows')
    hits = [r['hit'] for r in v if r['gun'].startswith('Main')]
    dis = Counter()
    for a, b in zip(v, v[1:]):
        if a['ft'] == b['ft'] and a['gun'] != b['gun']:
            m, s = (a, b) if a['gun'].startswith('Main') else (b, a)
            if round(m['fa'] / 0.0207) != round(s['fa'] / 0.0207):
                dis['TripHammer' if m['score'] > s['score'] else 'AntiSurfer'] += 1
    print(f'## {run}: real-shot wave hit angle (rad, orbit-normalised) mean={statistics.mean(hits):.4f} '
          f'mean|.|={statistics.mean(abs(h) for h in hits):.4f} n={len(hits)}; candidate disagreements won by {dict(dis)}')


def autocorrelation():
    tot = Counter(); cond = Counter()
    for run in ['v2a', 'v2c', 'vg1', 'vg2']:
        seq, rnd, fires = [], 0, {}
        for r in rows(run, 'trace'):
            if r['ev'] == 'round': rnd = r['round']
            if r['ev'] == 'fire': fires[(rnd, r['turn'])] = r
            if r['ev'] in ('hitBy', 'blocked') and r.get('matched', -1) > 0 and r.get('solverError') is not None:
                f = fires[(rnd, r['matched'])]
                a = math.hypot(f['tx'] - f['ox'], f['ty'] - f['oy']) * math.tan(math.asin(8 / (20 - 3 * f['power'])) / 29)
                seq.append('A' if abs(r['solverError']) >= a / 2 else '0')
        for i, c in enumerate(seq):
            tot[c] += 1
            if i >= 8: cond[(min(3, seq[i - 8:i].count('A')), c)] += 1
    print(f'## Off-centre persistence (resolved shots, game order): base={tot["A"] / sum(tot.values()):.3f}')
    for k in range(4):
        n = cond[(k, '0')] + cond[(k, 'A')]
        print(f'off-centre among previous 8 = {k if k < 3 else "3+"}: P(next off-centre)={cond[(k, "A")] / n:.3f} n={n}')


def by_power_and_errors():
    st = defaultdict(Counter); err = Counter(); n = 0
    for run in ['vg3', 'eg1']:
        rnd, fires = 0, {}
        for r in rows(run, 'trace'):
            if r['ev'] == 'round': rnd = r['round']
            if r['ev'] == 'fire': fires[(rnd, r['turn'])] = r
            if r['ev'] in ('hitBy', 'blocked') and r.get('matched', -1) > 0:
                f = fires[(rnd, r['matched'])]; p = f['power']
                st['>=1.5' if p >= 1.5 else '0.5-1.5' if p >= 0.5 else '0.2-0.5' if p >= 0.2 else '<0.2'][r['ev']] += 1
                if r.get('solverError') is not None:
                    a = math.hypot(f['tx'] - f['ox'], f['ty'] - f['oy']) * math.tan(math.asin(8 / (20 - 3 * p)) / 29)
                    e = abs(r['solverError']); n += 1
                    err['<1 px' if e < 1 else '1 px .. a/2' if e < a / 2 else 'about a' if e < 1.5 * a else '>=1.5a'] += 1
    print('## vg3+eg1 resolved shots (blocked or hit, matched): block share by inferred enemy power')
    for b in ['>=1.5', '0.5-1.5', '0.2-0.5', '<0.2']:
        c = st[b]; m = c['blocked'] + c['hitBy']
        print(f'{b}: resolved={m} blocked={c["blocked"] / m:.3f}')
    print(f'## vg3+eg1 actual-line error vs solver centre (n={n}):', {k: round(v / n, 3) for k, v in err.items()})


def screens():
    print('## Screens: blocked / enemyShotsDetected (all inferred shots) from metrics.jsonl')
    for run in ['v2a', 'v2b', 'v2c', 'vg1', 'vg2', 'vg3', 'eg1', 'vg4', 'vg5']:
        m = metrics(run)
        blocked, detected = sum(r['bulletsBlocked'] for r in m), sum(r['enemyShotsDetected'] for r in m)
        print(f'{run}: rounds={len(m)} wins={sum(r["won"] for r in m)} blocked={blocked}/{detected}={blocked / detected:.4f} skips={sum(r["skippedTurns"] for r in m)}')


if __name__ == '__main__':
    screens(); self_prediction(); label_bias(); per_round_guns('v2c'); ratings('vg1'); ratings('vg3'); autocorrelation(); by_power_and_errors()
    sys.exit(0)
