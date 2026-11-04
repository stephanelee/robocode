#!/usr/bin/env python3
"""Materialize/verify the fixed gun-plan fixture using local Git objects; no network."""
import hashlib
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REV = '8989201545a09acebfaed9444aee263272bf01d8'
PATH = 'benchmarks/meridian-prior/raw/meridian-prior-v3-control-diamond-screen/game-2026-11-04-00-21-43.battle.gz'
SHA = 'f0c38e81e0965b29e5c71c79ec2f2eb7a1c18232dd275172d37b1dbf6251c2c1'
BASE = '809b880cdcee98d7946d05e6bcd30f623f604348'

def main():
    target = ROOT / 'local-evidence/control.battle.gz'
    raw = target.read_bytes() if target.exists() else subprocess.check_output(
        ['git', 'show', f'{REV}:{PATH}'], cwd=ROOT)
    if hashlib.sha256(raw).hexdigest() != SHA:
        raise SystemExit('Fixture hash mismatch; no file was overwritten')
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    original = subprocess.check_output(['git', 'show', f'{BASE}:experiments/Meridian/src/Meridian.java'], cwd=ROOT).decode()
    renamed = original.replace('class Meridian', 'class BaselineMeridian').replace(
        '    Meridian(){', '    BaselineMeridian(){').replace('new Meridian()', 'new BaselineMeridian()')
    if renamed != (ROOT / 'experiments/Meridian/test/baseline/BaselineMeridian.java').read_text():
        raise SystemExit('Baseline replay source differs beyond the declared class/constructor rename')
    print(f'Fixture verified: {SHA}; exact v3 replay source verified')

if __name__ == '__main__':
    main()
