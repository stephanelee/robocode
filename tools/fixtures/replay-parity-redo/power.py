"""Prospective score-gate power model; reads spent planning evidence, plays no games."""
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'tools'))
from decision_rule_analysis import acceptance

old = json.loads((ROOT / 'benchmarks/replay-harness-parity/raw/experiment.json').read_bytes())
deltas = [b['candidate']['scoreShare'] - b['previous']['scoreShare'] for b in old['primaryBlocks']]
sd = statistics.stdev(deltas)
rows = []
for effect in (.05, 0.0, .05 + statistics.mean(deltas)):
    rows.append(dict(netEffectToMargin=effect, blocks=30, pilotSD=sd,
        fixedSDScoreAcceptance=acceptance(30, effect, sd, 't', None, 50000, 20262800),
        pilotSDUncertaintyScoreAcceptance=acceptance(30, effect, sd, 't', None, 50000, 20262800, pilot_df=7)))
print(json.dumps(rows, indent=2))
