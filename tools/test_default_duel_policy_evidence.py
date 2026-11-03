"""Durable screen integrity, cold endpoint counters and gate reproduction."""
import gzip
import hashlib
import json
from pathlib import Path
import struct
import unittest

from default_duel_policy_screen import decision, mechanism

ROOT = Path(__file__).resolve().parents[1] / 'benchmarks/default-duel-policy-screen'


class DefaultPolicyEvidenceTests(unittest.TestCase):
    def test_hashes_full_rounds_cold_endpoints_and_exact_decision(self):
        for name, digest in json.loads((ROOT / 'files.sha256.json').read_text()).items():
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), digest, name)
        report = json.loads((ROOT / 'screen.json').read_text())
        count = 0
        for group, values in [('retcon', report['blocks']), ('samples', report['samples'])]:
            for item in values:
                opponent = 'Retcon' if group == 'retcon' else item['opponent']
                key = str(item['seed']) if group == 'retcon' else opponent
                rounds = 5 if group == 'retcon' else 3
                for arm in ('previous', 'candidate'):
                    base = ROOT / 'raw' / group / key / arm
                    rows = [json.loads(row) for row in (base / (opponent + '-train') / 'metrics.jsonl').read_text().splitlines()]
                    self.assertEqual(len(rows), rounds)
                    self.assertEqual([r['totalRounds'] for r in rows], list(range(1, rounds + 1)))
                    self.assertTrue(all(r['mode'] == 'train' and not r['checkpointLoaded'] for r in rows))
                    model = gzip.decompress((base / 'model.dat.gz').read_bytes())
                    self.assertEqual(hashlib.sha256(model).hexdigest(), item[arm]['modelSHA256'])
                    self.assertEqual(struct.unpack('>7i', model[:28])[:6], (0x41495356, 1, 14, 32, 6, rounds))
                    if arm == 'candidate':
                        self.assertEqual(mechanism(rows), item[arm]['policyMechanism'])
                    count += len(rows)
        self.assertEqual(count, 48)
        self.assertEqual(decision(report['blocks'], report['samples']), report['decision'])


if __name__ == '__main__':
    unittest.main()
