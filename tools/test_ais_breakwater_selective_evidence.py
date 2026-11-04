"""Durable V2 preservation/rejection checks; no acquisition or controller execution."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
EVIDENCE=ROOT/'benchmarks/ais-breakwater-selective-transfer-v2'

class SelectiveTransferEvidenceTests(unittest.TestCase):
    def test_all_retained_file_hashes(self):
        manifest=json.loads((EVIDENCE/'files.sha256.json').read_bytes())
        for name,entry in manifest.items():
            data=(EVIDENCE/name).read_bytes()
            self.assertEqual(len(data),entry['bytes'],name)
            self.assertEqual(hashlib.sha256(data).hexdigest(),entry['sha256'],name)
    def test_frozen_original_payload_models_and_decision_reproduce_without_games(self):
        result=subprocess.run([sys.executable,str(EVIDENCE/'reproduce.py')],cwd=ROOT,text=True,capture_output=True,timeout=60)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertIn('956 original files/550 unique payloads',result.stdout)
        self.assertIn('frozen V2 rejection',result.stdout)

if __name__=='__main__':unittest.main()
