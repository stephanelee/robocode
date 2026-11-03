import json,tempfile,unittest
from pathlib import Path
import challenger_diagnostic_experiment as e
class DiagnosticCaptureTests(unittest.TestCase):
    def test_only_one_fixed_five_round_capture_without_score_gate(self):
        self.assertEqual([dict(gameId='diamond-diagnostic-v5-g01',stage='diagnostic',seed=20263501,rounds=5)],e.layout())
        with self.assertRaises(ValueError):e.layout('confirmation')
        self.assertFalse(hasattr(e,'decision'))
    def test_cap_and_coverage_are_checked_and_drops_remain_visible(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'trace';s=Path(tmp)/'status'
            p.write_bytes(b'{"type":"round","round":1,"turn":20}\n')
            status=dict(schemaVersion=1,roundsCompleted=1,records=1,bytes=p.stat().st_size,droppedRecords=3,failed=False)
            s.write_text(json.dumps(status));self.assertEqual(3,e.check_diagnostic(p,s,1)[1]['droppedRecords'])
            status['records']=2;s.write_text(json.dumps(status))
            with self.assertRaises(ValueError):e.check_diagnostic(p,s,1)
if __name__=='__main__':unittest.main()
