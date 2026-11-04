import unittest,json,hashlib
from pathlib import Path
import keel_prism as pair
class FrozenContinuation(unittest.TestCase):
 def test_completed_game_combines_without_round_key_collision(self):
  arc=pair.REG/'partial-utf8'; plan=json.loads((arc/'plan.json').read_bytes()); game=plan['games'][0]
  row=dict(**game,status='valid',**pair.analyze(arc/'raw'/game['id'],game))
  self.assertEqual(10,row['rounds']);self.assertEqual(10,len(row['perRound']))
 def test_original_partial_bytes_and_exact_unplayed_registration(self):
  arc=pair.REG/'partial-utf8'
  for path,h in json.loads((arc/'SHA256.json').read_bytes()).items():self.assertEqual(h,hashlib.sha256((arc/path).read_bytes()).hexdigest(),path)
  old=json.loads((arc/'plan.json').read_bytes());new=json.loads((pair.REG/'plan.json').read_bytes())
  self.assertEqual(old['games'][1:],new['games']);self.assertEqual(50,sum(g['rounds'] for g in new['games']))
  original=json.loads((arc/'raw/report.json').read_bytes());self.assertEqual('invalid; stopped without replacement',original['decision'])
if __name__=='__main__':unittest.main()