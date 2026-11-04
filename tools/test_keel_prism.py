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
class CompletedComparison(unittest.TestCase):
 def test_all_original_bytes_and_sixty_round_metrics(self):
  import statistics
  root=pair.REG
  for path,h in json.loads((root/'SHA256.json').read_bytes()).items():self.assertEqual(h,hashlib.sha256((root/path).read_bytes()).hexdigest(),path)
  report=json.loads((root/'combined-report.json').read_bytes());self.assertEqual(6,len(report['games']));self.assertEqual(60,sum(g['rounds'] for g in report['games']))
  self.assertEqual([f'kp-utf8-{i:02}' for i in range(1,7)],[g['id'] for g in report['games']])
  for i,g in enumerate(report['games']):
   raw=root/('partial-utf8' if i==0 else 'completed-remaining')/'raw'/g['id'];actual=pair.analyze(raw,g)
   for k,v in actual.items():self.assertEqual(v,g[k],(g['id'],k))
  for n,summary in report['summary'].items():
   for key in ('score','firstPlaces','shots','hits','bulletDamage','shotEnergy','privateSkips'):self.assertEqual(sum(g['bots'][n][key] for g in report['games']),summary[key])
   self.assertEqual(summary['hits']/summary['shots'],summary['hitRate'])
   self.assertEqual(statistics.mean(g['bots'][n]['score']/sum(b['score'] for b in g['bots'].values()) for g in report['games']),summary['equalGameMeanShare'])

if __name__=='__main__':unittest.main()
