import unittest
import keelhelm_top4 as b
class TopFourContracts(unittest.TestCase):
 def test_schedule_covers_every_launch_slot_once_with_fixed_budget(self):
  games=b.schedule();self.assertEqual(80,sum(g['rounds']for g in games));self.assertEqual(4,len(games))
  for name in games[0]['order']:self.assertEqual([0,1,2,3],sorted(g['order'].index(name)for g in games))
 def test_total_score_rank_is_separate_from_equal_game_mean_share(self):
  games=[dict(official=dict(results=[dict(name='A',version='1',totalScore=a,survival=0,bulletDamage=0,firstPlaces=w),dict(name='B',version='1',totalScore=z,survival=0,bulletDamage=0,firstPlaces=1-w)]))for a,z,w in [(90,10,1),(1,9,0)]]
  rows=b.standings(games);self.assertEqual(['A','B'],[r['name']for r in rows]);self.assertEqual(91,rows[0]['totalScore']);self.assertEqual('1/2',rows[0]['meanScoreShareExact']);self.assertEqual(1,rows[0]['firstPlaces'])
if __name__=='__main__':unittest.main()
