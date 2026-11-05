import unittest
import keelhelm_next as n

class NextContracts(unittest.TestCase):
 def test_new_policies_have_distinct_disabled_bits_and_exposure_keys(self):
  self.assertEqual([8,16,32,64],[n.POLICIES[i][1] for i in range(1,5)])
 def test_a_score_gain_without_observed_action_is_not_promotable(self):
  games=[]
  for scenario in ('melee','Diamond Port','Retcon','Walls','SpinBot'):
   for arm,score in [('control',25),('candidate',30)]:
    games.append(dict(scenario=scenario,arm=arm,helm=dict(totalScore=score,firstPlaces=1),official=dict(results=[dict(totalScore=score),dict(totalScore=100-score)]),telemetry=['1 stats riskDifferences=0 radarRevisits=0 obstructedShots=0 transferredScans=0']))
  for stage in range(1,5):
   self.assertEqual('reject',n.summary(games,stage)['decision'])
   key=n.POLICIES[stage][2];games[1]['telemetry']=[f'1 stats {key}=1']
   self.assertEqual('accept',n.summary(games,stage)['decision'])
   games[1]['telemetry']=['1 stats riskDifferences=0 radarRevisits=0 obstructedShots=0 transferredScans=0']
if __name__=='__main__':unittest.main()
