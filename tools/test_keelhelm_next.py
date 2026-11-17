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

class CandidateProjectContracts(unittest.TestCase):
 def make_project(self,root,version,jar):
  import json
  (root/'bot').mkdir(parents=True)
  (root/'bot/KeelHelm.json').write_text(json.dumps(dict(name='KeelHelm',version=version)))
  (root/'bot/KeelHelm.sh').write_text('#!/bin/sh\nexit 0\n')
  (root/'KeelHelm.jar').write_bytes(jar)
 def test_frozen_candidate_path_is_used_without_changing_control(self):
  import tempfile
  from pathlib import Path
  import keelhelm_melee as core
  with tempfile.TemporaryDirectory() as temp:
   base=Path(temp);candidate=base/'evidence-project';evidence=base/'study'
   self.make_project(candidate,'0.4.0',b'candidate')
   self.make_project(evidence/'control-source','0.2.1',b'control')
   freeze=dict(candidateProject=str(candidate))
   for arm,expected in [('candidate',b'candidate'),('control',b'control')]:
    run=base/arm;run.mkdir()
    dirs,ids=core.stage_bots(dict(order=['KeelHelm'],arm=arm,scenario='melee'),base,evidence,freeze,run)
    self.assertEqual(expected,(Path(dirs[0])/'KeelHelm.jar').read_bytes())
    self.assertEqual([('KeelHelm','0.4.0'if arm=='candidate'else'0.2.1')],ids)
 def test_older_freezes_keep_their_legacy_project_path(self):
  import tempfile
  from pathlib import Path
  from unittest.mock import patch
  import keelhelm_melee as core
  with tempfile.TemporaryDirectory() as temp:
   base=Path(temp);self.make_project(base/'experiments/KeelHelm','0.2.1',b'legacy');run=base/'run';run.mkdir()
   with patch.object(core,'ROOT',base):
    dirs,ids=core.stage_bots(dict(order=['KeelHelm'],arm='candidate',scenario='melee'),base,base,{},run)
   self.assertEqual(b'legacy',(Path(dirs[0])/'KeelHelm.jar').read_bytes())

if __name__=='__main__':unittest.main()
