import json
import math
from pathlib import Path
import tempfile
import unittest
import diamond_surf_experiment as e

def games(stage,share=.6,wins=15,skip=0):
    return [dict(**g,scoreShare=share,battle=dict(firstPlaces=wins),metrics=dict(skippedTurns=skip)) for g in e.layout(stage)]

class SurfExperimentTests(unittest.TestCase):
    def test_fixed20_pilot_is_only_progress_and_confirmation_is_fresh60(self):
        self.assertEqual((20,60),tuple(sum(g['rounds'] for g in e.layout(s)) for s in ('pilot','confirmation')))
        self.assertEqual('advance',e.decision(games('pilot',.200001,1),'pilot')['status'])
        for share,wins,skip in ((.2,1,0),(.5,0,0),(.8,20,1)):
            self.assertEqual('reject',e.decision(games('pilot',share,wins,skip),'pilot')['status'])
        self.assertTrue(set(g['seed'] for g in e.layout('pilot')).isdisjoint(g['seed'] for g in e.layout('confirmation')))

    def test_confirmation_requires_all_three_winning_guards_and_zero_own_skips(self):
        self.assertEqual('confirmed',e.decision(games('confirmation'),'confirmation')['status'])
        for data in (games('confirmation',.5,15),games('confirmation',.7,10),games('confirmation',.7,15,1)):
            self.assertEqual('reject',e.decision(data,'confirmation')['status'])
        data=games('confirmation',.9,15);data[1]['scoreShare']=data[2]['scoreShare']=.4
        self.assertFalse(e.decision(data,'confirmation')['checks']['majorityGames'])
        with self.assertRaises(ValueError):e.decision(games('confirmation')[:-1],'confirmation')

    def test_own_endpoint_is_not_an_ais_checkpoint_and_rejects_invalid_counts(self):
        state=dict(schemaVersion=1,name=e.NAME,version=e.VERSION,seed=17,roundsCompleted=20,checkpointLoaded=False,
                   gunAttempts=[3,4,5],gunHits=[1,2,3],contextMemory=[],contextMemorySize=0,contextAccepted=0,dangerBins=[[0]*47 for _ in range(4)],inferredWavesTotal=0,matchedHitsTotal=0)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'model.json';path.write_text(json.dumps(state))
            self.assertEqual(state,e.endpoint(path,20,17)['state'])
            state['gunHits'][0]=4;path.write_text(json.dumps(state))
            with self.assertRaises(ValueError):e.endpoint(path,20,17)

    def test_context_snapshot_rejects_future_labels_and_unbounded_memory(self):
        sample=dict(features=[.2,.3,-.2,0,-.1,1,.5],guessFactor=.4,aimTurn=10,shotTurn=11,observedTurn=20,
                    createdSequence=10,observedSequence=20,insertionSequence=1,generation=1,originX=0,originY=0,
                    referenceX=100,referenceY=0,observedX=100*math.cos(math.radians(12)),observedY=100*math.sin(math.radians(12)),sign=1,escape=30)
        state=dict(schemaVersion=1,name=e.NAME,version=e.VERSION,seed=17,roundsCompleted=20,checkpointLoaded=False,
                   gunAttempts=[3,4,5],gunHits=[1,2,3],contextMemory=[sample],contextMemorySize=1,contextAccepted=1,dangerBins=[[0]*47 for _ in range(4)],inferredWavesTotal=0,matchedHitsTotal=0)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'model.json';path.write_text(json.dumps(state));self.assertEqual(1,e.endpoint(path,20,17)['state']['contextMemorySize'])
            sample['observedSequence']=10;path.write_text(json.dumps(state))
            with self.assertRaises(ValueError):e.endpoint(path,20,17)
            sample['observedSequence']=20;sample['features'][0]=1.1;path.write_text(json.dumps(state))
            with self.assertRaises(ValueError):e.endpoint(path,20,17)

if __name__=='__main__':unittest.main()
