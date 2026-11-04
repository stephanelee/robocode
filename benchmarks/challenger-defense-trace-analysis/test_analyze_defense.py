import unittest
from analyze_defense import analyze

def wave(owner=2,creation=1,x=100):
    return dict(type="wave",round=1,turn=creation,owner=owner,creationTurn=creation,originX=x,originY=300,
                power=1,referenceBearing=0,sign=1,context=0,aimOwn=[300,300,90,0])
def planner(turn=9,w=None):
    w=wave() if w is None else w
    return dict(type="planner",round=1,turn=turn,chosen=-1,scoresCCW_STOP_CW=[1,2,3],
                unsafeCCW=False,unsafeSTOP=False,unsafeCW=False,
                waves=[[w["creationTurn"],w["originX"],w["originY"],17,1,0,1,False,0]],
                rolloutTimesCCW_STOP_CW=[[1,2,-1,20]]*3)
def hit():
    return dict(type="enemyHit",round=1,turn=10,owner=2,creationTurn=1,matched=True,
                radialResidual=0,direction=0)

class BoundaryTests(unittest.TestCase):
    def test_same_turn_and_later_sequence_planner_cannot_be_pre_hit_evidence(self):
        result=analyze([wave(),planner(8),planner(10),hit(),planner(9)])
        self.assertEqual(8,result["hits"][0]["plannerTurn"])
        self.assertEqual(2,result["hits"][0]["decisionLag"])

    def test_ambiguous_hit_wave_key_is_unknown_not_guessed(self):
        result=analyze([wave(),wave(),planner(),hit()])
        self.assertEqual({"ambiguous":1},result["waveJoin"])
        self.assertNotIn("matchedWaveSlot",result["hits"][0])

    def test_ownerless_slot_with_two_possible_identities_is_unknown(self):
        result=analyze([wave(),wave(owner=3),planner(),hit()])
        self.assertEqual("ambiguous_identity",result["hits"][0]["slotMissingReason"])
        self.assertIsNone(result["hits"][0]["matchedWaveSlot"])

    def test_wave_inferred_after_decision_is_not_a_planner_omission(self):
        other=wave(creation=2,x=200)
        result=analyze([other,planner(8,other),wave(),hit()])
        self.assertEqual("not_yet_inferred",result["hits"][0]["slotMissingReason"])
        self.assertFalse(result["hits"][0]["waveInferredBeforeDecision"])

if __name__=="__main__":unittest.main()
