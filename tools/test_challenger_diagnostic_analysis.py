import math
import unittest
from challenger_diagnostic_analysis import angle_delta, passage_geometry, segment_intersects_body, distribution, analyze


class GeometryTests(unittest.TestCase):
    def test_wrapped_angular_error_does_not_report_358_degrees(self):
        self.assertEqual(2, angle_delta(1, 359))
        self.assertEqual(-2, angle_delta(359, 1))

    def test_confirmed_origin_shift_is_measurable_without_changing_the_prediction(self):
        decision = passage_geometry(0, 0, 100, 0, 0)
        actual = passage_geometry(0, 8, 100, 0, 0)
        self.assertEqual(0, decision["absoluteAngularError"])
        self.assertAlmostEqual(math.degrees(math.atan2(8, 100)), actual["absoluteAngularError"])

    def test_circle_proxy_containment_is_not_a_finite_bullet_segment_hit(self):
        self.assertTrue(passage_geometry(0, 0, 100, 0, 0)["contained"])
        self.assertFalse(segment_intersects_body(0, 0, 0, 40, 60, 100, 0))
        self.assertTrue(segment_intersects_body(0, 0, 0, 84, 98, 100, 0))

    def test_body_edge_and_missed_scan_are_not_silently_zero_errors(self):
        edge = math.degrees(math.asin(18/100))
        self.assertTrue(passage_geometry(0, 0, 100, 0, edge)["contained"])
        self.assertFalse(passage_geometry(0, 0, 100, 0, edge+.01)["contained"])
        self.assertEqual({"count": 0, "mean": None, "median": None, "p90": None}, distribution([None, float("nan")]))

    def test_actual_hit_ids_are_round_scoped_and_virtual_hits_are_not_actual_hits(self):
        shots = [dict(type="shot", round=r, shotId=7, originX=0, originY=0, power=.6,
                      direction=0, aimDirection=1, decisionGunDirection=359, model=1, learned=False)
                 for r in [1, 2]]
        rows = shots + [dict(type="passage", round=1, shotId=7, x=100, y=0, predictedDirection=0),
                       dict(type="passage", round=2, shotId=None, x=100, y=0),
                       dict(type="virtual", round=2, shotId=7, model=1, hit=True),
                       dict(type="hit", round=1, shotId=7, damage=2.4),
                       dict(type="round", round=2, shots=1, hits=0, truncated=False, failed=False, droppedRecords=0)]
        result = analyze(rows)
        self.assertEqual(1, result["joinedPassages"])
        self.assertEqual(1, result["unjoinedPassages"])
        self.assertEqual(1, result["distinctMatchedActualHitShots"])
        self.assertEqual(1, result["selectedLegacyVirtualVsActual"]["virtualHitNoActualHit"])
        self.assertEqual(0, result["absoluteRawGFError"]["count"])
        self.assertAlmostEqual(1, result["decisionToCreationGunPhaseDegrees"]["mean"])

    def test_duplicate_confirmed_shot_identity_is_rejected(self):
        shot = dict(type="shot", round=1, shotId=0)
        with self.assertRaises(ValueError): analyze([shot, shot])

    def test_missing_or_truncated_terminal_coverage_cannot_turn_absent_hit_into_miss(self):
        rows=[dict(type="shot", round=r, shotId=1, model=0) for r in [1,2,3]]
        rows += [dict(type="virtual", round=r, shotId=1, model=0, hit=True) for r in [1,2,3]]
        rows += [dict(type="round", round=2, shots=1, hits=0, truncated=True, failed=False, droppedRecords=1),
                 dict(type="round", round=3, shots=2, hits=0, truncated=False, failed=False, droppedRecords=0)]
        result=analyze(rows)
        self.assertEqual(3,result["selectedLegacyVirtualVsActual"]["unknownActualOutcome"])
        self.assertEqual(0,result["selectedLegacyVirtualVsActual"]["virtualHitNoActualHit"])

    def test_bullet_collision_is_preserved_as_an_explanation_not_an_aiming_miss(self):
        rows=[dict(type="shot",round=1,shotId=1,originX=0,originY=0,direction=0),
              dict(type="passage",round=1,shotId=1,x=100,y=0),
              dict(type="removal",round=1,shotId=1,reason="bulletCollision"),
              dict(type="round",round=1,shots=1,hits=0,truncated=False,failed=False,droppedRecords=0)]
        result=analyze(rows)
        self.assertEqual({"contained:bulletCollision":1},result["passageActualRayOutcomes"])


if __name__ == "__main__":
    unittest.main()
