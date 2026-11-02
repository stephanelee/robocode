"""Independent numeric/geometry fixtures for the bounded offline diagnostic."""
import math
import unittest

from dual_aim_diagnostic import (State, cases, clearance, constrain, linear_aim,
                                 next_speed, sector_distance, step, trace)


class MovementFixtures(unittest.TestCase):
    def test_server_speed_regressions(self):
        # v1.4.0 math.kt literal behavior, including probable upstream bug.
        for current, target, expected in ((0, 8, 1), (7, 8, 8), (8, 4, 6),
                                           (8, 0, 0), (8, -8, 3), (1, -8, -.5),
                                           (-8, 8, -3), (-8, 0, 0)):
            with self.subTest(current=current, target=target):
                self.assertEqual(next_speed(current, target), expected)

    def test_move_before_turn_and_new_speed_turn_limit(self):
        state, wall = step(State(500, 500, 0, 7), (8, 10))
        self.assertEqual(state, State(508, 500, 4, 8))
        self.assertFalse(wall)
        state, _ = step(State(500, 500, 0, 8), (0, 10))
        self.assertEqual(state, State(500, 500, 10, 0))

    def test_first_wall_contact_not_independent_coordinate_clamp(self):
        self.assertEqual(constrain(970, 300, 990, 320), (982, 312))
        self.assertEqual(constrain(970, 970, 990, 1010), (976, 982))
        state, hit = step(State(980, 300, 45, 8), (8, 10))
        self.assertTrue(hit)
        self.assertAlmostEqual(state.x, 982)
        self.assertAlmostEqual(state.y, 302)
        self.assertEqual(state.speed, 0)
        self.assertEqual(state.heading, 49)  # turn limit uses pre-impact new speed

    def test_reaction_delay_preserves_pending_motion(self):
        path = trace(State(500, 500, 0, 8), (0, 10), 1, 2)
        self.assertEqual(path[0][0], State(508, 500, 0, 8))
        self.assertEqual(path[1][0], State(508, 500, 10, 0))

    def test_linear_retcon_iteration_not_headon_time(self):
        aim, turns, point = linear_aim((100, 300), State(550, 300, 90, 8), 16.4)
        self.assertEqual(turns, 32)
        self.assertAlmostEqual(point[1], 556)
        self.assertAlmostEqual(aim, math.degrees(math.atan2(256, 450)))
        _, _, point = linear_aim((100, 300), State(550, 980, 90, 8), 16.4)
        self.assertEqual(point[1], 982)


class SweptCorridorFixtures(unittest.TestCase):
    def test_segment_crossing_is_hit_when_endpoints_miss(self):
        # Legal forward bullet segment: neither endpoint is within radius18.
        self.assertEqual(sector_distance(50, 0, 0, 0, 20, 80), 0)
        self.assertGreater(abs(50 - 20), 18)
        self.assertGreater(abs(50 - 80), 18)

    def test_interval_interior_and_wraparound(self):
        self.assertAlmostEqual(sector_distance(50, 0, 0, 10, 40, 60), 0)
        self.assertAlmostEqual(sector_distance(-50, 0, 179, 3, 40, 60), 0)
        self.assertAlmostEqual(sector_distance(50, 0, 90, 0, 40, 60), math.hypot(50, 40))

    def test_analytic_distance_matches_dense_independent_segment_oracle(self):
        for x, y, center, half, near, far in ((45, 12, 0, 20, 40, 60),
                                             (-10, 90, 179, 30, 15, 85),
                                             (10, -30, 90, 180, 50, 70)):
            distances = []
            for i in range(2001):
                a = math.radians(center - half + 2 * half * i / 2000)
                along = max(near, min(far, x * math.cos(a) + y * math.sin(a)))
                distances.append(math.hypot(x - along * math.cos(a), y - along * math.sin(a)))
            actual = sector_distance(x, y, center, half, near, far)
            self.assertLessEqual(actual, min(distances) + 1e-9)
            self.assertAlmostEqual(actual, min(distances), delta=.02)

    def test_creation_turn_sweep_is_advanced_not_origin_segment(self):
        # Bullet tick=1 => sweep[11,22] against post-move target center40.
        margin, limiting = clearance([(State(40, 0, 0, 0), False)], (0, 0), [("head-on", 11, 0, 0)])
        self.assertEqual(margin, 0)
        self.assertEqual(limiting[0], 1)

    def test_wider_corridor_cannot_increase_clearance(self):
        for half in (0, 2, 5, 15):
            self.assertGreaterEqual(sector_distance(100, 25, 0, half, 90, 110),
                                    sector_distance(100, 25, 0, half + 1, 90, 110))

    def test_versioned_withheld_cases_are_disjoint(self):
        development = list(cases("development"))
        withheld = list(cases("withheld"))
        self.assertEqual(len(development), 816)
        self.assertEqual(len(withheld), 90)
        self.assertFalse({r[:4] for r in development} & {r[:4] for r in withheld})
        self.assertEqual({r[1] for r in development}, set(range(-8, 9)))


if __name__ == "__main__":
    unittest.main()
