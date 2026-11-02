"""Boundary checks for the unchanged, preregistered promotion thresholds."""
import unittest

import compare
from test_compare import result


class Cycle2PromotionBounds(unittest.TestCase):
    @staticmethod
    def blocks(shares=(0.25, 0.25, 0.25), candidate_walls=12, candidate_wins=4):
        return [dict(previous=result(0.2, wins=5, walls=10),
                     candidate=result(share, wins=candidate_wins, walls=candidate_walls))
                for share in shares]

    def test_exact_mean_wall_and_wins_boundaries_are_inclusive(self):
        decision = compare.decision(self.blocks())
        self.assertEqual(decision['status'], 'retain')
        self.assertAlmostEqual(decision['candidate']['meanScoreShare']
                               - decision['previous']['meanScoreShare'], 0.05)
        self.assertAlmostEqual(decision['candidate']['wallsPer1000Turns'],
                               1.2 * decision['previous']['wallsPer1000Turns'])
        self.assertEqual(decision['candidate']['wins'], decision['previous']['wins'] - 3)

    def test_exact_maximum_single_block_loss_is_allowed_but_larger_loss_is_not(self):
        self.assertEqual(compare.decision(self.blocks((0.15, 0.35, 0.35)))['status'], 'retain')
        decision = compare.decision(self.blocks((0.149999, 0.35, 0.35)))
        self.assertFalse(decision['checks']['maximumBlockLoss'])
        self.assertEqual(decision['status'], 'reject')

    def test_zero_previous_wall_rate_requires_zero_candidate_collisions(self):
        blocks = self.blocks(candidate_walls=0)
        for block in blocks:
            block['previous']['metrics']['wallCollisions'] = 0
        self.assertEqual(compare.decision(blocks)['status'], 'retain')
        blocks[0]['candidate']['metrics']['wallCollisions'] = 1
        self.assertFalse(compare.decision(blocks)['checks']['wallRate'])

    def test_cycle2_plan_path_is_generic_and_cannot_modify_thresholds(self):
        from unittest.mock import patch
        path = compare.benchmark.ROOT / 'docs' / 'improvements' / 'cycle-2.md'
        commit = 'f34cffbf5bbe8c79fb55d1b4acba5b581ac802d6'
        with patch.object(compare.subprocess, 'check_output', return_value=b'cycle 2 immutable blob') as show:
            self.assertEqual(compare.registered_plan(path, commit), b'cycle 2 immutable blob')
            show.assert_called_once_with(['git', 'show', commit + ':docs/improvements/cycle-2.md'],
                                         cwd=compare.benchmark.ROOT)
        self.assertEqual(compare.THRESHOLDS['meanShareGain'], 0.05)
        self.assertEqual(compare.THRESHOLDS['maximumBlockShareLoss'], 0.05)
        self.assertEqual(compare.THRESHOLDS['maximumWallRateRatio'], 1.2)


if __name__ == '__main__':
    unittest.main()
