import math
import unittest

from aa_calibration import current_gate_pass_rate, required_blocks, summarize


def arm(share, wins=1, hits=10, shots=50):
    return dict(scoreShare=share, rounds=20, firstPlaces=wins, retconStats=dict(actualGun=dict(hits=hits, shots=shots)))


class AACalibrationTests(unittest.TestCase):
    def test_required_blocks_matches_normal_power_formula(self):
        # (1.645 + 0.842)^2 * (0.08 / 0.05)^2 = 15.83 -> 16 blocks.
        self.assertEqual(required_blocks(0.08), 16)
        self.assertEqual(required_blocks(0.05), 7)
        with self.assertRaises(ValueError):
            required_blocks(0)
        with self.assertRaises(ValueError):
            required_blocks(float('nan'))

    def test_current_gate_pass_rate_is_seeded_and_monotone(self):
        low = current_gate_pass_rate(0.07, 0.0, draws=20000)
        mid = current_gate_pass_rate(0.07, 0.05, draws=20000)
        high = current_gate_pass_rate(0.07, 0.10, draws=20000)
        self.assertEqual(low, current_gate_pass_rate(0.07, 0.0, draws=20000))
        self.assertTrue(low < mid < high)

    def test_summary_pairs_blocks_and_pools_arms(self):
        blocks = [dict(a=arm(.20), b=arm(.10)), dict(a=arm(.10), b=arm(.20)), dict(a=arm(.30), b=arm(.30))]
        result = summarize(blocks)
        self.assertAlmostEqual(result['meanDelta'], 0.0)
        self.assertAlmostEqual(result['deltaSD'], 0.1)
        self.assertAlmostEqual(result['deltaSE'], 0.1 / math.sqrt(3))
        self.assertAlmostEqual(result['pooledMeanShare'], 0.2)
        self.assertEqual(result['mechanism']['a']['retconShots'], 150)
        self.assertEqual(result['blocksPerArmFor80PctPowerAt5pp'], required_blocks(0.1))
        self.assertEqual(result['roundsPerArmFor80PctPowerAt5pp'], 20 * required_blocks(0.1))


if __name__ == '__main__':
    unittest.main()
