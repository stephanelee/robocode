import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class CombatMathTest {
    @Test void clockwiseOrbitEastOfTargetHeadsSouth() {
        // At the east side of an enemy, bearing toward it is west (180).
        assertArrayEquals(new double[]{0, -8}, CombatMath.project(0, 0,
                CombatMath.orbitHeading(180, true), 8, 0, 1), 1e-9);
        assertArrayEquals(new double[]{0, 8}, CombatMath.project(0, 0,
                CombatMath.orbitHeading(180, false), 8, 0, 1), 1e-9);
    }

    @Test void delayedFreshScanCannotTrainAnAlreadyPassedTargetWave() {
        assertEquals(CombatMath.WaveObservation.BEFORE, CombatMath.observeTargetWave(70, 140, 14));
        assertEquals(CombatMath.WaveObservation.CROSSING, CombatMath.observeTargetWave(140, 140, 14));
        // No scans for ten more turns: this fresh position is too late to score.
        assertEquals(CombatMath.WaveObservation.PASSED, CombatMath.observeTargetWave(280, 140, 14));
        assertEquals(CombatMath.WaveObservation.PASSED, CombatMath.observeTargetWave(173, 140, 14));
    }

    @Test void cardinalHeadingsUseEastZeroCounterclockwiseAngles() {
        assertArrayEquals(new double[]{108, 100}, CombatMath.project(100, 100, 0, 8, 0, 1), 1e-9);
        assertArrayEquals(new double[]{100, 108}, CombatMath.project(100, 100, 90, 8, 0, 1), 1e-9);
        assertArrayEquals(new double[]{92, 100}, CombatMath.project(100, 100, 180, 8, 0, 1), 1e-9);
        assertArrayEquals(new double[]{100, 92}, CombatMath.project(100, 100, 270, 8, 0, 1), 1e-9);
        assertEquals(0, CombatMath.bearing(8, 0), 1e-9);
        assertEquals(90, CombatMath.bearing(0, 8), 1e-9);
        assertEquals(-90, CombatMath.bearing(0, -8), 1e-9);
    }

    @Test void circularPredictionIsContinuousWithLinearPrediction() {
        double[] linear = CombatMath.project(10, 20, 37, 8, 0, 15);
        assertArrayEquals(linear, CombatMath.project(10, 20, 37, 8, 1e-6, 15), 0.001);
        double radius = 8 / Math.toRadians(5);
        assertArrayEquals(new double[]{radius, radius}, CombatMath.project(0, 0, 0, 8, 5, 18), 1e-8);
        assertArrayEquals(new double[]{radius, -radius}, CombatMath.project(0, 0, 0, 8, -5, 18), 1e-8);
    }

    @Test void turnRatesNormalizeAcrossZeroAndUseElapsedTurns() {
        assertEquals(2, CombatMath.turnRate(350, 10, 10), 1e-9);
        assertEquals(-2, CombatMath.turnRate(10, 350, 10), 1e-9);
        assertEquals(0, CombatMath.turnRate(10, 90, 0), 1e-9);
    }

    @Test void emptyOrTiedTargetingBinsPreserveCentre() {
        double[] enemy = new double[47], opponent = new double[47];
        assertEquals(23, CombatMath.bestBin(enemy, opponent, 0.5));
        enemy[0] = 1; enemy[23] = 1;
        assertEquals(23, CombatMath.bestBin(enemy, opponent, 1));
        opponent[46] = 4;
        assertEquals(46, CombatMath.bestBin(enemy, opponent, 0.2));
    }
}
