import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class EnemyWaveTest {
    @Test void evaluationLeavesSurfDangerFrozen() {
        EnemyWave wave = new EnemyWave(0, 0, 14, 0, 10, 7);
        double[] danger = new double[47];
        wave.learnHit(danger, 140, 0, false);
        assertEquals(0, java.util.Arrays.stream(danger).sum());
        wave.learnHit(danger, 140, 0, true);
        assertEquals(1, danger[23]);
        assertEquals(1, java.util.Arrays.stream(danger).sum());
    }

    @Test void hitMatchesOnlyShooterPowerAndArrivalRadius() {
        EnemyWave wave = new EnemyWave(0, 0, 14, 0, 10, 7);
        assertEquals(0, wave.matchError(7, 2, 140, 0, 20), 1e-9);
        assertEquals(Double.POSITIVE_INFINITY, wave.matchError(8, 2, 140, 0, 20));
        assertEquals(Double.POSITIVE_INFINITY, wave.matchError(7, 1, 140, 0, 20));
        assertEquals(Double.POSITIVE_INFINITY, wave.matchError(7, 2, 300, 0, 20));
    }

    @Test void dangerBinUsesActualHitBearingAndWrapsAngles() {
        EnemyWave wave = new EnemyWave(0, 0, 14, 0, 10, 7);
        assertEquals(23, wave.hitBin(140, 0, 47));
        assertTrue(wave.hitBin(140, 30, 47) > 23);
        assertTrue(wave.hitBin(140, -30, 47) < 23);
        EnemyWave wrapped = new EnemyWave(0, 0, 14, 179, 10, 7);
        assertTrue(wrapped.hitBin(-140, -2, 47) < 27);
    }
}
