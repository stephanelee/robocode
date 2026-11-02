import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class EnemyFireTrackerTest {
    @Test void detectsConsecutiveScanShotWithoutRoundingPower() {
        EnemyFireTracker tracker = new EnemyFireTracker();
        assertEquals(1.23, tracker.inferPower(7, 10, 11, 100, 98.77), 1e-9);
        assertEquals(0, tracker.inferPower(7, 10, 13, 100, 98));
        assertEquals(0, tracker.inferPower(7, 10, 10, 100, 98));
    }

    @Test void ourDamageDoesNotBecomeEnemyFire() {
        EnemyFireTracker tracker = new EnemyFireTracker();
        tracker.recordDamage(7, 11, 2);
        assertEquals(0, tracker.inferPower(7, 10, 11, 100, 98));
        tracker.recordDamage(7, 12, 10);
        assertEquals(1.5, tracker.inferPower(7, 11, 12, 98, 86.5), 1e-9);
    }

    @Test void hitEnergyBonusIsAccountedForAndCorrectionsAreConsumed() {
        EnemyFireTracker tracker = new EnemyFireTracker();
        tracker.recordHitBonus(7, 11, 2);
        assertEquals(2, tracker.inferPower(7, 10, 11, 100, 104), 1e-9);
        assertEquals(1, tracker.inferPower(7, 11, 12, 104, 103), 1e-9);
    }

    @Test void correctionsAreIsolatedByBotAndScanInterval() {
        EnemyFireTracker tracker = new EnemyFireTracker();
        tracker.recordDamage(8, 11, 2);
        tracker.recordDamage(7, 10, 2);
        assertEquals(2, tracker.inferPower(7, 10, 11, 100, 98), 1e-9);
        tracker.recordDamage(7, 13, 2);
        assertEquals(1, tracker.inferPower(7, 11, 12, 98, 97), 1e-9);
        assertEquals(0, tracker.inferPower(7, 12, 13, 97, 95));
    }
}
