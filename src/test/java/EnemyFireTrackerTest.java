import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import java.nio.file.*;
import com.google.gson.JsonParser;

class EnemyFireTrackerTest {
    @Test void replaysRecordedPinnedEngineHitWindows() throws Exception {
        var report = JsonParser.parseString(Files.readString(Path.of(
                "benchmarks/foundation-scan-lag/engine-trace/trace.json"))).getAsJsonObject();
        assertEquals("verified", report.get("status").getAsString());
        assertEquals(4, report.getAsJsonArray("assertions").size());
        for (var assertion : report.getAsJsonArray("assertions")) {
            var row = assertion.getAsJsonObject();
            int turn = row.get("hitTurn").getAsInt(), enemy = row.get("enemy").getAsInt();
            double before = row.get("previousScan").getAsDouble();
            double current = row.get("sameTurnScan").getAsDouble();
            double after = row.get("nextTurnScan").getAsDouble();
            for (boolean callbackFirst : new boolean[]{false, true}) {
                EnemyFireTracker tracker = new EnemyFireTracker();
                Runnable hit = row.get("effect").getAsDouble() < 0
                        ? () -> tracker.recordDamage(enemy, turn, 2)
                        : () -> tracker.recordHitBonus(enemy, turn, 0.5);
                if (callbackFirst) hit.run();
                assertEquals(0, tracker.inferPower(enemy, turn - 1, turn, before, current));
                if (!callbackFirst) hit.run();
                assertEquals(0, tracker.inferPower(enemy, turn, turn + 1, current, after));
            }
        }
    }
    @Test void detectsFireCostInItsOwnTurnWithoutRoundingPower() {
        EnemyFireTracker tracker = new EnemyFireTracker();
        assertEquals(1.23, tracker.inferPower(7, 10, 11, 100, 98.77), 1e-9);
        assertEquals(0, tracker.inferPower(7, 10, 13, 100, 98));
        assertEquals(0, tracker.inferPower(7, 10, 10, 100, 98));
    }

    @Test void bulletDamageAppearsOnNextScanForBothCallbackOrders() {
        for (boolean hitCallbackFirst : new boolean[]{false, true}) {
            EnemyFireTracker tracker = new EnemyFireTracker();
            // Synthetic contract fixture; the separate official-engine probe records real events.
            if (hitCallbackFirst) tracker.recordDamage(7, 11, 2);
            assertEquals(0, tracker.inferPower(7, 10, 11, 100, 100));
            if (!hitCallbackFirst) tracker.recordDamage(7, 11, 2);
            assertEquals(0, tracker.inferPower(7, 11, 12, 100, 98));
            assertEquals(0, tracker.inferPower(7, 12, 13, 98, 98));
        }
    }

    @Test void currentFireAndPostScanBulletDamageAreNotConfused() {
        EnemyFireTracker tracker = new EnemyFireTracker();
        tracker.recordDamage(7, 11, 2);
        assertEquals(1.5, tracker.inferPower(7, 10, 11, 100, 98.5), 1e-9);
        // Next scan includes that hit plus a new power-0.7 shot.
        assertEquals(0.7, tracker.inferPower(7, 11, 12, 98.5, 95.8), 1e-9);
        assertEquals(0, tracker.inferPower(7, 12, 13, 95.8, 95.8));
    }

    @Test void hitBonusFirstAppearsNextTurnAndCanOverlapANewShot() {
        for (boolean bonusCallbackFirst : new boolean[]{false, true}) {
            EnemyFireTracker tracker = new EnemyFireTracker();
            if (bonusCallbackFirst) tracker.recordHitBonus(7, 11, 2);
            assertEquals(1, tracker.inferPower(7, 10, 11, 100, 99), 1e-9);
            if (!bonusCallbackFirst) tracker.recordHitBonus(7, 11, 2);
            assertEquals(2, tracker.inferPower(7, 11, 12, 99, 103), 1e-9);
            assertEquals(1, tracker.inferPower(7, 12, 13, 103, 102), 1e-9);
        }
    }

    @Test void smallHitBonusCannotManufactureAPowerOnePointFiveShot() {
        for (boolean callbackFirst : new boolean[]{false, true}) {
            for (double actualFire : new double[]{0, 0.7}) {
                EnemyFireTracker tracker = new EnemyFireTracker();
                if (callbackFirst) tracker.recordHitBonus(7, 11, 0.5);
                assertEquals(0, tracker.inferPower(7, 10, 11, 100, 100));
                if (!callbackFirst) tracker.recordHitBonus(7, 11, 0.5);
                assertEquals(actualFire, tracker.inferPower(7, 11, 12, 100, 101.5 - actualFire), 1e-9);
                assertEquals(0, tracker.inferPower(7, 12, 13, 101.5 - actualFire, 101.5 - actualFire));
            }
        }
    }

    @Test void bothBulletEffectsCombineOnlyAtTheirEffectiveScan() {
        EnemyFireTracker tracker = new EnemyFireTracker();
        tracker.recordDamage(7, 11, 4);
        tracker.recordHitBonus(7, 11, 1);
        assertEquals(0, tracker.inferPower(7, 10, 11, 100, 100));
        assertEquals(1.5, tracker.inferPower(7, 11, 12, 100, 97.5), 1e-9);
    }

    @Test void gapSuppressesAmbiguousFireAndConsumesOnlyVisibleCorrections() {
        EnemyFireTracker tracker = new EnemyFireTracker();
        tracker.recordDamage(7, 11, 2); // Effective scan 12, covered by the gap.
        tracker.recordDamage(7, 13, 2); // Effective scan 14, must survive gap consumption.
        assertEquals(0, tracker.inferPower(7, 10, 13, 100, 98));
        assertEquals(0, tracker.inferPower(7, 13, 14, 98, 96));
        assertEquals(1, tracker.inferPower(7, 14, 15, 96, 95), 1e-9);
    }

    @Test void correctionsAreIsolatedByBotAndFirstVisibleScan() {
        EnemyFireTracker tracker = new EnemyFireTracker();
        tracker.recordDamage(8, 11, 2);
        tracker.recordDamage(7, 10, 2); // Effective scan 11.
        assertEquals(0, tracker.inferPower(7, 10, 11, 100, 98));
        assertEquals(0, tracker.inferPower(8, 10, 11, 100, 100));
        assertEquals(0, tracker.inferPower(8, 11, 12, 100, 98));
        tracker.clear();
        assertEquals(2, tracker.inferPower(7, 12, 13, 98, 96), 1e-9);
    }
}
