import org.junit.jupiter.api.Test;
import java.lang.reflect.Method;
import java.nio.file.Path;
import java.util.HashMap;
import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

class DuelCompositionTest {
    @Test void radarCrossesBearingAndReacquiresWithoutStallingAtZero() {
        assertEquals(8, DuelRadar.turn(400, 300, 0, 600, 300, 0, 1));
        assertEquals(-8, DuelRadar.turn(400, 300, 0, 600, 300, 1, -1));
        assertEquals(12, DuelRadar.turn(400, 300, 358, 600, 300, 0, 1));
        assertEquals(-12, DuelRadar.turn(400, 300, 2, 600, 300, 0, 1));
        assertEquals(45, DuelRadar.turn(400, 300, 0, 600, 300, 3, 1));
        assertEquals(-45, DuelRadar.turn(400, 300, 0, 600, 300, 40, -1));
    }

    @Test void defaultDuelFullyDecouplesRadarButOtherModesKeepLegacyPath() throws Exception {
        Method radar = AISurvivor.class.getDeclaredMethod("updateRadar");
        radar.setAccessible(true);
        for (BotConfig.Mode mode : BotConfig.Mode.values()) {
            AISurvivor bot = mock(AISurvivor.class, CALLS_REAL_METHODS);
            TargetingCharacterizationTest.set(bot, "config", new BotConfig(mode, 3, Path.of("unused"), 0, null));
            TargetingCharacterizationTest.set(bot, "enemies", new HashMap<>());
            TargetingCharacterizationTest.set(bot, "radarSign", 1);
            doReturn(1).when(bot).getEnemyCount();
            doNothing().when(bot).setAdjustRadarForBodyTurn(anyBoolean());
            doNothing().when(bot).setRadarTurnRate(anyDouble());
            radar.invoke(bot);
            verify(bot).setAdjustRadarForBodyTurn(mode == BotConfig.Mode.TRAIN);
            verify(bot).setRadarTurnRate(45);
        }
    }

    @Test void bulletDamageBecomesVisibleNextScanInBothCallbackOrders() {
        for (boolean callbackFirst : new boolean[]{false, true}) {
            EnemyFireTracker tracker = new EnemyFireTracker();
            if (callbackFirst) tracker.recordBulletDamage(7, 11, 2);
            assertEquals(0, tracker.inferPower(7, 10, 11, 100, 100));
            if (!callbackFirst) tracker.recordBulletDamage(7, 11, 2);
            assertEquals(1.5, tracker.inferPower(7, 11, 12, 100, 96.5), 1e-9);
            assertEquals(0, tracker.inferPower(7, 12, 13, 96.5, 96.5));
        }
    }

    @Test void nextScanHitBonusDoesNotEraseSameScanFireOrLeakAcrossRound() {
        EnemyFireTracker tracker = new EnemyFireTracker();
        tracker.recordBulletHitBonus(7, 11, .5);
        assertEquals(2, tracker.inferPower(7, 10, 11, 100, 98), 1e-9);
        assertEquals(1, tracker.inferPower(7, 11, 12, 98, 98.5), 1e-9);
        tracker.recordBulletDamage(7, 13, 2);
        tracker.clear();
        assertEquals(2, tracker.inferPower(7, 13, 14, 100, 98), 1e-9);
    }
}
