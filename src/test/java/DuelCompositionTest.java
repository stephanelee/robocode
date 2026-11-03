import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import java.lang.reflect.Method;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;
import java.util.ArrayList;
import dev.robocode.tankroyale.botapi.BulletState;
import dev.robocode.tankroyale.botapi.events.BulletFiredEvent;
import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

class DuelCompositionTest {
    @Test void disabledZeroEnergyTargetReachesRealGunAndActualFireRequest() throws Exception {
        AISurvivor bot = mock(AISurvivor.class, CALLS_REAL_METHODS);
        var config = new BotConfig(BotConfig.Mode.TRAIN, 5, Path.of("unused"), 0, null);
        TargetingCharacterizationTest.set(bot, "config", config);
        TargetingCharacterizationTest.set(bot, "duelGun", new DuelGun());
        TargetingCharacterizationTest.set(bot, "agent", new DQNAgent(14, 6, config));
        TargetingCharacterizationTest.set(bot, "gfStats", new HashMap<>(Map.of(7, new double[9][47])));
        var constructor = Class.forName("AISurvivor$Enemy").getDeclaredConstructors()[0];
        constructor.setAccessible(true);
        Object target = constructor.newInstance(7, 600., 300., 0., 0., 0., 0., 0., 0., 100);
        TargetingCharacterizationTest.set(bot, "enemies", new HashMap<>(Map.of(7, target)));
        doReturn(1).when(bot).getEnemyCount(); doReturn(100).when(bot).getTurnNumber();
        doReturn(400.).when(bot).getX(); doReturn(300.).when(bot).getY();
        doReturn(800).when(bot).getArenaWidth(); doReturn(600).when(bot).getArenaHeight();
        doReturn(10.).when(bot).getEnergy();
        doReturn(0.).when(bot).getGunDirection(); doReturn(0.).when(bot).getGunHeat();
        doAnswer(call -> 20 - 3 * (double) call.getArgument(0)).when(bot).calcBulletSpeed(anyDouble());
        doNothing().when(bot).setGunTurnRate(anyDouble()); doReturn(true).when(bot).setFire(.1);
        Method aim = AISurvivor.class.getDeclaredMethod("aimGF"); aim.setAccessible(true); aim.invoke(bot);
        verify(bot).setFire(.1);
        assertEquals(true, TargetingCharacterizationTest.get(bot, "pendingDuelShot"));
    }

    @Test void actualAimWiringUsesLegalSnapshotAndOnlyConfirmedShotsReachGun(@TempDir Path temporary) throws Exception {
        AISurvivor bot = mock(AISurvivor.class, CALLS_REAL_METHODS);
        DuelGun gun = mock(DuelGun.class);
        var config = new BotConfig(BotConfig.Mode.TRAIN, 3, Path.of("unused"), 0, null);
        TargetingCharacterizationTest.set(bot, "config", config);
        TargetingCharacterizationTest.set(bot, "duelGun", gun);
        TargetingCharacterizationTest.set(bot, "metrics", new RoundMetrics());
        TargetingCharacterizationTest.set(bot, "scanRecorder", LegalScanRecorder.fromEnvironment(Map.of()));
        var agent = new DQNAgent(14, 6, config);
        agent.classGF[0][5][34] = 4; // Non-central GF proves the live fourth candidate is not linear.
        double[][] profile = new double[9][47];
        TargetingCharacterizationTest.set(bot, "agent", agent);
        TargetingCharacterizationTest.set(bot, "gfStats", new HashMap<>(Map.of(7, profile)));
        var waves = new ArrayList<>();
        TargetingCharacterizationTest.set(bot, "targetingWaves", waves);
        var constructor = Class.forName("AISurvivor$Enemy").getDeclaredConstructors()[0];
        constructor.setAccessible(true);
        Object enemy = constructor.newInstance(7, 600., 300., 90., 8., 20., 8., 20., 4., 100);
        TargetingCharacterizationTest.set(bot, "enemies", new HashMap<>(Map.of(7, enemy)));
        doReturn(1).when(bot).getEnemyCount(); doReturn(100).when(bot).getTurnNumber();
        doReturn(400.).when(bot).getX(); doReturn(300.).when(bot).getY();
        doReturn(800).when(bot).getArenaWidth(); doReturn(600).when(bot).getArenaHeight();
        doReturn(100.).when(bot).getEnergy();
        doReturn(0.).when(bot).getGunDirection(); doReturn(0.).when(bot).getGunHeat();
        doNothing().when(bot).setGunTurnRate(anyDouble());
        doAnswer(call -> 20 - 3 * (double) call.getArgument(0)).when(bot).calcBulletSpeed(anyDouble());
        doReturn(true).when(bot).setFire(1.8);
        when(gun.decide(any())).thenReturn(new DuelGun.Plan(30, 1.8, true, true, "current", 0, 0, 0));
        Method aim = AISurvivor.class.getDeclaredMethod("aimGF"); aim.setAccessible(true); aim.invoke(bot);
        verify(bot).setGunTurnRate(20);
        verify(bot).setFire(1.8);
        verify(gun, never()).onFired(anyInt(), anyDouble(), anyDouble(), anyDouble());
        var input = org.mockito.ArgumentCaptor.forClass(DuelGun.Input.class);
        verify(gun).decide(input.capture());
        assertEquals(new DuelGun.Scan(7, 100, 600, 300, 90, 8, 20), input.getValue().target());
        TargetingGeometry.Aim expected = TargetingGeometry.calculate(new TargetingGeometry.Snapshot(
                400, 300, 800, 600, 100, 0, 600, 300, 90, 8, 4, 100,
                200, 14.6, 34 / 46.0 * 2 - 1, 8));
        assertEquals(expected.aimAngle(), input.getValue().legacyAim().direction());
        assertEquals(Math.hypot(expected.x() - 400, expected.y() - 300), input.getValue().legacyAim().range());
        assertEquals(1.8, input.getValue().legacyAim().power());
        assertNotEquals(expected.baseAngle(), expected.aimAngle());
        assertTrue(waves.isEmpty(), "a requested shot is not a confirmed passage wave");
        BulletState bullet = mock(BulletState.class);
        when(bullet.getX()).thenReturn(401.); when(bullet.getY()).thenReturn(302.);
        when(bullet.getPower()).thenReturn(1.8); when(bullet.getSpeed()).thenReturn(14.6);
        BulletFiredEvent event = mock(BulletFiredEvent.class);
        when(event.getBullet()).thenReturn(bullet); when(event.getTurnNumber()).thenReturn(101);
        bot.onBulletFired(event);
        verify(gun).onFired(101, 401, 302, 1.8);
        assertEquals(false, TargetingCharacterizationTest.get(bot, "pendingDuelShot"));
        assertEquals(1, waves.size());
        Object wave = waves.get(0);
        assertEquals(expected.x(), TargetingCharacterizationTest.get(wave, "aimX"));
        assertEquals(expected.y(), TargetingCharacterizationTest.get(wave, "aimY"));
        assertSame(profile[5], TargetingCharacterizationTest.get(wave, "bins"));
        assertEquals(101, TargetingCharacterizationTest.get(wave, "fireTurn"));
        assertEquals(14.6, TargetingCharacterizationTest.get(wave, "bulletSpeed"));
        // A fresh observation at the wave's passage trains residual GF even with no hit callback.
        doReturn(120).when(bot).getTurnNumber();
        TargetingCharacterizationTest.set(enemy, "lastScanTurn", 120);
        TargetingCharacterizationTest.set(enemy, "x", 401 + 19 * 14.6);
        TargetingCharacterizationTest.set(enemy, "y", 302.);
        Method advance = AISurvivor.class.getDeclaredMethod("advanceTargetingWaves");
        advance.setAccessible(true); advance.invoke(bot);
        assertTrue(waves.isEmpty());
        assertEquals(1, java.util.Arrays.stream(profile[5]).sum());
        assertEquals(5, java.util.Arrays.stream(agent.classGF[0][5]).sum());
        assertEquals(0, ((RoundMetrics) TargetingCharacterizationTest.get(bot, "metrics")).bulletHits,
                "GF passage labels are not actual bullet-hit labels");
        Path checkpoint = temporary.resolve("confirmed-passage.dat");
        ModelStore.save(checkpoint, agent);
        DQNAgent loaded = new DQNAgent(14, 6, config);
        assertTrue(ModelStore.load(checkpoint, loaded));
        assertArrayEquals(agent.classGF[0][5], loaded.classGF[0][5],
                "confirmed passage class evidence uses the existing checkpoint format");
    }

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
