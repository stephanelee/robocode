import static org.junit.jupiter.api.Assertions.*;
import dev.robocode.tankroyale.botapi.BotInfo;
import dev.robocode.tankroyale.botapi.BulletState;
import dev.robocode.tankroyale.botapi.events.BulletFiredEvent;
import dev.robocode.tankroyale.botapi.events.RoundStartedEvent;
import java.lang.reflect.Field;
import org.junit.jupiter.api.Test;

/** Invokes the real callback before any tick, without starting a bot or server. */
class RoundLifecycleTest {
    private static Field field(String name) throws Exception {
        Field f = KeelHelm.class.getDeclaredField(name); f.setAccessible(true); return f;
    }

    @Test void roundCallbackNeedsNoLiveTickAndClearsPreviousMeleeRound() throws Exception {
        var bot = new KeelHelm(BotInfo.fromFile("bot/KeelHelmCrossfire.json"));
        assertThrows(RuntimeException.class, bot::getEnemyCount, "no current tick exists in this fixture");
        var melee = (MeleePolicy)field("melee").get(bot);
        for (int round = 1; round <= 3; round++) {
            melee.activate(); melee.observe(7, 10, 100, 100, 0, 8, 100); melee.died(8);
            for (String name : new String[]{"meleeTurns","meleeShots","duelTransitions"}) field(name).setInt(bot,99);
            for (String name : new String[]{"alternateRequests","alternateAccepted","alternateConfirmed"}) field(name).setInt(bot,99);
            field("pendingAlternate").set(bot, new MeleeFire.Request(11, 7, 1.6, 0));
            int nextRound = round;
            assertDoesNotThrow(() -> bot.onRoundStarted(new RoundStartedEvent(nextRound)));
            assertTrue(melee.opponents().isEmpty()); assertFalse(melee.active()); assertFalse(melee.isDead(8));
            for (String name : new String[]{"meleeTurns","meleeShots","duelTransitions"}) assertEquals(0,field(name).getInt(bot));
            for (String name : new String[]{"alternateRequests","alternateAccepted","alternateConfirmed"}) assertEquals(0,field(name).getInt(bot));
            assertNull(field("pendingAlternate").get(bot));
            assertEquals(round,field("round").getInt(bot));
            melee.activate(); assertTrue(melee.active(), "first live melee tick can activate after reset");
        }
    }

    @Test void realCallbackConfirmsOnceAndLeavesMismatchesUnknown() throws Exception {
        var bot = new KeelHelm(BotInfo.fromFile("bot/KeelHelmCrossfire.json"));
        var melee = (MeleePolicy)field("melee").get(bot);
        bot.onRoundStarted(new RoundStartedEvent(1));
        melee.activate();
        field("pendingAlternate").set(bot, new MeleeFire.Request(11, 7, 1.6, 0));
        var aligned = new BulletState(1, 2, 1.6, 400, 300, 0, null);
        bot.onBulletFired(new BulletFiredEvent(10, aligned));
        assertEquals(0, field("alternateConfirmed").getInt(bot));
        bot.onBulletFired(new BulletFiredEvent(11, aligned));
        bot.onBulletFired(new BulletFiredEvent(11, aligned));
        assertEquals(1, field("alternateConfirmed").getInt(bot));
        field("pendingAlternate").set(bot, new MeleeFire.Request(12, 7, 1.6, 90));
        bot.onBulletFired(new BulletFiredEvent(12, aligned));
        assertNull(field("pendingAlternate").get(bot));
        assertEquals(1, field("alternateConfirmed").getInt(bot));
    }
}
